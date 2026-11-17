# vision.py -VERSION 3
import os
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class FinderCandidate:
    pos: tuple
    score: float
    contour: np.ndarray
    area_ratio: float
    has_child: bool = False
    dist: float = 0.0  # 距離評価用フィールドを追加


class VisionProcessor:
    """DonutCodeの画像処理モジュール
    ファインダパタン・アライメントパタンの検出と、透視変換による正規化を行います。
    """

    def __init__(self, config=None, target_side=500):
        self.config = config
        self.grid_size = config.GRID_SIZE if config else 27
        self.target_side = target_side

    # =======================================================
    #  1Dスキャンライン (ZBar風) 検出モジュール
    # =======================================================
    def _check_ratio(self, state_count):
        total_width = sum(state_count)
        if total_width < 7:
            return False
        module_size = total_width / 7.0
        max_variance = module_size * 0.7
        return (
            abs(module_size - state_count[0]) < max_variance
            and abs(module_size - state_count[1]) < max_variance
            and abs(3.0 * module_size - state_count[2]) < 3.0 * max_variance
            and abs(module_size - state_count[3]) < max_variance
            and abs(module_size - state_count[4]) < max_variance
        )

    def _cross_check_vertical(self, thresh, center_x, center_y):
        h, w = thresh.shape
        state_count = [0, 0, 0, 0, 0]
        y = center_y
        while y >= 0 and thresh[y, center_x] == 255:
            state_count[2] += 1
            y -= 1
        if y < 0:
            return False
        while y >= 0 and thresh[y, center_x] == 0:
            state_count[1] += 1
            y -= 1
        if y < 0:
            return False
        while y >= 0 and thresh[y, center_x] == 255:
            state_count[0] += 1
            y -= 1

        y = center_y + 1
        while y < h and thresh[y, center_x] == 255:
            state_count[2] += 1
            y += 1
        if y == h:
            return False
        while y < h and thresh[y, center_x] == 0:
            state_count[3] += 1
            y += 1
        if y == h:
            return False
        while y < h and thresh[y, center_x] == 255:
            state_count[4] += 1
            y += 1
        return self._check_ratio(state_count)

    def _cluster_centers(self, centers):
        clusters = []
        for cx, cy in centers:
            found = False
            for cluster in clusters:
                dist = np.hypot(cluster["x"] - cx, cluster["y"] - cy)
                if dist < 15:
                    cnt = cluster["count"]
                    cluster["x"] = (cluster["x"] * cnt + cx) / (cnt + 1)
                    cluster["y"] = (cluster["y"] * cnt + cy) / (cnt + 1)
                    cluster["count"] += 1
                    found = True
                    break
            if not found:
                clusters.append({"x": cx, "y": cy, "count": 1})
        return [(c["x"], c["y"]) for c in clusters]

    def _scan_finder_patterns_1d(self, thresh):
        h, w = thresh.shape
        centers = []
        skip = max(1, h // 200)
        for y in range(0, h, skip):
            state_count = [0, 0, 0, 0, 0]
            current_state = 0
            for x in range(w):
                is_black = thresh[y, x] == 255
                if is_black:
                    if current_state % 2 == 1:
                        current_state += 1
                    state_count[current_state] += 1
                else:
                    if current_state == 0 and state_count[0] == 0:
                        continue
                    if current_state % 2 == 0:
                        if current_state == 4:
                            if self._check_ratio(state_count):
                                center_x = (
                                    x - state_count[4] - state_count[3] - state_count[2] / 2.0
                                )
                                if self._cross_check_vertical(thresh, int(center_x), y):
                                    centers.append((center_x, y))
                            state_count = [0, 0, 0, 0, 0]
                            current_state = 0
                        else:
                            current_state += 1
                            state_count[current_state] += 1
                    else:
                        state_count[current_state] += 1
        return self._cluster_centers(centers)

    # =======================================================
    #  輪郭ベース検出モジュール
    # =======================================================
    def _get_square_score(self, contour):
        """輪郭の正方形度合いを計算 (高いほど正方形に近い)"""
        rect = cv2.minAreaRect(contour)
        w, h = rect[1]
        if w == 0 or h == 0:
            return 0.0
        aspect_ratio = max(w, h) / min(w, h)
        return max(0.0, 100.0 - abs(aspect_ratio - 1.0) * 100.0)

    def _get_finder_candidates(self, contours, hierarchy, strict_mode=True):
        candidates = []
        if hierarchy is None:
            return candidates

        for i in range(len(contours)):
            c1_idx = hierarchy[0][i][2]
            if c1_idx != -1:
                c2_idx = hierarchy[0][c1_idx][2]
                if c2_idx != -1:
                    c = contours[i]
                    area = cv2.contourArea(c)
                    if area < 50:
                        continue

                    area_child = cv2.contourArea(contours[c1_idx])
                    area_ratio = area_child / area if area > 0 else 0.0

                    score = self._get_square_score(c)
                    if strict_mode and score < 70:
                        continue

                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
                        if not any(
                            np.hypot(cx - cand.pos[0], cy - cand.pos[1]) < 15
                            for cand in candidates
                        ):
                            candidates.append(
                                FinderCandidate(
                                    pos=(cx, cy),
                                    score=score,
                                    contour=c,
                                    area_ratio=area_ratio,
                                    has_child=True,
                                )
                            )
        # スコアが高い順にソートして上位候補を優先できるようにする
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates

    def _find_alignment_pattern(self, contours, hierarchy, TL, TR, BL):
        """ファインダ3点からアライメントパターンの座標を高精度に探索"""
        if contours is None or hierarchy is None:
            return None
        if not (self.config and hasattr(self.config, "ALIGNMENT_POS")):
            return None

        gs = self.grid_size
        ax, ay = self.config.ALIGNMENT_POS
        a_center_x, a_center_y = ax + 2.5, ay + 2.5

        # ファインダ中心(3.5, 3.5)からのグリッド距離比率でアライメント予測位置を計算
        span = gs - 7.0  # 23.5 - 3.5 = 20.0
        ratio_x = (a_center_x - 3.5) / span
        ratio_y = (a_center_y - 3.5) / span
        expected_align_pos = TL + ratio_x * (TR - TL) + ratio_y * (BL - TL)

        dist_x_px = np.linalg.norm(TR - TL)
        cell_size = dist_x_px / span
        search_radius = cell_size * 5.0

        # 5x5アライメントパターンの外枠面積期待値
        expected_area = (cell_size * 5.0) ** 2

        candidates = []
        for i, c in enumerate(contours):
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]

            dist = float(np.hypot(cx - expected_align_pos[0], cy - expected_align_pos[1]))
            if dist > search_radius:
                continue

            area = cv2.contourArea(c)
            if not (0.3 * expected_area < area < 2.2 * expected_area):
                continue

            score = self._get_square_score(c)
            if score < 60:
                continue

            # 子輪郭（内側の白枠）が存在するかチェック（-1の時は参照しないよう修正）
            child_idx = hierarchy[0][i][2]
            has_child = child_idx != -1
            if has_child:
                child_area = cv2.contourArea(contours[child_idx])
                area_ratio = child_area / (area + 1e-5)
            else:
                area_ratio = 0.0

            candidates.append(
                FinderCandidate(
                    pos=(cx, cy),
                    score=score,
                    contour=c,
                    area_ratio=area_ratio,
                    has_child=has_child,
                    dist=dist,
                )
            )

        if not candidates:
            return None

        # 子輪郭を持つものを優先し、距離が近く正方形スコアが高いものを選択
        best = min(
            candidates,
            key=lambda x: (0 if x.has_child else 1, x.dist / (x.score + 1.0)),
        )
        return best.pos

    # =======================================================
    #  メイン処理 (process)
    # =======================================================
    def process(self, img_path, debug_mode=False):
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"画像 '{img_path}' が読み込めません。")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        thresh1 = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
        )
        thresh2 = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 5
        )
        _, thresh3 = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )
        thresholds = [
            ("Adaptive-21", thresh1),
            ("Adaptive-41", thresh2),
            ("OTSU", thresh3),
        ]

        best_centers = None
        best_contours = None
        best_hierarchy = None

        for _, thresh in thresholds:
            contours, hierarchy = cv2.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            candidates = self._get_finder_candidates(contours, hierarchy, strict_mode=True)
            if len(candidates) >= 3:
                best_centers = [c.pos for c in candidates]
                best_contours, best_hierarchy = contours, hierarchy
                break

            candidates = self._get_finder_candidates(contours, hierarchy, strict_mode=False)
            if len(candidates) >= 3:
                best_centers = [c.pos for c in candidates]
                best_contours, best_hierarchy = contours, hierarchy
                break

        if not best_centers or len(best_centers) < 3:
            for _, thresh in thresholds:
                centers = self._scan_finder_patterns_1d(thresh)
                if len(centers) >= 3:
                    best_centers = centers
                    best_contours, best_hierarchy = None, None
                    break

        if not best_centers or len(best_centers) < 3:
            return self._fallback_crop(img, thresh1)

        # 3つのファインダの位置関係から TL, TR, BL を特定
        pts = np.array(best_centers[:3], dtype=np.float32)
        d01 = np.linalg.norm(pts[0] - pts[1])
        d12 = np.linalg.norm(pts[1] - pts[2])
        d02 = np.linalg.norm(pts[0] - pts[2])

        # 斜辺（最も長い辺）の対角にある頂点が TL
        max_d = max(d01, d12, d02)
        if max_d == d01:
            tl_idx, p1_idx, p2_idx = 2, 0, 1
        elif max_d == d12:
            tl_idx, p1_idx, p2_idx = 0, 1, 2
        else:
            tl_idx, p1_idx, p2_idx = 1, 0, 2

        TL = pts[tl_idx]
        v1 = pts[p1_idx] - TL
        v2 = pts[p2_idx] - TL

        # 2D外積 (v1.x * v2.y - v1.y * v2.x) で画像座標系(Y軸下向き)のTRとBLを判別
        cross_z = v1[0] * v2[1] - v1[1] * v2[0]
        if cross_z > 0:
            TR, BL = pts[p1_idx], pts[p2_idx]
        else:
            TR, BL = pts[p2_idx], pts[p1_idx]

        estimated_BR = TR + BL - TL
        actual_alignment = self._find_alignment_pattern(
            best_contours, best_hierarchy, TL, TR, BL
        )

        gs, ts = self.grid_size, self.target_side

        if debug_mode:
            os.makedirs("sample-result/debug", exist_ok=True)
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            debug_img = img.copy()

            cv2.circle(debug_img, tuple(map(int, TL)), 4, (0, 0, 255), -1)  # TL: 赤
            cv2.circle(debug_img, tuple(map(int, TR)), 4, (0, 255, 0), -1)  # TR: 緑
            cv2.circle(debug_img, tuple(map(int, BL)), 4, (255, 0, 0), -1)  # BL: 青

            if actual_alignment:
                cv2.circle(debug_img, tuple(map(int, actual_alignment)), 4, (255, 0, 255), -1)
                poly_pts = [TL, TR, actual_alignment, BL]
            else:
                cv2.circle(debug_img, tuple(map(int, estimated_BR)), 4, (255, 255, 0), -1)
                poly_pts = [TL, TR, estimated_BR, BL]

            cv2.polylines(debug_img, [np.int32(poly_pts)], True, (0, 255, 255), 2)
            cv2.imwrite(f"sample-result/debug/{base_name}_01_anchors.png", debug_img)

        # 透視変換
        if actual_alignment:
            ax, ay = self.config.ALIGNMENT_POS
            a_center_x, a_center_y = ax + 2.5, ay + 2.5
            src_pts = np.float32([TL, TR, BL, actual_alignment])
            dst_pts = np.float32([
                [3.5 * (ts / gs), 3.5 * (ts / gs)],
                [(gs - 3.5) * (ts / gs), 3.5 * (ts / gs)],
                [3.5 * (ts / gs), (gs - 3.5) * (ts / gs)],
                [a_center_x * (ts / gs), a_center_y * (ts / gs)],
            ])
        else:
            src_pts = np.float32([TL, TR, BL, estimated_BR])
            dst_pts = np.float32([
                [3.5 * (ts / gs), 3.5 * (ts / gs)],
                [(gs - 3.5) * (ts / gs), 3.5 * (ts / gs)],
                [3.5 * (ts / gs), (gs - 3.5) * (ts / gs)],
                [(gs - 3.5) * (ts / gs), (gs - 3.5) * (ts / gs)],
            ])

        M_persp = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(img, M_persp, (ts, ts))

        if debug_mode:
            cv2.imwrite(f"sample-result/debug/{base_name}_02_warped.png", warped)

        return warped

    def _fallback_crop(self, img, thresh):
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ts = self.target_side
        if not contours:
            return cv2.resize(img, (ts, ts))
        rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
        box = cv2.boxPoints(rect)
        s, diff = box.sum(axis=1), np.diff(box, axis=1)
        rect_pts = np.float32([
            box[np.argmin(s)],
            box[np.argmin(diff)],
            box[np.argmax(s)],
            box[np.argmax(diff)],
        ])
        dst_pts = np.float32([[0, 0], [ts, 0], [ts, ts], [0, ts]])
        return cv2.warpPerspective(img, cv2.getPerspectiveTransform(rect_pts, dst_pts), (ts, ts))
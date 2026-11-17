# vision.py -VERSION 4
import os
from dataclasses import dataclass
from itertools import combinations
import cv2
import numpy as np


@dataclass
class FinderCandidate:
    pos: tuple
    score: float
    contour: np.ndarray
    area: float
    area_ratio: float
    has_child: bool = False
    dist: float = 0.0


class VisionProcessor:
    """DonutCodeの高精度画像処理モジュール (対ブレ・ピンボケ強化版)"""

    def __init__(self, config=None, target_side=500):
        self.config = config
        self.grid_size = config.GRID_SIZE if config else 27
        self.target_side = target_side

    # =======================================================
    #  前処理・画像強調モジュール (ブレ・ボケ対策)
    # =======================================================
    def _generate_preprocessed_binaries(self, gray):
        """ボケや照明ムラに対応するため、複数の強調・二値化画像を生成"""
        h, w = gray.shape
        short_side = min(h, w)

        # 1. CLAHEによる局所コントラスト改善
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 2. アンシャープマスキング (ピンボケ・ブレで鈍ったエッジを立たせる)
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        sharpened = cv2.addWeighted(enhanced, 1.8, gaussian, -0.8, 0)

        # 画像サイズに比例した適応的ブロックサイズ (奇数に丸める)
        b_small = max(15, (short_side // 12) | 1)
        b_large = max(31, (short_side // 6) | 1)

        binaries = []
        # 通常グレースケールでの二値化
        binaries.append((
            "OTSU",
            cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1],
        ))
        binaries.append((
            f"Adapt-{b_large}",
            cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, b_large, 4
            ),
        ))

        # 鮮鋭化画像での二値化 (ピンボケ時に極めて有効)
        binaries.append((
            "Sharp-OTSU",
            cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1],
        ))
        binaries.append((
            f"Sharp-Adapt-{b_large}",
            cv2.adaptiveThreshold(
                sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, b_large, 3
            ),
        ))
        binaries.append((
            f"Sharp-Adapt-{b_small}",
            cv2.adaptiveThreshold(
                sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, b_small, 3
            ),
        ))

        return binaries, sharpened

    def _enhance_warped_output(self, warped_bgr):
        """decoder.pyの大津の二値化でビット化けしないよう、出力画像を局所正規化・鮮鋭化する"""
        gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)

        # 照明ムラを除去してコントラストを均一化
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6))
        norm_gray = clahe.apply(gray)

        # セル境界のボケを解消するシャープニング
        blur = cv2.GaussianBlur(norm_gray, (0, 0), 1.5)
        sharp_gray = cv2.addWeighted(norm_gray, 1.6, blur, -0.6, 0)

        return cv2.cvtColor(sharp_gray, cv2.COLOR_GRAY2BGR)

    # =======================================================
    #  1Dスキャンライン (ZBar風) 検出モジュール
    # =======================================================
    def _check_ratio(self, state_count):
        total_width = sum(state_count)
        if total_width < 7:
            return False
        module_size = total_width / 7.0
        # ブレを考慮して許容誤差を0.7から0.8に緩和
        max_variance = module_size * 0.8
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
        # スキャンラインが多くヒットした信頼度の高いクラスタ順にソート
        clusters.sort(key=lambda c: c["count"], reverse=True)
        return [(c["x"], c["y"]) for c in clusters]

    def _scan_finder_patterns_1d(self, thresh):
        h, w = thresh.shape
        centers = []
        skip = max(1, h // 250)
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
    #  輪郭ベース検出＆幾何学フィルタリングモジュール
    # =======================================================
    def _get_square_score(self, contour):
        rect = cv2.minAreaRect(contour)
        w, h = rect[1]
        if w == 0 or h == 0:
            return 0.0
        aspect_ratio = max(w, h) / min(w, h)
        return max(0.0, 100.0 - abs(aspect_ratio - 1.0) * 100.0)

    def _get_centroid(self, contour):
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None
        return (M["m10"] / M["m00"], M["m01"] / M["m00"])

    def _get_finder_candidates(self, contours, hierarchy, strict_mode=True):
        candidates = []
        if hierarchy is None:
            return candidates

        for i in range(len(contours)):
            c1_idx = hierarchy[0][i][2]
            if c1_idx == -1:
                continue
            c2_idx = hierarchy[0][c1_idx][2]
            if c2_idx == -1:
                continue

            c_outer = contours[i]
            c_mid = contours[c1_idx]
            c_inner = contours[c2_idx]

            area_outer = cv2.contourArea(c_outer)
            if area_outer < 36:
                continue

            # 理論面積比: 外枠(49) : 内枠(25) : 中心(9)
            area_mid = cv2.contourArea(c_mid)
            area_inner = cv2.contourArea(c_inner)
            ratio_mid = area_mid / area_outer
            ratio_inner = area_inner / area_outer

            # 明らかなノイズ輪郭を面積比で除外
            if strict_mode and not (0.20 < ratio_mid < 0.85 and 0.04 < ratio_inner < 0.45):
                continue

            score = self._get_square_score(c_outer)
            if strict_mode and score < 65:
                continue
            if not strict_mode and score < 40:
                continue

            # 3階層(外・中・内)の重心を平均してサブピクセル精度化 (ブレによる歪み相殺)
            p0 = self._get_centroid(c_outer)
            p1 = self._get_centroid(c_mid)
            p2 = self._get_centroid(c_inner)
            if not p0:
                continue

            valid_pts = [p for p in (p0, p1, p2) if p is not None]
            cx = sum(p[0] for p in valid_pts) / len(valid_pts)
            cy = sum(p[1] for p in valid_pts) / len(valid_pts)

            if not any(np.hypot(cx - cand.pos[0], cy - cand.pos[1]) < 12 for cand in candidates):
                candidates.append(
                    FinderCandidate(
                        pos=(cx, cy),
                        score=score,
                        contour=c_outer,
                        area=area_outer,
                        area_ratio=ratio_mid,
                        has_child=True,
                    )
                )

        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates

    def _select_best_finder_triplet(self, candidates):
        """候補が3つ以上ある場合、直角二等辺三角形の幾何学的制約から最も確からしい3点を選出"""
        if len(candidates) < 3:
            return None

        # 上位10候補に絞って組み合わせ探索 (計算量抑制)
        pool = candidates[:10]
        best_triplet = None
        best_cost = float("inf")

        for c0, c1, c2 in combinations(pool, 3):
            pts = np.array([c0.pos, c1.pos, c2.pos], dtype=np.float32)
            areas = [c0.area, c1.area, c2.area]

            # 面積のばらつきチェック (3つのファインダはほぼ同じサイズのはず)
            if max(areas) > 0 and (min(areas) / max(areas)) < 0.25:
                continue

            d01 = np.linalg.norm(pts[0] - pts[1])
            d12 = np.linalg.norm(pts[1] - pts[2])
            d02 = np.linalg.norm(pts[0] - pts[2])

            max_d = max(d01, d12, d02)
            if max_d < 20:
                continue

            if max_d == d01:
                tl_idx, p1_idx, p2_idx = 2, 0, 1
            elif max_d == d12:
                tl_idx, p1_idx, p2_idx = 0, 1, 2
            else:
                tl_idx, p1_idx, p2_idx = 1, 0, 2

            TL = pts[tl_idx]
            v1 = pts[p1_idx] - TL
            v2 = pts[p2_idx] - TL
            len1 = np.linalg.norm(v1)
            len2 = np.linalg.norm(v2)
            if len1 == 0 or len2 == 0:
                continue

            # 1. 2辺の長さの比率誤差 (正方形なので 1.0 に近いはず)
            length_diff = abs(len1 - len2) / max(len1, len2)
            if length_diff > 0.45:  # パースを考慮しても45%以上違う場合は除外
                continue

            # 2. 直角性誤差 (cosθ = 内積 / (len1 * len2) が 0 に近いはず)
            cos_theta = abs(np.dot(v1, v2) / (len1 * len2))
            if cos_theta > 0.45:
                continue

            # 3. 面積の一貫性誤差
            area_diff = 1.0 - (min(areas) / max(areas)) if max(areas) > 0 else 0.0

            # トータルコストが最小の組を選ぶ
            cost = length_diff * 2.0 + cos_theta * 3.0 + area_diff * 1.0
            if cost < best_cost:
                best_cost = cost
                # 外積でTRとBLを確定させて保存
                cross_z = v1[0] * v2[1] - v1[1] * v2[0]
                if cross_z > 0:
                    TR, BL = pts[p1_idx], pts[p2_idx]
                else:
                    TR, BL = pts[p2_idx], pts[p1_idx]
                best_triplet = (TL, TR, BL)

        return best_triplet

    def _find_alignment_pattern(self, contours, hierarchy, TL, TR, BL):
        """ファインダ3点からアライメントパターンの座標を高精度に探索"""
        if contours is None or hierarchy is None:
            return None
        if not (self.config and hasattr(self.config, "ALIGNMENT_POS")):
            return None

        gs = self.grid_size
        ax, ay = self.config.ALIGNMENT_POS
        a_center_x, a_center_y = ax + 2.5, ay + 2.5

        span = gs - 7.0
        ratio_x = (a_center_x - 3.5) / span
        ratio_y = (a_center_y - 3.5) / span
        expected_align_pos = TL + ratio_x * (TR - TL) + ratio_y * (BL - TL)

        dist_x_px = np.linalg.norm(TR - TL)
        cell_size = dist_x_px / span
        search_radius = cell_size * 4.5
        expected_area = (cell_size * 5.0) ** 2

        candidates = []
        for i, c in enumerate(contours):
            pt = self._get_centroid(c)
            if not pt:
                continue
            cx, cy = pt

            dist = float(np.hypot(cx - expected_align_pos[0], cy - expected_align_pos[1]))
            if dist > search_radius:
                continue

            area = cv2.contourArea(c)
            if not (0.25 * expected_area < area < 2.2 * expected_area):
                continue

            score = self._get_square_score(c)
            if score < 50:
                continue

            child_idx = hierarchy[0][i][2]
            has_child = child_idx != -1
            if has_child:
                # 子輪郭の重心もブレンドして精度向上
                c_pt = self._get_centroid(contours[child_idx])
                if c_pt:
                    cx, cy = (cx + c_pt[0]) / 2.0, (cy + c_pt[1]) / 2.0
                child_area = cv2.contourArea(contours[child_idx])
                area_ratio = child_area / (area + 1e-5)
            else:
                area_ratio = 0.0

            candidates.append(
                FinderCandidate(
                    pos=(cx, cy),
                    score=score,
                    contour=c,
                    area=area,
                    area_ratio=area_ratio,
                    has_child=has_child,
                    dist=dist,
                )
            )

        if not candidates:
            return None

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

        binaries, _ = self._generate_preprocessed_binaries(gray)

        triplet = None
        best_contours = None
        best_hierarchy = None

        # 1. 輪郭階層ベースでの探索 (Strict -> Loose)
        for strict in (True, False):
            for name, thresh in binaries:
                contours, hierarchy = cv2.findContours(
                    thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )
                candidates = self._get_finder_candidates(
                    contours, hierarchy, strict_mode=strict
                )
                if len(candidates) >= 3:
                    triplet = self._select_best_finder_triplet(candidates)
                    if triplet is not None:
                        best_contours, best_hierarchy = contours, hierarchy
                        if debug_mode:
                            print(f"[Debug] Finder検出成功 (Mode: Contour, Strict={strict}, Filter={name})")
                        break
            if triplet is not None:
                break

        # 2. 見つからない場合は1Dスキャンラインで探索
        if triplet is None:
            for name, thresh in binaries:
                centers = self._scan_finder_patterns_1d(thresh)
                if len(centers) >= 3:
                    dummy_cands = [
                        FinderCandidate(pos=p, score=80.0, contour=None, area=100.0, area_ratio=0.5)
                        for p in centers
                    ]
                    triplet = self._select_best_finder_triplet(dummy_cands)
                    if triplet is not None:
                        if debug_mode:
                            print(f"[Debug] Finder検出成功 (Mode: 1D-Scan, Filter={name})")
                        break

        # 3. それでも見つからない場合はフォールバック
        if triplet is None:
            if debug_mode:
                print("[Debug] Finder検出失敗 -> Fallback Cropを実行")
            warped = self._fallback_crop(img, binaries[0][1])
            return self._enhance_warped_output(warped)

        TL, TR, BL = triplet
        estimated_BR = TR + BL - TL
        actual_alignment = self._find_alignment_pattern(
            best_contours, best_hierarchy, TL, TR, BL
        )

        gs, ts = self.grid_size, self.target_side

        if debug_mode:
            os.makedirs("sample-result/debug", exist_ok=True)
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            debug_img = img.copy()

            cv2.circle(debug_img, tuple(map(int, TL)), 4, (0, 0, 255), -1)
            cv2.circle(debug_img, tuple(map(int, TR)), 4, (0, 255, 0), -1)
            cv2.circle(debug_img, tuple(map(int, BL)), 4, (255, 0, 0), -1)

            if actual_alignment:
                cv2.circle(debug_img, tuple(map(int, actual_alignment)), 4, (255, 0, 255), -1)
                poly_pts = [TL, TR, actual_alignment, BL]
            else:
                cv2.circle(debug_img, tuple(map(int, estimated_BR)), 4, (255, 255, 0), -1)
                poly_pts = [TL, TR, estimated_BR, BL]

            cv2.polylines(debug_img, [np.int32(poly_pts)], True, (0, 255, 255), 2)
            cv2.imwrite(f"sample-result/debug/{base_name}_01_anchors.png", debug_img)

        # 透視変換 (バイキュービック補間でボケを抑えてリサイズ)
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
        warped = cv2.warpPerspective(img, M_persp, (ts, ts), flags=cv2.INTER_CUBIC)

        # デコーダの二値化精度を上げるため、出力前にコントラスト・エッジ強調を実施
        warped_enhanced = self._enhance_warped_output(warped)

        if debug_mode:
            cv2.imwrite(f"sample-result/debug/{base_name}_02_warped.png", warped_enhanced)

        return warped_enhanced

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
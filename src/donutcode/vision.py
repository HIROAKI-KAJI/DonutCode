# vision.py -VERSION 4.1
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
    mid_contour: np.ndarray
    inner_contour: np.ndarray
    area: float
    area_ratio: float
    has_child: bool = False
    dist: float = 0.0


class VisionProcessor:
    """DonutCodeの高精度画像処理モジュール (v5.0: 3D射影歪み・ホモグラフィ推定強化版)"""

    def __init__(self, config=None, target_side=500):
        self.config = config
        self.grid_size = config.GRID_SIZE if config else 27
        self.target_side = target_side

    # =======================================================
    #  前処理・画像強調モジュール
    # =======================================================
    def _generate_preprocessed_binaries(self, gray):
        h, w = gray.shape
        short_side = min(h, w)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        sharpened = cv2.addWeighted(enhanced, 1.8, gaussian, -0.8, 0)

        b_small = max(15, (short_side // 12) | 1)
        b_large = max(31, (short_side // 6) | 1)

        binaries = [
            (
                "OTSU",
                cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1],
            ),
            (
                f"Adapt-{b_large}",
                cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, b_large, 4
                ),
            ),
            (
                "Sharp-OTSU",
                cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1],
            ),
            (
                f"Sharp-Adapt-{b_large}",
                cv2.adaptiveThreshold(
                    sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, b_large, 3
                ),
            ),
            (
                f"Sharp-Adapt-{b_small}",
                cv2.adaptiveThreshold(
                    sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, b_small, 3
                ),
            ),
        ]
        return binaries

    def _enhance_warped_output(self, warped_bgr):
        gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6))
        norm_gray = clahe.apply(gray)
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
        max_variance = module_size * 0.85
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
        clusters.sort(key=lambda c: c["count"], reverse=True)
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
    #  輪郭ベース検出＆射影幾何モジュール
    # =======================================================
    def _get_quad_score(self, contour):
        """射影変換やシアーでひしゃげた四角形でも高スコアを維持する評価関数"""
        area = cv2.contourArea(contour)
        if area <= 0:
            return 0.0
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0

        rect = cv2.minAreaRect(contour)
        w, h = rect[1]
        if w == 0 or h == 0:
            return 0.0
        aspect = max(w, h) / min(w, h)
        # 強い斜め射影でもアスペクト比2.2程度までは許容し、凸性(solidity)を重視する
        aspect_score = max(0.0, 100.0 - max(0.0, aspect - 1.2) * 60.0)
        return aspect_score * (solidity ** 2)

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
            if area_outer < 30:
                continue

            area_mid = cv2.contourArea(c_mid)
            area_inner = cv2.contourArea(c_inner)
            ratio_mid = area_mid / area_outer
            ratio_inner = area_inner / area_outer

            if strict_mode and not (0.15 < ratio_mid < 0.88 and 0.03 < ratio_inner < 0.50):
                continue

            score = self._get_quad_score(c_outer)
            if strict_mode and score < 50:
                continue
            if not strict_mode and score < 30:
                continue

            p0 = self._get_centroid(c_outer)
            p1 = self._get_centroid(c_mid)
            p2 = self._get_centroid(c_inner)
            if not p0:
                continue

            # パース歪みによる外枠重心のズレを抑えるため、最小の3x3内枠(p2)に最大の重みを置く
            if p1 and p2:
                cx = (1.0 * p0[0] + 2.0 * p1[0] + 4.0 * p2[0]) / 7.0
                cy = (1.0 * p0[1] + 2.0 * p1[1] + 4.0 * p2[1]) / 7.0
            else:
                valid_pts = [p for p in (p0, p1, p2) if p is not None]
                cx = sum(p[0] for p in valid_pts) / len(valid_pts)
                cy = sum(p[1] for p in valid_pts) / len(valid_pts)

            if not any(np.hypot(cx - cand.pos[0], cy - cand.pos[1]) < 12 for cand in candidates):
                candidates.append(
                    FinderCandidate(
                        pos=(cx, cy),
                        score=score,
                        contour=c_outer,
                        mid_contour=c_mid,
                        inner_contour=c_inner,
                        area=area_outer,
                        area_ratio=ratio_mid,
                        has_child=True,
                    )
                )

        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates

    def _select_best_finder_triplet(self, candidates):
        """候補から最も妥当なファインダ3個 (TL_cand, TR_cand, BL_cand) を選出"""
        if len(candidates) < 3:
            return None

        pool = candidates[:10]
        best_triplet = None
        best_cost = float("inf")

        for c0, c1, c2 in combinations(pool, 3):
            pts = np.array([c0.pos, c1.pos, c2.pos], dtype=np.float32)
            areas = [c0.area, c1.area, c2.area]

            # 強い射影変換では奥と手前で面積が最大5倍程度変わるため閾値を0.15に緩和
            if max(areas) > 0 and (min(areas) / max(areas)) < 0.15:
                continue

            d01 = np.linalg.norm(pts[0] - pts[1])
            d12 = np.linalg.norm(pts[1] - pts[2])
            d02 = np.linalg.norm(pts[0] - pts[2])

            max_d = max(d01, d12, d02)
            if max_d < 20:
                continue

            cands_list = [c0, c1, c2]
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

            length_diff = abs(len1 - len2) / max(len1, len2)
            if length_diff > 0.55:
                continue

            cos_theta = abs(np.dot(v1, v2) / (len1 * len2))
            if cos_theta > 0.60:
                continue

            area_diff = 1.0 - (min(areas) / max(areas)) if max(areas) > 0 else 0.0
            cost = length_diff * 2.0 + cos_theta * 3.0 + area_diff * 0.8

            if cost < best_cost:
                best_cost = cost
                cross_z = v1[0] * v2[1] - v1[1] * v2[0]
                if cross_z > 0:
                    tr_idx, bl_idx = p1_idx, p2_idx
                else:
                    tr_idx, bl_idx = p2_idx, p1_idx
                best_triplet = (cands_list[tl_idx], cands_list[tr_idx], cands_list[bl_idx])

        return best_triplet

    def _extract_4_corners(self, contour, center_tl, vec_u, vec_v):
        """ファインダ輪郭を4頂点に近似し、コード座標系の (左上, 右上, 右下, 左下) の順に並べる"""
        if contour is None:
            return None
        hull = cv2.convexHull(contour)
        peri = cv2.arcLength(hull, True)

        approx = None
        for eps in np.linspace(0.02, 0.15, 10):
            cand = cv2.approxPolyDP(hull, eps * peri, True)
            if len(cand) == 4:
                approx = cand.reshape(4, 2).astype(np.float32)
                break

        if approx is None:
            rect = cv2.minAreaRect(hull)
            approx = cv2.boxPoints(rect).astype(np.float32)

        # ファインダ中心からの相対ベクトルを (vec_u, vec_v) 基底に投影して4隅をソート
        c = np.mean(approx, axis=0)
        rel = approx - c
        u_norm = vec_u / (np.linalg.norm(vec_u) + 1e-6)
        v_norm = vec_v / (np.linalg.norm(vec_v) + 1e-6)

        proj_u = rel @ u_norm
        proj_v = rel @ v_norm

        # TL: u<0, v<0 / TR: u>0, v<0 / BR: u>0, v>0 / BL: u<0, v>0
        tl_pt = approx[np.argmin(proj_u + proj_v)]
        br_pt = approx[np.argmax(proj_u + proj_v)]
        tr_pt = approx[np.argmax(proj_u - proj_v)]
        bl_pt = approx[np.argmin(proj_u - proj_v)]

        return np.array([tl_pt, tr_pt, br_pt, bl_pt], dtype=np.float32)

    def _estimate_initial_homography(self, cand_TL, cand_TR, cand_BL):
        """
        3つのファインダの外枠4頂点（計12点）と中心3点（計15点）を使って、
        3D斜め射影歪みを含んだ初期ホモグラフィ行列 H (Grid座標 -> 画像座標) を推定する
        """
        gs = float(self.grid_size)
        TL = np.array(cand_TL.pos, dtype=np.float32)
        TR = np.array(cand_TR.pos, dtype=np.float32)
        BL = np.array(cand_BL.pos, dtype=np.float32)

        vec_u = TR - TL
        vec_v = BL - TL

        grid_pts = [
            [3.5, 3.5],
            [gs - 3.5, 3.5],
            [3.5, gs - 3.5],
        ]
        img_pts = [TL, TR, BL]

        # 各ファインダの外枠7x7の4隅 (グリッド座標)
        finder_configs = [
            (cand_TL.contour, [[0.0, 0.0], [7.0, 0.0], [7.0, 7.0], [0.0, 7.0]]),
            (cand_TR.contour, [[gs - 7.0, 0.0], [gs, 0.0], [gs, 7.0], [gs - 7.0, 7.0]]),
            (cand_BL.contour, [[0.0, gs - 7.0], [7.0, gs - 7.0], [7.0, gs], [0.0, gs]]),
        ]

        for contour, g_corners in finder_configs:
            corners = self._extract_4_corners(contour, TL, vec_u, vec_v)
            if corners is not None:
                grid_pts.extend(g_corners)
                img_pts.extend(corners)

        grid_pts = np.array(grid_pts, dtype=np.float32)
        img_pts = np.array(img_pts, dtype=np.float32)

        if len(grid_pts) >= 7:
            H, _ = cv2.findHomography(grid_pts, img_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
            if H is not None:
                return H

        # フォールバック（アフィン近似）
        br_affine = TR + BL - TL
        src_4 = np.float32([[3.5, 3.5], [gs - 3.5, 3.5], [3.5, gs - 3.5], [gs - 3.5, gs - 3.5]])
        dst_4 = np.float32([TL, TR, BL, br_affine])
        return cv2.getPerspectiveTransform(src_4, dst_4)

    def _project_point(self, H, gx, gy):
        pt = np.array([[[gx, gy]]], dtype=np.float32)
        proj = cv2.perspectiveTransform(pt, H)[0, 0]
        return proj

    def _find_alignment_pattern(self, gray, contours, hierarchy, H_grid_to_img, TL, TR, BL):
        """
        1. 初期ホモグラフィ H で予測した位置の周辺から輪郭ベースでアライメントを探索
        2. 輪郭で見つからない、または精度を高めるため、仮ワープ画像上で5x5テンプレートマッチングを実行
        """
        if not (self.config and hasattr(self.config, "ALIGNMENT_POS")):
            return None

        gs = float(self.grid_size)
        ax, ay = self.config.ALIGNMENT_POS
        a_cx, a_cy = ax + 2.5, ay + 2.5

        # ホモグラフィによる台形歪みを考慮した予測座標
        expected_pos = self._project_point(H_grid_to_img, a_cx, a_cy)

        # 局所スケール（右下付近での1セルあたりのピクセル幅）を計算
        p_right = self._project_point(H_grid_to_img, a_cx + 1.0, a_cy)
        p_down = self._project_point(H_grid_to_img, a_cx, a_cy + 1.0)
        local_cell_px = float((np.linalg.norm(p_right - expected_pos) + np.linalg.norm(p_down - expected_pos)) * 0.5)
        local_cell_px = max(1.5, local_cell_px)

        search_radius = local_cell_px * 5.5
        expected_area = (local_cell_px * 5.0) ** 2

        # Step 1: 輪郭階層からの探索
        candidates = []
        if contours is not None and hierarchy is not None:
            for i, c in enumerate(contours):
                pt = self._get_centroid(c)
                if not pt:
                    continue
                cx, cy = pt
                dist = float(np.hypot(cx - expected_pos[0], cy - expected_pos[1]))
                if dist > search_radius:
                    continue

                area = cv2.contourArea(c)
                if not (0.18 * expected_area < area < 2.8 * expected_area):
                    continue

                score = self._get_quad_score(c)
                if score < 35:
                    continue

                child_idx = hierarchy[0][i][2]
                has_child = child_idx != -1
                has_grandchild = False
                if has_child:
                    gc_idx = hierarchy[0][child_idx][2]
                    has_grandchild = gc_idx != -1
                    c_pt = self._get_centroid(contours[child_idx])
                    if c_pt:
                        cx, cy = 0.35 * cx + 0.65 * c_pt[0], 0.35 * cy + 0.65 * c_pt[1]
                    if has_grandchild:
                        gc_pt = self._get_centroid(contours[gc_idx])
                        if gc_pt:
                            cx, cy = 0.3 * cx + 0.7 * gc_pt[0], 0.3 * cy + 0.7 * gc_pt[1]

                # 3階層(黒枠->白枠->中心黒) または 2階層を持つものを最優先
                tier = 0 if has_grandchild else (1 if has_child else 2)
                candidates.append((tier, dist / (score + 1.0), (cx, cy)))

        if candidates:
            candidates.sort(key=lambda x: (x[0], x[1]))
            best_tier, _, best_pos = candidates[0]
            if best_tier <= 1:
                return np.array(best_pos, dtype=np.float32)

        # Step 2: 仮ワープ画像上での 5x5 アライメントパターン テンプレートマッチング
        # (パースで輪郭が切れていても正確に中心を捕捉できる)
        ts = self.target_side
        scale_grid_to_warp = ts / gs
        S_mat = np.array([
            [scale_grid_to_warp, 0, 0],
            [0, scale_grid_to_warp, 0],
            [0, 0, 1]
        ], dtype=np.float64)

        try:
            H_img_to_warp = S_mat @ np.linalg.inv(H_grid_to_img)
            pre_warped = cv2.warpPerspective(gray, H_img_to_warp, (ts, ts), flags=cv2.INTER_LINEAR)

            cell_w = ts / gs
            # 5x5 の理想的なアライメントテンプレート画像を生成 (黒0, 白255)
            tmpl_cells = np.array([
                [0,   0,   0,   0,   0],
                [0, 255, 255, 255,   0],
                [0, 255,   0, 255,   0],
                [0, 255, 255, 255,   0],
                [0,   0,   0,   0,   0],
            ], dtype=np.uint8)
            tmpl_px = int(round(5.0 * cell_w))
            tmpl_img = cv2.resize(tmpl_cells, (tmpl_px, tmpl_px), interpolation=cv2.INTER_NEAREST)

            # 探索ROI: 本来の(ax, ay)の周囲 ±4.5 セル
            margin_px = int(round(4.5 * cell_w))
            exp_x0 = int(round(ax * cell_w))
            exp_y0 = int(round(ay * cell_w))

            rx0 = max(0, exp_x0 - margin_px)
            ry0 = max(0, exp_y0 - margin_px)
            rx1 = min(ts, exp_x0 + tmpl_px + margin_px)
            ry1 = min(ts, exp_y0 + tmpl_px + margin_px)

            roi = pre_warped[ry0:ry1, rx0:rx1]
            if roi.shape[0] > tmpl_px and roi.shape[1] > tmpl_px:
                roi_norm = cv2.equalizeHist(roi)
                res = cv2.matchTemplate(roi_norm, tmpl_img, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val > 0.35:
                    found_warp_cx = rx0 + max_loc[0] + tmpl_px / 2.0
                    found_warp_cy = ry0 + max_loc[1] + tmpl_px / 2.0
                    # 元画像座標へ逆投影
                    H_warp_to_img = np.linalg.inv(H_img_to_warp)
                    pt_warp = np.array([[[found_warp_cx, found_warp_cy]]], dtype=np.float32)
                    refined_img_pt = cv2.perspectiveTransform(pt_warp, H_warp_to_img)[0, 0]
                    return refined_img_pt
        except np.linalg.LinAlgError:
            pass

        if candidates:
            return np.array(candidates[0][2], dtype=np.float32)
        return expected_pos

    # =======================================================
    #  メイン処理 (process)
    # =======================================================
    def process(self, img_path, debug_mode=False):
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"画像 '{img_path}' が読み込めません。")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        binaries = self._generate_preprocessed_binaries(gray)

        triplet_cands = None
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
                    triplet_cands = self._select_best_finder_triplet(candidates)
                    if triplet_cands is not None:
                        best_contours, best_hierarchy = contours, hierarchy
                        if debug_mode:
                            print(f"[Debug] Finder検出成功 (Mode: Contour, Strict={strict}, Filter={name})")
                        break
            if triplet_cands is not None:
                break

        # 2. 1Dスキャンラインで探索
        if triplet_cands is None:
            for name, thresh in binaries:
                centers = self._scan_finder_patterns_1d(thresh)
                if len(centers) >= 3:
                    dummy_cands = [
                        FinderCandidate(
                            pos=p, score=80.0, contour=None, mid_contour=None,
                            inner_contour=None, area=100.0, area_ratio=0.5
                        )
                        for p in centers
                    ]
                    triplet_cands = self._select_best_finder_triplet(dummy_cands)
                    if triplet_cands is not None:
                        if debug_mode:
                            print(f"[Debug] Finder検出成功 (Mode: 1D-Scan, Filter={name})")
                        break

        # 3. フォールバック
        if triplet_cands is None:
            if debug_mode:
                print("[Debug] Finder検出失敗 -> Fallback Cropを実行")
            warped = self._fallback_crop(img, binaries[0][1])
            return self._enhance_warped_output(warped)

        cand_TL, cand_TR, cand_BL = triplet_cands
        TL = np.array(cand_TL.pos, dtype=np.float32)
        TR = np.array(cand_TR.pos, dtype=np.float32)
        BL = np.array(cand_BL.pos, dtype=np.float32)

        # ファインダ外枠の4隅頂点群から初期ホモグラフィ行列を推定
        H_init = self._estimate_initial_homography(cand_TL, cand_TR, cand_BL)

        # 初期ホモグラフィとテンプレートマッチングを使って第4アンカー(Alignment)を特定
        actual_alignment = self._find_alignment_pattern(
            gray, best_contours, best_hierarchy, H_init, TL, TR, BL
        )

        gs, ts = self.grid_size, self.target_side

        if debug_mode:
            os.makedirs("sample-result/debug", exist_ok=True)
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            debug_img = img.copy()

            cv2.circle(debug_img, tuple(map(int, TL)), 4, (0, 0, 255), -1)
            cv2.circle(debug_img, tuple(map(int, TR)), 4, (0, 255, 0), -1)
            cv2.circle(debug_img, tuple(map(int, BL)), 4, (255, 0, 0), -1)

            if actual_alignment is not None:
                cv2.circle(debug_img, tuple(map(int, actual_alignment)), 4, (255, 0, 255), -1)
                poly_pts = [TL, TR, actual_alignment, BL]
            else:
                est_br = self._project_point(H_init, gs - 3.5, gs - 3.5)
                cv2.circle(debug_img, tuple(map(int, est_br)), 4, (255, 255, 0), -1)
                poly_pts = [TL, TR, est_br, BL]

            cv2.polylines(debug_img, [np.int32(poly_pts)], True, (0, 255, 255), 2)
            cv2.imwrite(f"sample-result/debug/{base_name}_01_anchors.png", debug_img)

        # 最終透視変換
        if actual_alignment is not None:
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
            est_br = self._project_point(H_init, gs - 3.5, gs - 3.5)
            src_pts = np.float32([TL, TR, BL, est_br])
            dst_pts = np.float32([
                [3.5 * (ts / gs), 3.5 * (ts / gs)],
                [(gs - 3.5) * (ts / gs), 3.5 * (ts / gs)],
                [3.5 * (ts / gs), (gs - 3.5) * (ts / gs)],
                [(gs - 3.5) * (ts / gs), (gs - 3.5) * (ts / gs)],
            ])

        M_persp = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(img, M_persp, (ts, ts), flags=cv2.INTER_CUBIC)
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
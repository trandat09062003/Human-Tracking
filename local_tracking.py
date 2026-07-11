"""
Human tracking với best_v5.pt
Pipeline: YOLO.predict → lọc FP (xe đạp/xe máy) → BoT-SORT → IDRecoverer → vẽ
Lọc TRƯỚC tracker để FP không bao giờ nhận ID.
"""

import cv2
import time
import logging
import os
from collections import OrderedDict

import numpy as np
import torch
import yaml
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.utils import IterableSimpleNamespace

# --- CONFIG ---
VIDEO_SOURCE = r"C:\Users\DELL\Videos\Screen Recordings\Screen Recording 2026-07-11 113805.mp4"
MODEL_PATH = "best_v5.pt"
TRACKER_YAML = "custom_tracker.yaml"

# Detect rộng hơn một chút, rồi siết bằng filter trước khi vào tracker
DETECT_CONF = 0.28
TRACK_CONF = 0.42          # hạ nhẹ để bắt người xa/mờ; vẫn trên FP xe đạp ~0.39
IOU = 0.35                 # NMS YOLO chặt hơn → ít box trùng trước tracker
IMGSZ = 640
MAX_DET = 50

# Hình dạng người đứng trên CCTV (aspect thật thường ~2.0–4.6; xe đạp ~0.8–1.5)
MIN_ASPECT = 1.70
MAX_ASPECT = 7.00
MIN_HEIGHT_RATIO = 0.045
MIN_BOX_AREA = 600
NEST_THRESH = 0.50         # box nhỏ nằm ≥50% trong box lớn → bỏ
OVERLAP_IOU = 0.40         # IoU ≥ ngưỡng → giữ box conf cao hơn
TRACK_OVERLAP_IOU = 0.45   # sau tracking: 2 ID chồng nhau → giữ ID già hơn

# Hiện track sớm hơn (giảm sót người mới xuất hiện)
MIN_HITS = 1

# IDRecoverer — bám ID mạnh khi che/miss; chỉ tách khi nhảy xa + khác appearance
RECOVER_MAX_FRAMES = 150         # ~5s @ 30fps
RECOVER_MAX_DIST_RATIO = 1.60
RECOVER_COST_THRESH = 0.72
RECOVER_MAX_HIST = 0.75
RECOVER_MIN_IOU = 0.0
MAX_JUMP_RATIO = 2.20

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Palette rõ, tương phản cao (BGR)
ID_COLORS = [
    (0, 220, 255),    # vàng
    (255, 180, 0),    # xanh dương sáng
    (80, 255, 80),    # xanh lá
    (255, 100, 255),  # hồng
    (0, 165, 255),    # cam
    (255, 255, 100),  # cyan nhạt
    (180, 105, 255),  # đỏ hồng
    (50, 255, 200),   # mint
]


def _id_color(sid: int):
    return ID_COLORS[(sid - 1) % len(ID_COLORS)]


def _draw_label(img, text, x, y, color, font_scale=0.55, thickness=1):
    """Nhãn gọn: nền đậm + chữ trắng, đặt phía trên box."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad_x, pad_y = 5, 3
    box_h = th + baseline + pad_y * 2
    box_w = tw + pad_x * 2
    y1 = max(box_h, y)
    x1 = max(0, x)
    x2 = min(img.shape[1] - 1, x1 + box_w)
    y2 = y1
    y0 = y1 - box_h
    cv2.rectangle(img, (x1, y0), (x2, y2), color, -1)
    cv2.putText(img, text, (x1 + pad_x, y2 - pad_y - baseline), font, font_scale, (20, 20, 20), thickness, cv2.LINE_AA)


def _draw_hud(img, lines):
    """HUD góc trên trái, nền mờ gọn."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.55, 1
    pad = 8
    gap = 6
    sizes = [cv2.getTextSize(t, font, scale, thick)[0] for t in lines]
    w = max(s[0] for s in sizes) + pad * 2
    h = sum(s[1] for s in sizes) + gap * (len(lines) - 1) + pad * 2
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (10 + w, 10 + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
    y = 10 + pad
    for text, (tw, th) in zip(lines, sizes):
        y += th
        cv2.putText(img, text, (10 + pad, y), font, scale, (240, 240, 240), thick, cv2.LINE_AA)
        y += gap


def _box_center(xyxy):
    return np.array([(xyxy[0] + xyxy[2]) * 0.5, (xyxy[1] + xyxy[3]) * 0.5], dtype=np.float32)


def _box_area(xyxy):
    return max(0.0, float(xyxy[2] - xyxy[0])) * max(0.0, float(xyxy[3] - xyxy[1]))


def _iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    union = _box_area(a) + _box_area(b) - inter
    return inter / union if union > 0 else 0.0


def _hsv_hist(frame, xyxy):
    """Appearance nhẹ: histogram HSV vùng thân (nửa trên box)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    # torso: bỏ 20% đầu + 35% chân
    yh = y2 - y1
    ty1 = y1 + int(0.20 * yh)
    ty2 = y1 + int(0.65 * yh)
    crop = frame[ty1:ty2, x1:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten().astype(np.float32)


def _hist_dist(a, b):
    if a is None or b is None:
        return 0.5
    return float(cv2.compareHist(a.reshape(16, 16), b.reshape(16, 16), cv2.HISTCMP_BHATTACHARYYA))


class IDRecoverer:
    """Map raw BoT-SORT ID -> stable ID. Prefer new ID over wrong neighbor swap."""

    def __init__(
        self,
        max_frames=RECOVER_MAX_FRAMES,
        cost_thresh=RECOVER_COST_THRESH,
        max_hist=RECOVER_MAX_HIST,
        min_iou=RECOVER_MIN_IOU,
        dist_ratio=RECOVER_MAX_DIST_RATIO,
        jump_ratio=MAX_JUMP_RATIO,
    ):
        self.max_frames = max_frames
        self.cost_thresh = cost_thresh
        self.max_hist = max_hist
        self.min_iou = min_iou
        self.dist_ratio = dist_ratio
        self.jump_ratio = jump_ratio
        self.next_id = 1
        self.raw_to_stable = {}
        self.active = {}
        self.lost = OrderedDict()
        self.recovered = 0

    def _new_id(self):
        sid = self.next_id
        self.next_id += 1
        return sid

    def _box_h(self, xyxy):
        return max(1.0, float(xyxy[3] - xyxy[1]))

    def _max_dist_for(self, info):
        return self.dist_ratio * max(self._box_h(info["xyxy"]), 40.0)

    def _near_active(self, center, exclude_sid=None):
        for sid, info in self.active.items():
            if exclude_sid is not None and sid == exclude_sid:
                continue
            lim = 0.35 * self._box_h(info["xyxy"])
            if float(np.linalg.norm(center - info["center"])) < lim:
                return True
        return False

    def _looks_like_neighbor(self, center, hist, lost_hist, exclude_sid=None):
        """
        Chỉ chặn recover khi detection đứng sát người active khác
        VÀ giống người đó hơn là giống track đang mất.
        """
        for sid, info in self.active.items():
            if exclude_sid is not None and sid == exclude_sid:
                continue
            lim = 0.40 * self._box_h(info["xyxy"])
            if float(np.linalg.norm(center - info["center"])) >= lim:
                continue
            d_active = _hist_dist(hist, info.get("hist"))
            d_lost = _hist_dist(hist, lost_hist)
            # giống neighbor rõ hơn lost → nguy cơ cướp ID
            if d_active + 0.12 < d_lost:
                return True
        return False

    def _make_info(self, sid, rid, t, frame, frame_idx, prev=None):
        hist = _hsv_hist(frame, t["xyxy"])
        center = _box_center(t["xyxy"])
        if prev:
            vel = center - prev.get("center", center)
            vel = 0.7 * prev.get("vel", np.zeros(2, np.float32)) + 0.3 * vel
            hits = prev.get("hits", 0) + 1
            if hist is None:
                hist = prev.get("hist")
        else:
            vel = np.zeros(2, np.float32)
            hits = 1
        return {
            "stable_id": sid,
            "raw_id": rid,
            "xyxy": t["xyxy"],
            "conf": t["conf"],
            "center": center,
            "vel": vel,
            "hist": hist,
            "area": _box_area(t["xyxy"]),
            "last_frame": frame_idx,
            "hits": hits,
        }

    def update(self, tracks, frame, frame_idx):
        assigned_stable = set()
        output = []
        unmatched = []

        # 1) raw đã biết → giữ stable; chỉ cắt khi nhảy XA và appearance KHÁC rõ
        for t in tracks:
            rid = t["raw_id"]
            if rid not in self.raw_to_stable:
                unmatched.append(t)
                continue

            sid = self.raw_to_stable[rid]
            if sid in assigned_stable:
                del self.raw_to_stable[rid]
                unmatched.append(t)
                continue

            prev = self.active.get(sid) or self.lost.get(sid)
            center = _box_center(t["xyxy"])
            hist = _hsv_hist(frame, t["xyxy"])
            if prev is not None:
                jump = float(np.linalg.norm(center - prev["center"]))
                max_jump = self.jump_ratio * self._box_h(prev["xyxy"])
                hd = _hist_dist(hist, prev.get("hist"))
                # chỉ coi là ID switch khi vừa teleport vừa khác màu rõ
                if jump > max_jump and hd > 0.45:
                    del self.raw_to_stable[rid]
                    unmatched.append(t)
                    continue

            assigned_stable.add(sid)
            info = self._make_info(sid, rid, t, frame, frame_idx, prev)
            self.active[sid] = info
            self.lost.pop(sid, None)
            output.append(info)

        # 2) raw mới → khôi phục ID đã mất (nới mạnh, ưu tiên vị trí)
        lost_items = [(sid, info) for sid, info in self.lost.items() if sid not in assigned_stable]
        if unmatched and lost_items:
            cost = np.ones((len(unmatched), len(lost_items)), dtype=np.float32)
            for i, t in enumerate(unmatched):
                c = _box_center(t["xyxy"])
                hist = _hsv_hist(frame, t["xyxy"])
                area = _box_area(t["xyxy"])
                for j, (sid, info) in enumerate(lost_items):
                    # chỉ chặn khi giống neighbor hơn lost RÕ RỆT
                    if self._looks_like_neighbor(c, hist, info.get("hist"), exclude_sid=sid):
                        continue
                    age = frame_idx - info["last_frame"]
                    pred = info["center"] + info.get("vel", np.zeros(2, np.float32)) * min(age, 12)
                    dist = float(np.linalg.norm(c - pred))
                    max_dist = self._max_dist_for(info)
                    if dist > max_dist:
                        continue
                    pred_box = info["xyxy"].copy()
                    shift = pred - info["center"]
                    pred_box[[0, 2]] += shift[0]
                    pred_box[[1, 3]] += shift[1]
                    iou = _iou(t["xyxy"], pred_box)
                    hd = _hist_dist(hist, info.get("hist"))
                    if hd > self.max_hist:
                        continue
                    area_ratio = abs(area - info["area"]) / max(area, info["area"], 1.0)
                    # vị trí gần → ưu tiên mạnh (che khuất thường box lệch/co)
                    cost[i, j] = (
                        0.55 * min(dist / max_dist, 1.0)
                        + 0.30 * hd
                        + 0.10 * (1.0 - iou)
                        + 0.05 * min(area_ratio, 1.0)
                    )

            rows, cols = linear_sum_assignment(cost)
            matched_u = set()
            used_lost = set()
            for r, cidx in zip(rows, cols):
                if cost[r, cidx] > self.cost_thresh:
                    continue
                t = unmatched[r]
                sid, prev = lost_items[cidx]
                if sid in assigned_stable or cidx in used_lost:
                    continue
                matched_u.add(r)
                used_lost.add(cidx)
                assigned_stable.add(sid)
                self.raw_to_stable[t["raw_id"]] = sid
                self.recovered += 1
                info = self._make_info(sid, t["raw_id"], t, frame, frame_idx, prev)
                self.active[sid] = info
                self.lost.pop(sid, None)
                output.append(info)
            unmatched = [t for i, t in enumerate(unmatched) if i not in matched_u]

        # 2b) còn unmatched: gán nearest lost trong bán kính (1-1), tránh ID mới oan
        if unmatched and self.lost:
            still = []
            taken = set(assigned_stable)
            for t in unmatched:
                c = _box_center(t["xyxy"])
                hist = _hsv_hist(frame, t["xyxy"])
                best_sid, best_dist, best_prev = None, 1e9, None
                for sid, info in self.lost.items():
                    if sid in taken:
                        continue
                    if self._looks_like_neighbor(c, hist, info.get("hist"), exclude_sid=sid):
                        continue
                    age = frame_idx - info["last_frame"]
                    pred = info["center"] + info.get("vel", np.zeros(2, np.float32)) * min(age, 12)
                    dist = float(np.linalg.norm(c - pred))
                    max_dist = self._max_dist_for(info)
                    if dist <= max_dist and dist < best_dist:
                        best_sid, best_dist, best_prev = sid, dist, info
                if best_sid is not None:
                    taken.add(best_sid)
                    assigned_stable.add(best_sid)
                    self.raw_to_stable[t["raw_id"]] = best_sid
                    self.recovered += 1
                    info = self._make_info(best_sid, t["raw_id"], t, frame, frame_idx, best_prev)
                    self.active[best_sid] = info
                    self.lost.pop(best_sid, None)
                    output.append(info)
                else:
                    still.append(t)
            unmatched = still

        # 3) thật sự không khớp ai → ID mới
        for t in unmatched:
            sid = self._new_id()
            self.raw_to_stable[t["raw_id"]] = sid
            info = self._make_info(sid, t["raw_id"], t, frame, frame_idx, None)
            self.active[sid] = info
            output.append(info)

        seen = {o["stable_id"] for o in output}
        for sid in list(self.active.keys()):
            if sid not in seen:
                info = self.active.pop(sid)
                info["lost_frame"] = frame_idx
                self.lost[sid] = info

        for sid in list(self.lost.keys()):
            if frame_idx - self.lost[sid]["last_frame"] > self.max_frames:
                old_raw = self.lost[sid].get("raw_id")
                self.lost.pop(sid)
                if old_raw in self.raw_to_stable and self.raw_to_stable[old_raw] == sid:
                    del self.raw_to_stable[old_raw]

        alive = set(self.active) | set(self.lost)
        for rid, sid in list(self.raw_to_stable.items()):
            if sid not in alive:
                del self.raw_to_stable[rid]

        return output

    def demote_ids(self, sids, frame_idx):
        """Đưa các ID bị loại (box chồng) về lost để khỏi tranh ID."""
        for sid in list(sids):
            if sid not in self.active:
                continue
            info = self.active.pop(sid)
            info["lost_frame"] = frame_idx
            self.lost[sid] = info
            rid = info.get("raw_id")
            if rid in self.raw_to_stable and self.raw_to_stable[rid] == sid:
                del self.raw_to_stable[rid]


class HumanTracker:
    def __init__(self, model_path=MODEL_PATH, tracker_yaml=TRACKER_YAML):
        logging.info(f"Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.device = "0" if torch.cuda.is_available() else "cpu"
        self.use_half = torch.cuda.is_available()

        with open(tracker_yaml, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.tracker = BOTSORT(IterableSimpleNamespace(**cfg))
        self.recoverer = IDRecoverer()
        logging.info(
            f"Pipeline: predict→filter→BoT-SORT→IDRecoverer | device={self.device} | "
            f"detect_conf={DETECT_CONF} track_conf={TRACK_CONF} aspect>={MIN_ASPECT}"
        )

    def filter_detections(self, boxes, frame_h):
        """Lọc FP + gom box chồng/lồng trước khi vào tracker."""
        if boxes is None or len(boxes) == 0:
            return None

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        n = len(xyxy)
        keep = np.ones(n, dtype=bool)
        min_h = frame_h * MIN_HEIGHT_RATIO
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])

        for i in range(n):
            w = float(xyxy[i, 2] - xyxy[i, 0])
            h = float(xyxy[i, 3] - xyxy[i, 1])
            aspect = h / w if w > 1 else 0
            if (
                conf[i] < TRACK_CONF
                or h < min_h
                or areas[i] < MIN_BOX_AREA
                or aspect < MIN_ASPECT
                or aspect > MAX_ASPECT
            ):
                keep[i] = False

        # Nested: box nhỏ nằm trong box lớn
        for i in range(n):
            if not keep[i]:
                continue
            for j in range(n):
                if i == j or not keep[j]:
                    continue
                ix1 = max(xyxy[i, 0], xyxy[j, 0])
                iy1 = max(xyxy[i, 1], xyxy[j, 1])
                ix2 = min(xyxy[i, 2], xyxy[j, 2])
                iy2 = min(xyxy[i, 3], xyxy[j, 3])
                inter_a = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                if areas[i] > 0 and inter_a / areas[i] > NEST_THRESH and areas[j] > areas[i]:
                    keep[i] = False
                    break

        # Soft-NMS theo IoU: box chồng nhau → giữ conf cao hơn (rồi area)
        order = sorted([i for i in range(n) if keep[i]], key=lambda i: (conf[i], areas[i]), reverse=True)
        suppressed = set()
        selected = []
        for i in order:
            if i in suppressed:
                continue
            selected.append(i)
            for j in order:
                if j == i or j in suppressed:
                    continue
                if _iou(xyxy[i], xyxy[j]) >= OVERLAP_IOU:
                    suppressed.add(j)
                    keep[j] = False

        idx = np.array(selected, dtype=int)
        if len(idx) == 0:
            return None
        return boxes[idx]

    @staticmethod
    def suppress_overlapping_tracks(tracks, iou_thresh=TRACK_OVERLAP_IOU):
        """
        Khi 2 track chồng box: giữ ID có hits cao hơn / ID nhỏ hơn (cũ hơn).
        Track bị loại vẫn còn trong recoverer.lost ở frame sau nếu cần.
        """
        if len(tracks) <= 1:
            return tracks

        # ưu tiên: hits ↓, conf ↓, stable_id ↑ (ID cũ)
        ranked = sorted(
            tracks,
            key=lambda t: (-t.get("hits", 0), -t.get("conf", 0), t["stable_id"]),
        )
        kept = []
        for t in ranked:
            clash = False
            for k in kept:
                if _iou(t["xyxy"], k["xyxy"]) >= iou_thresh:
                    clash = True
                    break
            if not clash:
                kept.append(t)
        # giữ thứ tự ổn định theo ID
        kept.sort(key=lambda t: t["stable_id"])
        return kept

    def process_video(self, source, output_path=None, max_frames=None, start_frame=None):
        if output_path is None:
            filename = os.path.splitext(os.path.basename(source))[0]
            output_path = f"{filename}_tracked.mp4"

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logging.error(f"Cannot open video source: {source}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if start_frame is not None:
            if start_frame < 0:
                start_frame = max(0, total_frames + start_frame)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            logging.info(f"Jumped to starting frame: {start_frame}/{total_frames}")

        logging.info(f"Processing: {width}x{height} @ {fps_video:.1f} FPS, Total Frames: {total_frames}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps_video, (width, height))

        prev_time = 0.0
        frame_count = 0
        unique_ids = set()

        try:
            while cap.isOpened():
                if max_frames is not None and frame_count >= max_frames:
                    logging.info(f"Reached max_frames limit ({max_frames}). Stopping early.")
                    break

                success, frame = cap.read()
                if not success:
                    break
                frame_count += 1

                # 1) Detect
                results = self.model.predict(
                    frame,
                    conf=DETECT_CONF,
                    iou=IOU,
                    imgsz=IMGSZ,
                    max_det=MAX_DET,
                    classes=[0],
                    device=self.device,
                    half=self.use_half,
                    verbose=False,
                )[0]

                # 2) Filter FP TRƯỚC tracker
                filtered = self.filter_detections(results.boxes, height)

                # 3) BoT-SORT (luôn update, kể cả 0 detection, để age lost tracks)
                if filtered is not None and len(filtered):
                    tracks_np = self.tracker.update(filtered, frame)
                elif results.boxes is not None:
                    tracks_np = self.tracker.update(results.boxes[:0], frame)
                else:
                    tracks_np = np.empty((0, 8))

                raw_tracks = []
                if tracks_np is not None and len(tracks_np):
                    for row in tracks_np:
                        x1, y1, x2, y2 = row[:4]
                        raw_id = int(row[4])
                        conf = float(row[5]) if len(row) > 5 else 0.0
                        raw_tracks.append(
                            {
                                "raw_id": raw_id,
                                "xyxy": np.array([x1, y1, x2, y2], dtype=np.float32),
                                "conf": conf,
                            }
                        )

                # 3b) bỏ raw track chồng nhau (giữ conf cao hơn)
                if len(raw_tracks) > 1:
                    raw_tracks = sorted(raw_tracks, key=lambda t: t["conf"], reverse=True)
                    pruned = []
                    for t in raw_tracks:
                        if any(_iou(t["xyxy"], k["xyxy"]) >= OVERLAP_IOU for k in pruned):
                            continue
                        pruned.append(t)
                    raw_tracks = pruned

                # 4) Stabilize IDs
                stable = self.recoverer.update(raw_tracks, frame, frame_count)
                stable = [t for t in stable if t["hits"] >= MIN_HITS]
                # 4b) bỏ ID chồng box — giữ ID già/hits cao hơn
                before = {t["stable_id"] for t in stable}
                stable = self.suppress_overlapping_tracks(stable)
                dropped = before - {t["stable_id"] for t in stable}
                if dropped:
                    self.recoverer.demote_ids(dropped, frame_count)

                annotated = frame.copy()
                curr_time = time.time()
                fps = 1.0 / (curr_time - prev_time) if prev_time else 0.0
                prev_time = curr_time

                for t in stable:
                    x1, y1, x2, y2 = t["xyxy"].astype(int)
                    sid = t["stable_id"]
                    unique_ids.add(sid)
                    color = _id_color(sid)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                    _draw_label(annotated, f"ID {sid}", x1, y1, color)

                _draw_hud(
                    annotated,
                    [
                        f"People  {len(stable)}",
                        f"Unique  {len(unique_ids)}",
                        f"FPS     {fps:.1f}",
                    ],
                )

                try:
                    cv2.imshow("Tracking", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except Exception:
                    pass

                out.write(annotated)

                if frame_count % 100 == 0:
                    logging.info(
                        f"Frame {frame_count}/{total_frames} | "
                        f"unique={len(unique_ids)} | recovered={self.recoverer.recovered} | FPS={fps:.1f}"
                    )

        finally:
            cap.release()
            out.release()
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            logging.info(
                f"Done. Saved: {output_path} | frames={frame_count} | "
                f"unique_ids={len(unique_ids)} | recovered={self.recoverer.recovered}"
            )


if __name__ == "__main__":
    # Cần: pip install scipy pyyaml
    tracker = HumanTracker()
    tracker.process_video(source=VIDEO_SOURCE)

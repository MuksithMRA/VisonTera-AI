import threading
import time
from typing import Dict, List, Tuple

import numpy as np


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    a1 = max((a[2] - a[0]) * (a[3] - a[1]), 1e-6)
    b1 = max((b[2] - b[0]) * (b[3] - b[1]), 1e-6)
    union = a1 + b1 - inter
    return float(inter / union) if union > 0 else 0.0


class TrackingQualityMonitor:
    def __init__(
        self,
        fragmentation_iou: float,
        collision_iou_max: float,
        ghost_unassigned_frames: int,
    ):
        self._lock = threading.Lock()
        self._frag_iou = fragmentation_iou
        self._coll_iou_max = collision_iou_max
        self._ghost_frames = ghost_unassigned_frames
        self._fragmentation_events = 0
        self._collision_events = 0
        self._global_id_switch_events = 0
        self._stable_frames = 0
        self._tracked_frames = 0
        self._handoff_ms_samples: List[float] = []
        self._handoff_max_ms = 0.0
        self._unassigned_streak: Dict[Tuple[str, int], int] = {}
        self._ghost_unassigned_events = 0
        self._last_global: Dict[Tuple[str, int], int] = {}
        self._first_seen: Dict[Tuple[str, int], float] = {}
        self._assigned_once: Dict[Tuple[str, int], bool] = {}

    def reset(self) -> None:
        with self._lock:
            self._fragmentation_events = 0
            self._collision_events = 0
            self._global_id_switch_events = 0
            self._stable_frames = 0
            self._tracked_frames = 0
            self._handoff_ms_samples.clear()
            self._handoff_max_ms = 0.0
            self._unassigned_streak.clear()
            self._ghost_unassigned_events = 0
            self._last_global.clear()
            self._first_seen.clear()
            self._assigned_once.clear()

    def process_frame(
        self,
        camera_id: str,
        entries: List[Tuple[int, np.ndarray, int]],
    ) -> None:
        now = time.time()
        with self._lock:
            n = len(entries)
            for i in range(n):
                for j in range(i + 1, n):
                    t1, b1, g1 = entries[i][0], entries[i][1], entries[i][2]
                    t2, b2, g2 = entries[j][0], entries[j][1], entries[j][2]
                    if t1 == t2:
                        continue
                    iou = _iou_xyxy(b1, b2)
                    if iou >= self._frag_iou:
                        self._fragmentation_events += 1
                    if g1 >= 0 and g2 >= 0 and g1 == g2 and iou <= self._coll_iou_max:
                        self._collision_events += 1

            for tid, box, gid in entries:
                key = (camera_id, tid)
                if key not in self._first_seen:
                    self._first_seen[key] = now
                prev = self._last_global.get(key)

                if gid >= 0:
                    self._tracked_frames += 1
                    if prev is not None and prev >= 0 and prev != gid:
                        self._global_id_switch_events += 1
                    else:
                        self._stable_frames += 1

                    if not self._assigned_once.get(key, False):
                        dt_ms = (now - self._first_seen[key]) * 1000.0
                        self._handoff_ms_samples.append(dt_ms)
                        if dt_ms > self._handoff_max_ms:
                            self._handoff_max_ms = dt_ms
                        self._assigned_once[key] = True
                    self._unassigned_streak[key] = 0
                else:
                    self._unassigned_streak[key] = self._unassigned_streak.get(key, 0) + 1
                    if self._unassigned_streak[key] >= self._ghost_frames:
                        self._ghost_unassigned_events += 1
                        self._unassigned_streak[key] = 0

                self._last_global[key] = gid

    def prune_camera(self, camera_id: str) -> None:
        with self._lock:
            rm = [k for k in self._last_global if k[0] == camera_id]
            for k in rm:
                self._last_global.pop(k, None)
                self._first_seen.pop(k, None)
                self._assigned_once.pop(k, None)
                self._unassigned_streak.pop(k, None)

    def snapshot(self) -> Dict:
        with self._lock:
            total = self._tracked_frames
            stab = self._stable_frames
            pct = (100.0 * stab / total) if total > 0 else 100.0
            hs = self._handoff_ms_samples
            avg_h = float(sum(hs) / len(hs)) if hs else 0.0
            return {
                "fragmentation_events": self._fragmentation_events,
                "collision_events": self._collision_events,
                "global_id_switch_events": self._global_id_switch_events,
                "id_stability_percent": round(pct, 2),
                "tracked_frames_scored": total,
                "handoff_latency_avg_ms": round(avg_h, 2),
                "handoff_latency_max_ms": round(self._handoff_max_ms, 2),
                "ghost_unassigned_events": self._ghost_unassigned_events,
            }

import numpy as np
from scipy.optimize import linear_sum_assignment

class SimpleTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.next_id = 1
        self.tracks = []  # list of dicts

    @staticmethod
    def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # a: (Na,4), b: (Nb,4) in xyxy
        if a.size == 0 or b.size == 0:
            return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
        ax1, ay1, ax2, ay2 = a[:,0:1], a[:,1:2], a[:,2:3], a[:,3:4]
        bx1, by1, bx2, by2 = b[:,0], b[:,1], b[:,2], b[:,3]

        inter_x1 = np.maximum(ax1, bx1)
        inter_y1 = np.maximum(ay1, by1)
        inter_x2 = np.minimum(ax2, bx2)
        inter_y2 = np.minimum(ay2, by2)
        inter_w = np.maximum(0.0, inter_x2 - inter_x1)
        inter_h = np.maximum(0.0, inter_y2 - inter_y1)
        inter = inter_w * inter_h

        area_a = (ax2 - ax1) * (ay2 - ay1)  # (Na,1)
        area_b = (bx2 - bx1) * (by2 - by1)  # (Nb,)
        union = area_a + area_b - inter
        return (inter / np.maximum(union, 1e-6)).astype(np.float32)

    def update(self, dets: np.ndarray) -> list[dict]:
        bboxes = dets[:, :4].astype(np.float32) if dets is not None and len(dets) else np.empty((0,4), np.float32)
        active = [t for t in self.tracks if t["age"] <= self.cfg.max_age]
        T, M = len(active), len(bboxes)

        assigned_t = set()
        assigned_d = set()

        if T and M:
            cost = 1.0 - self.iou(np.stack([t["bbox"] for t in active], axis=0), bboxes)
            # Forbid low overlaps
            invalid = (1.0 - cost) < self.cfg.iou_thresh
            cost = cost.copy()
            cost[invalid] = 1e6

            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] >= 1e6:
                    continue
                t = active[r]
                t["bbox"] = bboxes[c]
                t["hits"] += 1
                t["age"] = 0
                assigned_t.add(id(t))
                assigned_d.add(c)
        else:
            cost = None  # not used

        # Age unassigned tracks
        for t in active:
            if id(t) not in assigned_t:
                t["age"] += 1

        # New tracks for unassigned detections
        for i in range(M):
            if i not in assigned_d:
                self.tracks.append({
                    "id": self.next_id,
                    "bbox": bboxes[i],
                    "hits": 1,
                    "age": 0,
                })
                self.next_id += 1

        # Prune old tracks
        self.tracks = [t for t in self.tracks if t["age"] <= self.cfg.max_age]
        return self.tracks

import cv2
import numpy as np
from typing import List, Dict, Tuple

# Optional: store colors for track IDs to keep them consistent across frames
_TRACK_COLORS: dict[int, Tuple[int, int, int]] = {}

def _id_color(track_id: int) -> Tuple[int, int, int]:
    """Deterministic color per ID without touching global RNG."""
    rng = np.random.default_rng(track_id)          # local, seeded by ID
    color = tuple(int(c) for c in rng.integers(0, 256, size=3))
    return color  # type: ignore # BGR-compatible tuple

def get_color(track_id: int) -> Tuple[int, int, int]:
    if track_id not in _TRACK_COLORS:
        _TRACK_COLORS[track_id] = _id_color(track_id)
    return _TRACK_COLORS[track_id]

def draw_tracks(
    frame: np.ndarray,
    tracks: List[Dict],
    label_prefix: str = "ID"
) -> np.ndarray:
    h, w = frame.shape[:2]
    for t in tracks:
        bbox = np.asarray(t["bbox"]).astype(int)
        x1, y1, x2, y2 = bbox
        # clip to image bounds to be safe
        x1 = int(np.clip(x1, 0, w-1)); y1 = int(np.clip(y1, 0, h-1))
        x2 = int(np.clip(x2, 0, w-1)); y2 = int(np.clip(y2, 0, h-1))
        if x2 <= x1 or y2 <= y1:
            continue  # skip degenerate boxes

        color = get_color(int(t["id"]))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{label_prefix}:{t['id']}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )
    return frame  # in-place drawing

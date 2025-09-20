import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional

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
    label_prefix: str = "ID",
    detector: Optional[object] = None
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
        track_id = t["id"]
        
        # Get class name - try multiple ways
        class_name = "unknown"
        if "class_name" in t:
            class_name = t["class_name"]
        elif "class_id" in t and detector and hasattr(detector, 'get_class_name'):
            try:
                class_name = detector.get_class_name(t["class_id"])
            except:
                class_name = f"cls{t['class_id']}"
        elif "class_id" in t:
            class_name = f"cls{t['class_id']}"
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Create label with class name and ID
        if class_name != "unknown":
            label = f"{class_name} ID:{track_id}"
        else:
            label = f"{label_prefix}:{track_id}"
        
        # Calculate text size for background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw background rectangle for text
        text_y = max(text_height + 5, y1 - 5)  # Position above bbox, or inside if too close to top
        cv2.rectangle(
            frame,
            (x1, text_y - text_height - 5),
            (x1 + text_width + 5, text_y + baseline),
            color,
            -1  # filled rectangle
        )
        
        # Draw text in white for better contrast
        cv2.putText(
            frame,
            label,
            (x1 + 2, text_y - 2),
            font,
            font_scale,
            (255, 255, 255),  # White text
            thickness
        )
        
        # Optional: Draw confidence if available
        if "confidence" in t or "conf" in t:
            conf = t.get("confidence", t.get("conf", 0))
            conf_text = f"{conf:.2f}"
            conf_y = y2 + 15
            cv2.putText(
                frame,
                conf_text,
                (x1, min(conf_y, h - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1
            )
    
    return frame  # in-place drawing
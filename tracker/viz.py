import cv2
import numpy as np
from typing import List, Dict, Tuple

# Optional: store colors for track IDs to keep them consistent across frames
TRACK_COLORS = {}

def get_color(track_id: int) -> Tuple[int, int, int]:
    """
    Generate a consistent color for a given track ID.
    """
    if track_id not in TRACK_COLORS:
        np.random.seed(track_id)  # ensure same color for same ID
        color = tuple(int(c) for c in np.random.randint(0, 256, size=3))
        TRACK_COLORS[track_id] = color
    return TRACK_COLORS[track_id]

def draw_tracks(
    frame: np.ndarray,
    tracks: List[Dict],
    label_prefix: str = "ID"
) -> np.ndarray:
    """
    Draw bounding boxes and track IDs on the frame.

    Args:
        frame: np.ndarray of shape (H, W, 3), BGR, uint8.
        tracks: List of dicts, each with:
            - 'id': int
            - 'bbox': np.ndarray of shape (4,) in xyxy format
        label_prefix: Prefix for track labels (default "ID")

    Returns:
        frame with drawn tracks (np.ndarray)
    """
    for track in tracks:
        bbox = track["bbox"].astype(int)
        track_id = track["id"]
        x1, y1, x2, y2 = bbox

        color = get_color(track_id)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label above the box
        cv2.putText(
            frame,
            f"{label_prefix}{track_id}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )
    
    return frame


# ----------------------
# Optional test block
# ----------------------
if __name__ == "__main__":
    # Create a dummy frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Dummy tracks
    tracks = [
        {"id": 1, "bbox": np.array([50, 50, 200, 200])},
        {"id": 2, "bbox": np.array([300, 100, 450, 300])},
        {"id": 5, "bbox": np.array([150, 250, 350, 400])}
    ]

    # Draw and show
    frame_with_tracks = draw_tracks(frame, tracks)
    cv2.imshow("Tracks", frame_with_tracks)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#!/usr/bin/env python3
import cv2
import numpy as np

def get_color(idx: int):
    """Generate a consistent color for a given track id."""
    np.random.seed(idx)  # seed with ID for reproducibility
    color = np.random.randint(0, 255, size=3).tolist()
    return (int(color[0]), int(color[1]), int(color[2]))

def draw_tracks(img, tracks, label_prefix="ID", names=None):
    """Draw tracking boxes and labels on image."""
    for t in tracks:
        x1, y1, x2, y2 = t["bbox"].astype(int)
        cls = t["cls"]
        tid = t["id"]

        cls_name = str(cls)
        if names is not None and cls in names:
            cls_name = names[cls]

        color = get_color(tid)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{label_prefix}:{tid} {cls_name}",
                    (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def extract_tracks_from_result(result):
    """Extract track data from YOLO result."""
    tracks = []
    boxes = result.boxes
    
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
        cls  = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), int)
        ids  = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else -np.ones(len(xyxy), int)

        for i in range(len(xyxy)):
            if ids[i] < 0:
                continue  # skip untracked detections
            tracks.append({
                "id": int(ids[i]),
                "bbox": xyxy[i].astype(np.float32),
                "cls": int(cls[i]),
                "conf": float(conf[i]),
            })
    
    return tracks

def resize_frames_for_display(frames, target_height=None):
    """
    Resize frames to have the same height for side-by-side display.
    
    Args:
        frames: List of frames
        target_height: Target height, if None uses minimum height
    
    Returns:
        List of resized frames
    """
    if len(frames) <= 1:
        return frames
    
    if target_height is None:
        target_height = min(frame.shape[0] for frame in frames)
    
    resized_frames = []
    for frame in frames:
        aspect_ratio = frame.shape[1] / frame.shape[0]
        target_width = int(target_height * aspect_ratio)
        resized_frame = cv2.resize(frame, (target_width, target_height))
        resized_frames.append(resized_frame)
    
    return resized_frames
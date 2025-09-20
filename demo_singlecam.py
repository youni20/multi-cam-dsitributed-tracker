#!/usr/bin/env python3
import cv2
import numpy as np
from ultralytics import YOLO

# Choose tracker backend
tracker_yaml = "bytetrack.yaml"   # or "botsort.yaml"

def get_color(idx: int):
    """Generate a consistent color for a given track id."""
    np.random.seed(idx)  # seed with ID for reproducibility
    color = np.random.randint(0, 255, size=3).tolist()
    return (int(color[0]), int(color[1]), int(color[2]))

def draw_tracks(img, tracks, label_prefix="ID", names=None):
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

def main():
    model = YOLO("yolov8n.pt")  # swap to yolov8s.pt for better boxes
    names = model.names          # {id: 'classname'}

    stream = model.track(
        source=0,               # webcam index or video path
        conf=0.35,
        iou=0.5,
        imgsz=640,
        tracker=tracker_yaml,
        persist=True,
        stream=True,
        verbose=False
    )

    for result in stream:
        frame = result.orig_img.copy()

        boxes = result.boxes
        tracks = []
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
            cls  = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), int)
            ids  = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else -np.ones(len(xyxy), int)

            for i in range(len(xyxy)):
                if ids[i] < 0:
                    continue  # skip untracked dets
                tracks.append({
                    "id": int(ids[i]),
                    "bbox": xyxy[i].astype(np.float32),
                    "cls": int(cls[i]),
                    "conf": float(conf[i]),
                })

        draw_tracks(frame, tracks, label_prefix="ID", names=names)

        cv2.imshow("YOLOv8 + ByteTrack", frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

import numpy as np
from ultralytics import YOLO
import cv2
import os

class YOLODetector:
    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.25,
                 iou: float = 0.5, imgsz: int = 640, classes: tuple[int, ...] | None = None):
        self.model = YOLO(weights)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.classes = classes

    def get_class_name(self, class_id):
        """Get class name from class ID"""
        return self.model.names[int(class_id)]

    def __call__(self, image: np.ndarray) -> np.ndarray:
        # Run inference with explicit params; Ultralytics accepts BGR numpy arrays
        res = self.model.predict(
            image,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            verbose=False
        )[0]

        # No detections
        if len(res.boxes) == 0: # type: ignore
            return np.empty((0, 6), dtype=np.float32)

        # Extract xyxy, conf, cls -> (N, 6) float32
        xyxy = res.boxes.xyxy.cpu().numpy() # type: ignore
        conf = res.boxes.conf.cpu().numpy().reshape(-1, 1) # type: ignore
        cls  = res.boxes.cls.cpu().numpy().reshape(-1, 1) # type: ignore
        return np.hstack([xyxy, conf, cls]).astype(np.float32)
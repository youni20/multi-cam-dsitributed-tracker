import numpy as np
from ultralytics import YOLO
import cv2
import os 

class  YoloDetector():
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        

    def detect(self, image: np.ndarray) -> np.ndarray:
        results = self.model(image) 

        result = results[0]

        # For no detections
        if len(result.boxes) == 0:
            return np.empty((0,6), dtype=np.float32)

        # Extract data
        x1y1x2y2 = result.boxes.xyxy.cpu().numpy() 
        conf = result.boxes.conf.cpu().numpy().reshape(-1,1)  
        cls = result.boxes.cls.cpu().numpy().reshape(-1,1)    

        # Concatenate into (N,6)
        detections = np.hstack([x1y1x2y2, conf, cls]).astype(np.float32)

        return detections


if __name__ == "__main__":
    # dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_img = cv2.imread(os.path.join("data", "images", "car.png"))
    dummy_img = cv2.resize(dummy_img, (640, 480))
    detector = YoloDetector("yolov8n.pt")

    results = detector.detect(dummy_img) 
    print("Detections shape:", results.shape)
    print("Detections array:", results)

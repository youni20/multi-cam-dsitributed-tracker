#!/usr/bin/env python3
# demo_singlecam.py
#
# End-to-end webcam test:
# Camera -> YOLO detections -> per-camera tracker -> draw -> display

import os
import sys
import time
import argparse
import numpy as np
import cv2

# Let Python find "tracker" inside src/
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from tracker.camera import CameraStream
from tracker.detector import YOLODetector
from tracker.tracker import SimpleTracker
from tracker.viz import draw_tracks

# If you have a dataclass TrackerConfig, import it; else use a tiny shim
try:
    from tracker.config import TrackerConfig  # optional
except Exception:
    from types import SimpleNamespace
    TrackerConfig = lambda **kw: SimpleNamespace(**{"max_age": 30, "iou_thresh": 0.3, "min_hits": 1} | kw)


def parse_args():
    ap = argparse.ArgumentParser(description="Single-camera MVP demo")
    ap.add_argument("--source", default="0", help="Camera index or video path (default: 0)")
    ap.add_argument("--weights", default="yolov8n.pt", help="YOLO weights")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    ap.add_argument("--classes", type=int, nargs="*", default=None,
                    help="Filter classes (e.g., --classes 0 for 'person')")
    ap.add_argument("--width", type=int, default=None, help="Force camera width")
    ap.add_argument("--height", type=int, default=None, help="Force camera height")
    ap.add_argument("--show-fps", action="store_true", help="Overlay FPS")
    return ap.parse_args()


def main():
    args = parse_args()

    # Parse source as int if it looks like a number
    source = int(args.source) if str(args.source).isdigit() else args.source

    # Bring up camera reader
    cam = CameraStream(source, width=args.width, height=args.height).start()

    # Build detector + tracker
    det = YOLODetector(weights=args.weights, conf=args.conf, iou=args.iou,
                       imgsz=args.imgsz, classes=tuple(args.classes) if args.classes else None)
    trk = SimpleTracker(TrackerConfig())

    last_t = time.time()
    fps = 0.0

    print("[Demo] Press 'q' to quit.")
    try:
        while True:
            frame = cam.read()
            if frame is None:
                # camera not ready yet
                time.sleep(0.005)
                continue

            # Run detector -> (N,6) [x1,y1,x2,y2,conf,cls]
            dets = det(frame)

            # Update tracker (list of dicts with stable 'id' and 'bbox')
            tracks = trk.update(dets)

            # Draw in-place
            draw_tracks(frame, tracks, label_prefix="cam0")

            # Optional: overlay FPS and detection count
            if args.show_fps:
                now = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, now - last_t))
                last_t = now
                txt = f"FPS:{fps:5.1f}  dets:{len(dets):2d}  tracks:{len(tracks):2d}"
                cv2.putText(frame, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("MCDT — single cam", frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

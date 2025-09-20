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
    TrackerConfig = lambda **kw: SimpleNamespace(**{"max_age": 3, "iou_thresh": 0.5, "min_hits": 1} | kw)


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


def debug_draw_tracks(frame, tracks, detector=None, label_prefix="cam"):
    """Debug version of draw_tracks that shows class names and IDs"""
    for track in tracks:
        bbox = track["bbox"]
        track_id = track["id"]
        
        # Get class name if available
        class_name = track.get("class_name", "unknown")
        if class_name == "unknown" and "class_id" in track and detector:
            class_name = detector.get_class_name(track["class_id"])
        
        # Draw bounding box
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label with class name and ID
        label = f"{label_prefix}:{class_name}_{track_id}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(frame, (x1, y1-label_size[1]-10), (x1+label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


def main():
    args = parse_args()

    # Parse source as int if it looks like a number
    source = int(args.source) if str(args.source).isdigit() else args.source

    print(f"[Debug] Trying to open camera source: {source}")
    
    # Bring up camera reader
    try:
        cam = CameraStream(source, width=args.width, height=args.height).start()
        print("[Debug] Camera stream started successfully")
    except Exception as e:
        print(f"[Debug] Failed to start camera: {e}")
        return

    # Build detector + tracker
    try:
        det = YOLODetector(weights=args.weights, conf=args.conf, iou=args.iou,
                           imgsz=args.imgsz, classes=tuple(args.classes) if args.classes else None)
        print("[Debug] YOLO detector loaded successfully")
    except Exception as e:
        print(f"[Debug] Failed to load detector: {e}")
        return

    try:
        trk = SimpleTracker(TrackerConfig())
        print("[Debug] Tracker initialized successfully")
    except Exception as e:
        print(f"[Debug] Failed to initialize tracker: {e}")
        return

    last_t = time.time()
    fps = 0.0
    frame_count = 0

    print("[Demo] Press 'q' to quit.")
    try:
        while True:
            frame = cam.read()
            if frame is None:
                print(f"[Debug] Frame {frame_count}: No frame received")
                time.sleep(0.005)
                continue

            frame_count += 1
            if frame_count % 30 == 0:  # Print every 30 frames
                print(f"[Debug] Processing frame {frame_count}")

            # Run detector -> (N,6) [x1,y1,x2,y2,conf,cls]
            try:
                dets = det(frame)
                if len(dets) > 0:
                    print(f"[Debug] Frame {frame_count}: Found {len(dets)} detections")
            except Exception as e:
                print(f"[Debug] Detector error: {e}")
                continue

            # Update tracker (list of dicts with stable 'id' and 'bbox')
            try:
                # Check if tracker update method accepts detector parameter
                import inspect
                sig = inspect.signature(trk.update)
                if 'detector' in sig.parameters:
                    tracks = trk.update(dets, detector=det)
                else:
                    tracks = trk.update(dets)
                
                if len(tracks) > 0:
                    print(f"[Debug] Frame {frame_count}: Tracking {len(tracks)} objects")
                    for track in tracks:
                        class_name = track.get("class_name", "unknown")
                        print(f"  - ID {track['id']}: {class_name}")
            except Exception as e:
                print(f"[Debug] Tracker error: {e}")
                continue

            # Draw in-place
            try:
                # Pass detector to draw_tracks for class names
                draw_tracks(frame, tracks, label_prefix="cam0", detector=det)
            except Exception as e:
                print(f"[Debug] Drawing error: {e}")
                # Fallback to debug version
                debug_draw_tracks(frame, tracks, detector=det, label_prefix="cam0")

            # Optional: overlay FPS and detection count
            if args.show_fps:
                now = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, now - last_t))
                last_t = now
                txt = f"FPS:{fps:5.1f}  dets:{len(dets):2d}  tracks:{len(tracks):2d}"
                cv2.putText(frame, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Always show frame info
            info_txt = f"Frame: {frame_count}  Dets: {len(dets)}  Tracks: {len(tracks)}"
            cv2.putText(frame, info_txt, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("MCDT — single cam", frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n[Debug] Interrupted by user")
    except Exception as e:
        print(f"[Debug] Main loop error: {e}")
    finally:
        print("[Debug] Cleaning up...")
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
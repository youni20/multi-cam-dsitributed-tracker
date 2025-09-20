#!/usr/bin/env python3
import cv2
import threading
from queue import Queue
from ultralytics import YOLO
from tracking_utils import draw_tracks, extract_tracks_from_result

class CameraProcessor:
    def __init__(self, camera_id, model_path="yolov8n.pt", tracker_yaml="bytetrack.yaml"):
        self.camera_id = camera_id
        self.model_path = model_path
        self.tracker_yaml = tracker_yaml
        self.frame_queue = Queue(maxsize=2)
        self.running = False
        self.thread = None
        self.model = None
        
    def start(self):
        """Start processing camera in a separate thread."""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._process_camera)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        """Stop processing camera."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            
    def get_frame(self):
        """Get the latest processed frame if available."""
        if not self.frame_queue.empty():
            return self.frame_queue.get()
        return None
        
    def _process_camera(self):
        """Process camera frames with YOLO tracking."""
        try:
            # Create a separate model instance for this thread
            self.model = YOLO(self.model_path)
            names = self.model.names
            
            # Start tracking stream
            stream = self.model.track(
                source=self.camera_id,
                conf=0.35,
                iou=0.5,
                imgsz=640,
                tracker=self.tracker_yaml,
                persist=True,
                stream=True,
                verbose=False
            )
            
            for result in stream:
                if not self.running:
                    break
                    
                frame = result.orig_img.copy()
                
                # Add camera label
                cv2.putText(frame, f"Camera {self.camera_id}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Extract and draw tracks
                tracks = extract_tracks_from_result(result)
                draw_tracks(frame, tracks, label_prefix="ID", names=names)
                
                # Put frame in queue for display (non-blocking)
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                else:
                    # Remove old frame and add new one
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put(frame)
                    except:
                        pass
                        
        except Exception as e:
            print(f"Error processing camera {self.camera_id}: {e}")
        finally:
            # Clean up
            if hasattr(self, 'model') and self.model:
                del self.model
#!/usr/bin/env python3
import cv2
import numpy as np
import time
from camera_utils import detect_cameras, validate_camera
from camera_processor import CameraProcessor
from tracking_utils import resize_frames_for_display

def main():
    print("Multi-Camera YOLO Tracker Starting...")
    print("=" * 50)
    
    # Detect available cameras with more robust testing
    print("Detecting cameras...")
    available_cameras = detect_cameras(max_cameras=5, test_frames=3)
    
    if not available_cameras:
        print("No working cameras detected!")
        return
    
    print(f"\nFound {len(available_cameras)} working camera(s): {available_cameras}")
    
    # Use up to 2 cameras
    cameras_to_use = available_cameras[:2]
    print(f"Using cameras: {cameras_to_use}")
    
    # Create camera processors
    processors = []
    for cam_id in cameras_to_use:
        print(f"Initializing camera {cam_id}...")
        processor = CameraProcessor(cam_id)
        processors.append(processor)
    
    # Start all camera processors
    print("\nStarting camera processing...")
    for processor in processors:
        processor.start()
    
    # Give cameras time to initialize
    print("Waiting for cameras to initialize...")
    time.sleep(3)
    
    # Main display loop
    window_name = f"YOLOv8 + ByteTrack - {len(cameras_to_use)} Camera(s)"
    print(f"\nDisplaying in window: '{window_name}'")
    print("Press 'q' to quit")
    
    frames_dict = {}
    last_frame_time = {}
    
    try:
        while True:
            # Get latest frames from each camera
            current_time = time.time()
            for i, processor in enumerate(processors):
                frame = processor.get_frame()
                if frame is not None:
                    frames_dict[processor.camera_id] = frame
                    last_frame_time[processor.camera_id] = current_time
            
            # Remove stale frames (older than 2 seconds)
            stale_cameras = [cam_id for cam_id, timestamp in last_frame_time.items() 
                           if current_time - timestamp > 2.0]
            for cam_id in stale_cameras:
                if cam_id in frames_dict:
                    del frames_dict[cam_id]
                if cam_id in last_frame_time:
                    del last_frame_time[cam_id]
            
            # Display frames if we have any
            if frames_dict:
                if len(cameras_to_use) == 1:
                    # Single camera - full window
                    camera_id = cameras_to_use[0]
                    if camera_id in frames_dict:
                        display_frame = frames_dict[camera_id]
                    else:
                        # Show blank frame with message
                        display_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(display_frame, "Waiting for camera...", 
                                   (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                else:
                    # Multiple cameras - side by side
                    frame_list = []
                    for cam_id in cameras_to_use:
                        if cam_id in frames_dict:
                            frame_list.append(frames_dict[cam_id])
                        else:
                            # Create placeholder frame
                            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                            cv2.putText(placeholder, f"Camera {cam_id} - No Signal", 
                                       (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            frame_list.append(placeholder)
                    
                    if frame_list:
                        # Resize frames to same height and concatenate
                        resized_frames = resize_frames_for_display(frame_list)
                        display_frame = np.hstack(resized_frames)
                    else:
                        display_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(display_frame, "No camera feeds", 
                                   (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                cv2.imshow(window_name, display_frame)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('r'):
                print("\nRestarting cameras...")
                # Stop all processors
                for processor in processors:
                    processor.stop()
                time.sleep(1)
                # Restart them
                for processor in processors:
                    processor.start()
                time.sleep(2)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Clean up
        print("Stopping camera processors...")
        for processor in processors:
            processor.stop()
        
        cv2.destroyAllWindows()
        print("Cleanup complete")

if __name__ == "__main__":
    main()
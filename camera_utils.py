#!/usr/bin/env python3
import cv2
import time

def detect_cameras(max_cameras=10, test_frames=3):
    """
    Detect available cameras by trying to open them.
    
    Args:
        max_cameras: Maximum number of camera indices to test
        test_frames: Number of frames to try reading to verify camera works
    
    Returns:
        List of working camera indices
    """
    cameras = []
    
    for i in range(max_cameras):
        print(f"Testing camera {i}...")
        cap = cv2.VideoCapture(i)
        
        if not cap.isOpened():
            print(f"Camera {i}: Could not open")
            cap.release()
            continue
            
        # Set a timeout for camera initialization
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Try to read a few frames to make sure camera actually works
        success_count = 0
        for _ in range(test_frames):
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                success_count += 1
            time.sleep(0.1)  # Small delay between reads
        
        cap.release()
        
        if success_count >= test_frames - 1:  # Allow for 1 failed frame
            cameras.append(i)
            print(f"Camera {i}: Working ✓")
        else:
            print(f"Camera {i}: Not working (only {success_count}/{test_frames} frames read)")
    
    return cameras

def validate_camera(camera_id):
    """Validate that a specific camera ID works."""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        cap.release()
        return False
    
    ret, frame = cap.read()
    cap.release()
    
    return ret and frame is not None and frame.size > 0
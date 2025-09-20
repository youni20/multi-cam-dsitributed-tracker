import cv2, threading, time
from typing import Optional

class CameraStream:
    def __init__(self, source: int | str, width: int | None = None, height: int | None = None):
        self.cap = cv2.VideoCapture(source)
        if width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.lock = threading.Lock()
        self.frame = None
        self.stopped = False
        self.fps = 0.0

    def start(self):
        t = threading.Thread(target=self._update, daemon=True)
        t.start()
        return self

    def _update(self):
        prev = time.time()
        while not self.stopped:
            ret, f = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = f
            now = time.time()
            self.fps = 1.0 / max(1e-6, (now - prev))
            prev = now

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.stopped = True
        try:
            self.cap.release()
        except Exception:
            pass

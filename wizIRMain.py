#!/usr/bin/env python3
import time
import signal
import numpy as np
from multiprocessing import shared_memory
from picamera2 import Picamera2

SHM_FRAME_NAME = "wizir_frame"
SHM_META_NAME  = "wizir_meta"

W, H = 320, 240
FPS_TARGET = 120

RUNNING = True
def _sig_handler(sig, frame):
    global RUNNING
    RUNNING = False

signal.signal(signal.SIGINT, _sig_handler)
signal.signal(signal.SIGTERM, _sig_handler)

def main():
    frame_nbytes = H * W * 3           # uint8
    meta_nbytes  = 4 * 8               # 4 x int64

    shm_frame = shared_memory.SharedMemory(name=SHM_FRAME_NAME, create=True, size=frame_nbytes)
    shm_meta  = shared_memory.SharedMemory(name=SHM_META_NAME,  create=True, size=meta_nbytes)

    frame_buf = np.ndarray((H, W, 3), dtype=np.uint8, buffer=shm_frame.buf)
    meta      = np.ndarray((4,), dtype=np.int64, buffer=shm_meta.buf)
    meta[:] = 0
    meta[2] = 1  # running

    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"format": "RGB888", "size": (W, H)},
        controls={"FrameRate": FPS_TARGET},
        buffer_count=6,
    )
    picam2.configure(config)
    picam2.start()

    frame_id = 0

    # FPS measurement
    t0 = time.perf_counter()
    frames_in_window = 0


    try:
        while RUNNING:
            rgb = picam2.capture_array("main")       # (H,W,3) RGB
            bgr = rgb[:, :, ::-1]                    # OpenCV-friendly

            # Write pixels first
            frame_buf[:, :, :] = bgr

            # Commit meta last
            frame_id += 1
            meta[0] = frame_id
            meta[1] = time.time_ns()
            meta[2] = 1

            # FPS counter
            frames_in_window += 1
            now = time.perf_counter()
            dt = now - t0
            if dt >= 1.0:
                fps = frames_in_window / dt
                print(f"[capture] FPS={fps:.1f} (target={FPS_TARGET}) frame_id={frame_id}")
                t0 = now
                frames_in_window = 0

    finally:
        meta[2] = 0
        try:
            picam2.stop()
        except Exception:
            pass

        shm_frame.close()
        shm_meta.close()
        shm_frame.unlink()
        shm_meta.unlink()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import sys
import time
import signal
import subprocess
import numpy as np
from multiprocessing import shared_memory

SHM_NAME = "WIZIR_RESULT"
SHM_SIZE = 4 * 4   # 4 float32

PRINT_HZ = 20.0
PRINT_DT = 1.0 / PRINT_HZ

STOP = False
def _sig(sig, frame):
    global STOP
    STOP = True

signal.signal(signal.SIGINT, _sig)
signal.signal(signal.SIGTERM, _sig)

def safe_unlink(name: str):
    try:
        s = shared_memory.SharedMemory(name=name, create=False)
        s.close()
        s.unlink()
    except FileNotFoundError:
        pass

def main():
    print("[orc] starting")

    safe_unlink(SHM_NAME)

    shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=SHM_SIZE)
    res = np.ndarray((4,), dtype=np.float32, buffer=shm.buf)
    res[:] = [-999.0, -999.0, -999.0, 0.0]

    print("[orc] starting WIZIR.py")
    p = subprocess.Popen([sys.executable, "WIZIR.py"])

    next_t = time.perf_counter()

    try:
        while not STOP:
            # If WIZIR died, stop
            if p.poll() is not None:
                print("[orc] WIZIR exited")
                break

            # print at 20 Hz
            now = time.perf_counter()
            if now >= next_t:
                cx, cy, area, fps = res.tolist()
                print(f"[orc] cx={cx:8.1f} cy={cy:8.1f} area={area:8.0f} fps={fps:6.1f}")

                # schedule next tick (prevents drift)
                next_t += PRINT_DT
                # if we fell behind a lot, resync
                if now - next_t > 0.5:
                    next_t = now + PRINT_DT

            # sleep until next tick (but stay responsive)
            sleep_for = max(0.0, next_t - time.perf_counter())
            time.sleep(min(sleep_for, 0.01))

    finally:
        print("[orc] stopping")

        if p.poll() is None:
            p.terminate()
            time.sleep(0.3)
            if p.poll() is None:
                p.kill()

        shm.close()
        try:
            shm.unlink()
        except FileNotFoundError:
            # If something else already unlinked it, ignore
            pass

        print("[orc] done")

if __name__ == "__main__":
    main()

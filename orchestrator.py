#!/usr/bin/env python3
import sys
import time
import signal
import subprocess
import numpy as np
from multiprocessing import shared_memory

SHM_NAME = "WIZIR_RESULT"
SHM_SIZE = 4 * 4   # 4 float32

STOP = False
def _sig(sig, frame):
    global STOP
    STOP = True

signal.signal(signal.SIGINT, _sig)
signal.signal(signal.SIGTERM, _sig)

def safe_unlink(name):
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

    try:
        while not STOP:
            cx, cy, area, fps = res.tolist()
            print(f"[orc] cx={cx:.1f} cy={cy:.1f} area={area:.0f} fps={fps:.1f}")
            time.sleep(1.0)

            if p.poll() is not None:
                print("[orc] WIZIR exited")
                break
    finally:
        print("[orc] stopping")

        if p.poll() is None:
            p.terminate()
            time.sleep(0.3)
            if p.poll() is None:
                p.kill()

        shm.close()
        shm.unlink()

        print("[orc] done")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import os
import sys
from multiprocessing import shared_memory
from multiprocessing import resource_tracker
import numpy as np

SHM_NAME = "WIZIR_RESULT"     # ONLY ONE SHM
SHM_SIZE = 4 * 4              # 4 float32 (cx, cy, area, fps)

def safe_unlink(name: str):
    try:
        s = shared_memory.SharedMemory(name=name, create=False)
        s.close()
        s.unlink()
    except FileNotFoundError:
        pass

def main():
    print("[orc] starting (execv mode)")

    # Cleanup from previous crash
    safe_unlink(SHM_NAME)

    # Create SHM
    shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=SHM_SIZE)
    arr = np.ndarray((4,), dtype=np.float32, buffer=shm.buf)
    arr[:] = np.array([-999.0, -999.0, -999.0, 0.0], dtype=np.float32)

    # CRITICAL: stop Python's resource_tracker from auto-unlinking it on execv
    # (Otherwise SHM disappears before WIZIR attaches)
    resource_tracker.unregister(shm._name, "shared_memory")

    # Close our handle; SHM remains alive
    shm.close()

    print("[orc] exec -> WIZIR.py")
    os.execv(sys.executable, [sys.executable, "WIZIR.py"])

if __name__ == "__main__":
    main()

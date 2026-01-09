#!/usr/bin/env python3
import sys
import time
import signal
import subprocess
import numpy as np
from multiprocessing import shared_memory

SHM_NAME = "WIZIR_RESULT"
SHM_SIZE = 4 * 4   # 4 float32: (cx, cy, area, fps_proc)

PRINT_HZ = 20.0
PRINT_DT = 1.0 / PRINT_HZ

STOP = False
def _sig(sig, frame):
    global STOP
    STOP = True

signal.signal(signal.SIGINT, _sig)
signal.signal(signal.SIGTERM, _sig)

# ----------------------------
# RPi telemetry helpers
# ----------------------------
def get_temp_c():
    with open("/sys/class/thermal/thermal_zone0/temp") as f:
        return int(f.read()) / 1000.0

# ----------------------------
# SHM helpers
# ----------------------------
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

    # CPU usage needs two samples
    last_telemetry_t = time.perf_counter()
    temp_c = None
    try:
        while not STOP:
            if p.poll() is not None:
                print("[orc] WIZIR exited")
                break

            now = time.perf_counter()

            # Update telemetry at ~2 Hz (cheap + avoids too much /proc IO at 20 Hz)
            if now - last_telemetry_t >= 0.5:
                temp_c = get_temp_c()
                last_telemetry_t = now

            # print at 20 Hz
            if now >= next_t:
                cx, cy, area, fps = res.tolist()

                # format extras
                t_str = f"{temp_c:4.1f}C" if temp_c is not None else "  n/a"

                print(
                    f"[orc] cx={cx:8.1f} cy={cy:8.1f} area={area:8.0f} fps={fps:6.1f} | "
                    f"temp={t_str}"
                )

                next_t += PRINT_DT
                if now - next_t > 0.5:
                    next_t = now + PRINT_DT

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
            pass

        print("[orc] done")

if __name__ == "__main__":
    main()

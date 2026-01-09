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
def read_temp_c():
    """
    Raspberry Pi CPU temp.
    Returns float or None if not available (e.g., macOS).
    """
    paths = [
        "/sys/class/thermal/thermal_zone0/temp",
        "/sys/devices/virtual/thermal/thermal_zone0/temp",
    ]
    for p in paths:
        try:
            with open(p, "r") as f:
                return int(f.read().strip()) / 1000.0
        except Exception:
            pass
    return None

def read_cpu_times():
    """
    Returns (idle, total) from /proc/stat or None if not available.
    """
    try:
        with open("/proc/stat", "r") as f:
            line = f.readline()
        parts = line.split()
        if parts[0] != "cpu":
            return None
        vals = list(map(int, parts[1:]))  # user nice system idle iowait irq softirq steal ...
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)  # idle + iowait
        total = sum(vals)
        return idle, total
    except Exception:
        return None

def cpu_usage_percent(prev, curr):
    """
    prev/curr are (idle,total).
    """
    if prev is None or curr is None:
        return None
    idle0, total0 = prev
    idle1, total1 = curr
    dt = total1 - total0
    di = idle1 - idle0
    if dt <= 0:
        return None
    usage = 100.0 * (1.0 - (di / dt))
    return max(0.0, min(100.0, usage))

def read_gpu_usage_percent():
    """
    Best-effort GPU busy % on Raspberry Pi.
    There is no single universal sysfs node across all OS/images.
    We try, in order:
      1) vcgencmd (works on many Raspberry Pi OS images; may require being on Pi)
      2) DRM/VC4 sysfs busy_percent if present
    Returns float or None.
    """
    # 1) vcgencmd path (common on Raspberry Pi OS)
    try:
        out = subprocess.check_output(["vcgencmd", "measure_clock", "v3d"], stderr=subprocess.DEVNULL, text=True).strip()
        # Not usage, just clock; so we won't report that as usage.
        # Try a different vcgencmd if available:
        out2 = subprocess.check_output(["vcgencmd", "get_mem", "gpu"], stderr=subprocess.DEVNULL, text=True).strip()
        # Still not usage. Many setups simply don't expose GPU busy%.
        # Fall through to sysfs.
    except Exception:
        pass

    # 2) Try common sysfs busy percent nodes (varies by kernel/driver)
    candidates = [
        "/sys/class/drm/card0/device/gpu_busy_percent",
        "/sys/class/drm/card1/device/gpu_busy_percent",
        "/sys/kernel/debug/dri/0/v3d_busy",         # sometimes exists but often needs root + debugfs
        "/sys/kernel/debug/dri/0/gpu_busy",         # variant
    ]
    for p in candidates:
        try:
            with open(p, "r") as f:
                s = f.read().strip()
            # Some nodes give "xx" or "xx\n"; some give "busy idle" etc.
            # We'll parse first float-ish number.
            token = s.replace("%", "").split()[0]
            val = float(token)
            if 0.0 <= val <= 100.0:
                return val
        except Exception:
            pass

    return None

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
    prev_cpu = read_cpu_times()
    last_telemetry_t = time.perf_counter()
    temp_c = None
    cpu_pct = None
    gpu_pct = None

    try:
        while not STOP:
            if p.poll() is not None:
                print("[orc] WIZIR exited")
                break

            now = time.perf_counter()

            # Update telemetry at ~2 Hz (cheap + avoids too much /proc IO at 20 Hz)
            if now - last_telemetry_t >= 0.5:
                temp_c = read_temp_c()

                curr_cpu = read_cpu_times()
                cpu_pct = cpu_usage_percent(prev_cpu, curr_cpu)
                prev_cpu = curr_cpu

                gpu_pct = read_gpu_usage_percent()
                last_telemetry_t = now

            # print at 20 Hz
            if now >= next_t:
                cx, cy, area, fps = res.tolist()

                # format extras
                t_str = f"{temp_c:4.1f}C" if temp_c is not None else "  n/a"
                c_str = f"{cpu_pct:5.1f}%" if cpu_pct is not None else "  n/a"
                g_str = f"{gpu_pct:5.1f}%" if gpu_pct is not None else "  n/a"

                print(
                    f"[orc] cx={cx:8.1f} cy={cy:8.1f} area={area:8.0f} fps={fps:6.1f} | "
                    f"temp={t_str} cpu={c_str} gpu={g_str}"
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

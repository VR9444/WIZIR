#!/usr/bin/env python3
import sys
import time
import signal
import subprocess
import numpy as np
from multiprocessing import shared_memory
import serial
import struct


# =========================
# UART protocol constants
# =========================
PRE0, PRE1 = 0xAA, 0x55
END0, END1 = 0x55, 0xAA
TYPE_TRACKING = 0x01

# payload: mode(u32), updated(u32), horizError(f32)  => 12 bytes
PAYLOAD_LEN = 12


def crc16_ccitt_false(data: bytes) -> int:
    """CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF)."""
    crc = 0xFFFF
    for b in data:
        crc ^= (b << 8) & 0xFFFF
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def pack_tracking(seq_u32: int, mode_u32: int, updated_u32: int, horiz_error: float) -> bytes:
    # payload: <II f  (little-endian)
    payload = struct.pack(
        "<IIf",
        int(mode_u32) & 0xFFFFFFFF,
        int(updated_u32) & 0xFFFFFFFF,
        float(horiz_error),
    )

    # body for CRC: TYPE(u8), SEQ(u32), LEN(u8), PAYLOAD
    body = struct.pack("<BI B", TYPE_TRACKING, int(seq_u32) & 0xFFFFFFFF, PAYLOAD_LEN) + payload

    crc = crc16_ccitt_false(body)

    frame = (
        bytes([PRE0, PRE1])
        + body
        + struct.pack("<H", crc)
        + bytes([END0, END1])
    )
    return frame


SHM_NAME = "WIZIR_RESULT"
SHM_SIZE = 4 * 4   # 4 float32: (cx, cy, area, fps_proc)

# ----------------------------
# Printing
# ----------------------------
PRINT_HZ = 20.0
PRINT_DT = 1.0 / PRINT_HZ

# ----------------------------
# UART TX rate (match your ref code: 10 Hz)
# ----------------------------
TX_HZ = 10.0
TX_DT = 1.0 / TX_HZ

# ----------------------------
# Tracking interpretation
# ----------------------------
# Set this to your actual frame center (half of capture width).
# If your WIZIR pipeline works on 500px width like earlier snippets, FRAME_CX = 250.
FRAME_CX = 250.0

# If area is below this, treat as stale (tune or set to 0 to disable).
MIN_VALID_AREA = 1.0

# Mode field (your firmware meaning)
MODE_U32 = 1


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

    # timing
    next_print_t = time.perf_counter()
    next_tx_t = time.perf_counter()

    # CPU usage needs two samples
    last_telemetry_t = time.perf_counter()
    temp_c = None

    # -------------------------
    # UART config
    # -------------------------
    UART_DEV = "/dev/ttyAMA2"
    BAUD = 115200

    ser = serial.Serial(
        port=UART_DEV,
        baudrate=BAUD,
        timeout=0,
        write_timeout=0,
    )

    seq = 0  # uint32 rolling

    try:
        while not STOP:
            if p.poll() is not None:
                print("[orc] WIZIR exited")
                break

            now = time.perf_counter()

            # Update telemetry at ~2 Hz
            if now - last_telemetry_t >= 0.5:
                try:
                    temp_c = get_temp_c()
                except Exception:
                    temp_c = None
                last_telemetry_t = now

            # -------------------------
            # UART TX loop (10 Hz)
            # -------------------------
            if now >= next_tx_t:
                cx, cy, area, fps = res.tolist()

                # "updated" logic: treat sentinel values or tiny area as stale
                updated_bool = (cx > -900.0) and (area >= MIN_VALID_AREA)
                updated_u32 = 1 if updated_bool else 0

                # horizontal error in pixels relative to image center
                herr = float(cx - FRAME_CX) if updated_bool else 0.0

                frame = pack_tracking(
                    seq_u32=seq,
                    mode_u32=MODE_U32,
                    updated_u32=updated_u32,
                    horiz_error=herr,
                )
                seq = (seq + 1) & 0xFFFFFFFF

                try:
                    ser.write(frame)
                except Exception:
                    pass

                next_tx_t += TX_DT
                if now - next_tx_t > 0.5:
                    next_tx_t = now + TX_DT

            # -------------------------
            # Print loop (20 Hz)
            # -------------------------
            if now >= next_print_t:
                cx, cy, area, fps = res.tolist()
                t_str = f"{temp_c:4.1f}C" if temp_c is not None else "  n/a"

                print(
                    f"[orc] cx={cx:8.1f} cy={cy:8.1f} area={area:8.0f} fps={fps:6.1f} | "
                    f"temp={t_str} | seq={seq:10d}"
                )

                next_print_t += PRINT_DT
                if now - next_print_t > 0.5:
                    next_print_t = now + PRINT_DT

            # short sleep to avoid busy loop
            sleep_for = max(0.0, min(next_print_t, next_tx_t) - time.perf_counter())
            time.sleep(min(sleep_for, 0.01))

    finally:
        print("[orc] stopping")

        try:
            ser.close()
        except Exception:
            pass

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

import time
import struct
import numpy as np
import cv2 as cv
import serial

from multiprocessing import Process, Manager, shared_memory
from apriltag_tracker import tracker_process


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


def main():
    # -------------------------
    # Camera/tracker shared mem
    # -------------------------
    w, h = 640, 480
    shm = shared_memory.SharedMemory(create=True, size=w * h * 3)
    shm_name = shm.name
    preview = np.ndarray((h, w, 3), dtype=np.uint8, buffer=shm.buf)
    preview[:] = 0

    mgr = Manager()
    state = mgr.dict()

    p = Process(
        target=tracker_process,
        args=(state, shm_name),
        kwargs=dict(w=w, h=h, left_id=11, right_id=10, rotate_180=True, target_hz=30.0),
        daemon=True,
    )
    p.start()

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

    # -------------------------
    # 10 Hz loop (your comment says 30 Hz but code was 1/10)
    # -------------------------
    period = 1.0 / 10.0
    next_t = time.perf_counter()
    seq = 0  # uint32 rolling

    try:
        while True:
            now = time.perf_counter()
            if now < next_t:
                time.sleep(next_t - now)
            next_t += period

            updated_bool = bool(state.get("updated", False))
            updated_u32 = 1 if updated_bool else 0
            herr = float(state.get("horizontal_error_px", 0.0)) if updated_bool else 0.0

            mode_u32 = 1

            frame = pack_tracking(seq, mode_u32=mode_u32, updated_u32=updated_u32, horiz_error=herr)
            seq = (seq + 1) & 0xFFFFFFFF

            try:
                ser.write(frame)
            except Exception:
                pass

            age = float(state.get("age_s", 0.0))
            if updated_bool:
                print(f"OK    age={age:5.2f}s horiz_err_px={herr:+8.1f} seq={seq:10d}")
            else:
                print(f"STALE age={age:5.2f}s seq={seq:10d}")

            cv.imshow("apriltag", preview)
            if (cv.waitKey(1) & 0xFF) == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        try:
            ser.close()
        except Exception:
            pass

        cv.destroyAllWindows()
        if p.is_alive():
            p.terminate()
            p.join(timeout=2.0)
        shm.close()
        shm.unlink()


if __name__ == "__main__":
    main()

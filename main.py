# Project WIZIR project (Wizair / Vision IR)
# Author: Viktor Rackov,

# IMPORTS
import os
import cv2
import numpy as np
from collections import deque
import time


# CONFIGURATION PARAMETERS

# Hard CUTOFF threshold for binary segmentation(from 0-255) 
# Everything above this value is considered 1 (white), below is 0 (black)
THRESHOLD = 230

FPS = 120.0  # frames per second of input video

# Morphological cleaning params
# Opening removes small noise, closing fills small holes
# OPEN_K defines the size of opening kernel(how meny pixels to make white)
# CLOSE_K defines the size of closing kernel(how meny pixels to make black)
# Larger kernels = more aggressive cleaning
# More iterations = more aggressive cleaning

OPEN_K = 3
CLOSE_K = 3
OPEN_ITERS = 3
CLOSE_ITERS = 1

# Blob filtering
# if blob area is outside this range, it is ignored
MIN_AREA = 50
MAX_AREA = 800

# maximum allowed aspect ratio error from 1:1 (in percent) eg. 50% = max ratio 1.5:1 or 1:1.5
ASPECT_ERR_PCT = 50.0

# Tracking params
# Track is a detected BLOB candidate over time

HISTORY_LEN = 24            # How many frames of history to keep per track
ASSOC_DIST_PX = 100         # max distance for nearest-neighbour association
MAX_MISSES = 12             # how many frames a track can go unseen before deletion
MAX_TRACKS = 10             # safety cap for max number of simultaneous tracks

DELAY = True          # whether to delay between frames for visualization
SLOW_FACTOR = 1.0       # 2 = half speed, 4 = quarter speed, etc.

# Probability thresholds for LED presence decision
# 5 out of 12 frames = ~0.42
# 7 out of 12 frames = ~0.58
P_MIN = 5/12
P_MAX = 7/12


# Processing functions

def preprocess_to_clean_binary(
    frame_bgr: np.ndarray,
    threshold: int,
    k_open: np.ndarray,
    k_close: np.ndarray,
    open_iters: int,
    close_iters: int,
) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)

    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open, iterations=open_iters)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_close, iterations=close_iters)
    return cleaned


def find_blobs(
    binary_mask: np.ndarray,
    min_area: int,
    max_area: int,
    aspect_err_pct_max: float,
):
    """
    Returns a list of blob dicts: {x,y,w,h,area,cx,cy,aspect_err_pct,label}
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=4)

    blobs = []
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < min_area or area > max_area:
            continue
        if w <= 0 or h <= 0:
            continue

        r = w / float(h)
        r_sym = max(r, 1.0 / r)                 
        aspect_err_pct = (r_sym - 1.0) * 100.0  # percent error from 1:1

        if aspect_err_pct > aspect_err_pct_max:
            continue

        cx, cy = centroids[label]
        blobs.append(
            {
                "label": int(label),
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "area": int(area),
                "cx": float(cx),
                "cy": float(cy),
                "aspect_err_pct": float(aspect_err_pct),
            }
        )

    return blobs


def _dist2(a, b) -> float:
    dx = a["cx"] - b["cx"]
    dy = a["cy"] - b["cy"]
    return dx * dx + dy * dy

def predict_track_position(tr):
    """
    Predict position misses+1 frames into the future using constant velocity.
    """
    vx = tr.get("vx", 0.0)
    vy = tr.get("vy", 0.0)
    k = tr["misses"] + 1  # predict to "now" given how many frames were missed
    return tr["cx"] + vx * k, tr["cy"] + vy * k

def associate_nearest_neighbor(tracks, detections, max_dist_px: float):
    """
    Greedy NN assignment with a distance gate.
    """
    max_d2 = max_dist_px * max_dist_px

    candidates = []
    for ti, tr in enumerate(tracks):
        if tr["misses"] > MAX_MISSES:
            continue
        # last known position
        # pred = {"cx": tr["cx"], "cy": tr["cy"]}
        px, py = predict_track_position(tr)
        pred = {"cx": px, "cy": py}

        for di, det in enumerate(detections):
            d2 = _dist2(pred, det)
            if d2 <= max_d2:
                candidates.append((d2, ti, di))

    candidates.sort(key=lambda x: x[0])

    matched_t = set()
    matched_d = set()
    matches = []
    for d2, ti, di in candidates:
        if ti in matched_t or di in matched_d:
            continue
        matched_t.add(ti)
        matched_d.add(di)
        matches.append((ti, di))

    unmatched_tracks = set(range(len(tracks))) - matched_t
    unmatched_dets = set(range(len(detections))) - matched_d
    return matches, unmatched_tracks, unmatched_dets

def update_tracks(tracks, detections, matches, unmatched_tracks, unmatched_dets):
    # Mark all tracks as "not updated this frame" initially
    updated_track_idxs = set()

    # Update matched
    for ti, di in matches:
        det = detections[di]
        tr = tracks[ti]

        ensure_track_history(tr, HISTORY_LEN)

        # Update state
        tr["cx"] = det["cx"]
        tr["cy"] = det["cy"]
        tr["area"] = det["area"]
        tr["bbox"] = (det["x"], det["y"], det["w"], det["h"])
        tr["misses"] = 0
        tr["age"] += 1

        # Existing positional trail if you still want it
        tr["history"].append((det["cx"], det["cy"]))

        #boolean history
        push_track_hist(tr, present=True, det=det)
        updated_track_idxs.add(ti)

    # Age unmatched tracks
    for ti in unmatched_tracks:
        tr = tracks[ti]
        ensure_track_history(tr, HISTORY_LEN)

        tr["misses"] += 1
        tr["age"] += 1

        #oolean history: not detected this frame
        push_track_hist(tr, present=False, det=None)

    # Spawn new tracks for unmatched detections
    next_id = (max([t["id"] for t in tracks], default=-1) + 1)
    for di in sorted(unmatched_dets):
        if len(tracks) >= MAX_TRACKS:
            break
        det = detections[di]

        tr = {
            "id": next_id,
            "cx": det["cx"],
            "cy": det["cy"],
            "area": det["area"],
            "bbox": (det["x"], det["y"], det["w"], det["h"]),
            "misses": 0,
            "age": 1,

            # positional trail (existing)
            "history": deque([(det["cx"], det["cy"])], maxlen=HISTORY_LEN),
        }
        ensure_track_history(tr, HISTORY_LEN)
        push_track_hist(tr, present=True, det=det)

        tracks.append(tr)
        next_id += 1

    # Remove dead tracks
    tracks[:] = [t for t in tracks if t["misses"] <= MAX_MISSES]


def choose_best_track(tracks):

    for t in tracks:
        if "present_hist" in t and t["age"] >= 12:
            procentage_present = sum(1 for p in t["present_hist"] if p) / len(t["present_hist"])
            if procentage_present >= P_MIN and procentage_present <= P_MAX:
                return t
    return None

def ensure_track_history(tr, history_len: int):
    if "present_hist" not in tr:
        tr["present_hist"] = deque(maxlen=history_len)  # bools
    if "area_hist" not in tr:
        tr["area_hist"] = deque(maxlen=history_len)     # int or None
    if "pos_hist" not in tr:
        tr["pos_hist"] = deque(maxlen=history_len)      # (cx,cy) or None


def push_track_hist(tr, present: bool, det=None):
    """
    Append one frame of history for this track.
    det: detection dict or None
    """
    tr["present_hist"].append(bool(present))
    if present and det is not None:
        tr["area_hist"].append(int(det["area"]))
        tr["pos_hist"].append((float(det["cx"]), float(det["cy"])))
    else:
        tr["area_hist"].append(None)
        tr["pos_hist"].append(None)


def draw_tracks(frame_bgr: np.ndarray, tracks, best=None):
    vis = frame_bgr.copy()
    for tr in tracks:
        x, y, w, h = tr["bbox"]
        cx_i, cy_i = int(round(tr["cx"])), int(round(tr["cy"]))

        # Dim color if currently missing
        color = (0, 255, 0) if tr["misses"] == 0 else (0, 128, 255)

        if(tr == best):
            color = (0, 255, 255)  # Highlight best track in yellow


        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 1)
        cv2.drawMarker(
            vis,
            (cx_i, cy_i),
            (0, 0, 255) if tr["misses"] == 0 else (0, 0, 128),
            markerType=cv2.MARKER_CROSS,
            markerSize=12,
            thickness=2,
        )

        cv2.putText(
            vis,
            f"id={tr['id']} miss={tr['misses']} area={tr['area']}",
            (x, max(0, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

        # Track trail (last positions)
        pts = list(tr["history"])
        for i in range(1, len(pts)):
            p0 = (int(round(pts[i - 1][0])), int(round(pts[i - 1][1])))
            p1 = (int(round(pts[i][0])), int(round(pts[i][1])))
            cv2.line(vis, p0, p1, color, 1)

    return vis

def presence_bar(present_hist):
    """Return a compact string visualization, newest at right."""
    countPresent = sum(1 for p in present_hist if p)
    countMissing = sum(1 for p in present_hist if not p)
    string = "█" * countPresent + "·" * countMissing
    return string

# =========================
# =========================

# Main
# =========================
# =========================
def main():
    video_path = os.path.join("video", "video_120fps.h264")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_K, OPEN_K))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_K, CLOSE_K))

    # Per frame blob history
    blob_history = deque(maxlen=HISTORY_LEN)

    # Active tracks
    tracks = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            time_start = time.perf_counter()


            cleaned = preprocess_to_clean_binary(
                frame_bgr=frame,
                threshold=THRESHOLD,
                k_open=k_open,
                k_close=k_close,
                open_iters=OPEN_ITERS,
                close_iters=CLOSE_ITERS,
            )

            blobs = find_blobs(
                binary_mask=cleaned,
                min_area=MIN_AREA,
                max_area=MAX_AREA,
                aspect_err_pct_max=ASPECT_ERR_PCT,
            )

            # Save blobs to history
            blob_history.append(blobs)

            # Associate + update tracks
            matches, unmatched_tracks, unmatched_dets = associate_nearest_neighbor(
                tracks, blobs, max_dist_px=ASSOC_DIST_PX
            )
            update_tracks(tracks, blobs, matches, unmatched_tracks, unmatched_dets)

            best = choose_best_track(tracks)

            time_end = time.perf_counter()
            proc_time_ms = (time_end - time_start) * 1000.0
            
            cv2.putText(
                frame,
                f"Proc time: {proc_time_ms:.1f} ms",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            # Visualize tracks on original
            vis = draw_tracks(frame, tracks, best)

            combined = np.hstack((vis, cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)))
            cv2.imshow("Tracked | Cleaned Binary", combined)

            if DELAY:
                delay_ms = int((1000.0 / FPS) * SLOW_FACTOR)
                key = cv2.waitKey(delay_ms) & 0xFF
            else:
                key = cv2.waitKey(1) & 0xFF


            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

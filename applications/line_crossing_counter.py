#!/usr/bin/env python3
"""Count the people and vehicles that cross a line in an RTSP camera stream.

This is notebooks/people_vehicle_crossings.ipynb turned into a live application:

    RTSP camera -> newest frame -> yolo box (detect + track) -> crossing counter
                                        |                         |            |
                                  detections.csv           annotated live   detections_counts.csv
                              (a row per detection,        window (boxes,   (a row per counted
                               updated after every frame)  lines, counts)    crossing, with the totals)

1. The first frame of the stream is shown and you click two points for each line
   segment: A (start) then B (end).  The arrow A->B is what "left" and "right" refer
   to: standing at A and looking at B, "left_to_right" is a track that goes from your
   left side to your right side.
2. Every frame goes to the ``yolo`` box.  Its detections are appended to a CSV with the
   same columns as the notebook's ``detections.csv`` and flushed after every frame, so
   the file is always up to date (and the notebook's section 4 can recount it offline).
   The counts go to a second CSV, ``<csv name>_counts.csv`` (``--counts-csv``): one row for every
   crossing that is counted - when, which line, which direction, which class and track, and the totals of
   that line afterwards - so the counts over time are in a file too, not just in the window.
3. A track crosses a segment when the straight step between two consecutive positions
   cuts the finite segment.  Counts are distinct tracks per segment, direction and class
   group (person / vehicle) - the notebook's rules, computed incrementally:
     * the position of a track is the bottom centre of its box (``--anchor``),
     * a track has one class, the label it has had most often,
     * tracks with fewer than ``--min-points`` detections are ignored (mostly false
       detections); a crossing made early in a track is counted once the track is long enough,
     * a track that crosses back and forth counts once in each direction.

Needs the yolo box running (``cd fleet && docker compose up -d yolo``, default port 9068)
and a camera reachable from this computer.  Run it from a desktop session (it opens windows)::

    python line_crossing_counter.py rtsp://user:password@192.168.1.101:554/stream1
    python line_crossing_counter.py $URL --segments 2 --fps 5 --csv gate.csv
    python line_crossing_counter.py $URL --reuse-segments --no-display      # unattended

Keys in the window: q or Esc = quit.  While clicking: u = undo, r = start over, Enter = done.
"""

import argparse
import csv
import datetime as dt
import json
import os
import re
import signal
import sys
import threading
import time
from collections import Counter, deque

import cv2
import numpy as np

# ---- classes (COCO ids), as in the notebook -------------------------------------------------------
COCO_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
              5: "bus", 6: "train", 7: "truck", 8: "boat"}
PERSON_IDS = [0]
VEHICLE_IDS = [1, 2, 3, 4, 5, 6, 7, 8]          # every COCO vehicle; remove the ones you do not want
DETECT_CLASSES = PERSON_IDS + VEHICLE_IDS
GROUP_OF = {**{COCO_NAMES[i]: "person" for i in PERSON_IDS},       # label -> "person" | "vehicle"
            **{COCO_NAMES[i]: "vehicle" for i in VEHICLE_IDS}}
DIRECTIONS = ("left_to_right", "right_to_left")

# ---- the CSVs: the notebook's columns, and one row per counted crossing ----------------------------
COLUMNS = ["timestamp", "t", "camera", "frame", "track_id", "class_id", "label", "score",
           "x1", "y1", "x2", "y2", "width", "height"]
COUNT_COLUMNS = ["timestamp", "t", "camera", "frame", "line", "direction", "group", "label", "track_id", "x", "y",
                 "person_left_to_right", "person_right_to_left", "vehicle_left_to_right", "vehicle_right_to_left"]

# ---- tracking / drawing constants -----------------------------------------------------------------
ID_STRIDE = 10 ** 9         # track ids of a new tracker session are offset by this (see YoloDetector)
TRACK_TTL_S = 30.0          # a track not seen for this long is closed (its crossings stay counted)
TRAIL_POINTS = 40           # positions kept per track, for drawing only
FLASH_S = 1.0               # a segment turns red this long after a crossing
STALL_WARN_S = 5.0          # warn when the camera sends nothing for this long
FONT = cv2.FONT_HERSHEY_SIMPLEX
BGR = {"person": (50, 205, 50), "vehicle": (0, 165, 255), None: (170, 170, 170)}   # green, orange, grey
YELLOW, RED = (0, 255, 255), (0, 0, 255)


# =================================================================================================
# The camera
# =================================================================================================
def mask_url(url):
    """The URL without its password, safe to print."""
    return re.sub(r"//[^@/]*@", "//***@", url)


def _open(url, timeout_ms=8000):
    return cv2.VideoCapture(url, cv2.CAP_FFMPEG,
                            [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms,
                             cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms])


class RTSPReader:
    """Reads one RTSP stream in a background thread and keeps only the newest frame, so the
    stream never builds up delay while the yolo box is busy.  Reconnects after ~5 s without frames."""

    def __init__(self, name, url, first_frame_timeout=15.0):
        self.name, self.url = name, url
        self._lock, self._stop = threading.Lock(), threading.Event()
        self._frame, self._t, self._seq = None, 0.0, 0
        self._cap = _open(url)
        if not self._cap.isOpened():
            raise RuntimeError(f"{name}: cannot open {mask_url(url)}")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        t0 = time.time()
        while self._seq == 0:
            if time.time() - t0 > first_frame_timeout:
                self.close()
                raise RuntimeError(f"{name}: opened {mask_url(url)} but no frame arrived "
                                   f"in {first_frame_timeout:.0f} s")
            time.sleep(0.05)

    def _run(self):
        fails = 0
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if ok:
                with self._lock:
                    self._frame, self._t, self._seq = frame, time.time(), self._seq + 1
                fails = 0
            else:
                fails += 1
                time.sleep(0.2)
                if fails >= 25:                                  # about 5 s without a frame: reconnect
                    self._cap.release()
                    self._cap = _open(self.url)
                    fails = 0

    def latest(self):
        """(frame_bgr, grab_time_epoch_s, sequence_number), or None before the first frame."""
        with self._lock:
            return None if self._frame is None else (self._frame, self._t, self._seq)

    def close(self):
        self._stop.set()
        thread = getattr(self, "_thread", None)
        if thread is not None:
            thread.join(timeout=10)            # a blocked read gives up after its 8 s timeout
            if thread.is_alive():              # still inside read(): releasing now could crash it
                return
        self._cap.release()


# =================================================================================================
# The yolo box
# =================================================================================================
class YoloDetector:
    """One frame in, one detection record out: boxes (xyxy, pixels), class_ids, labels, scores and
    track_ids.  The tracker lives in the box, under a session id, so ids stay stable from frame to frame.

    After a failed call the detector starts a new session (``epoch`` + 1): a box that restarted has
    forgotten the old one and numbers its tracks from 1 again, which would pair new objects with the
    old tracks of the same number.  ``id_offset`` (epoch * ID_STRIDE) is added to the box's ids by
    parse_detections(), so a track id is unique for the whole run, in the counter and in the CSV."""

    def __init__(self, address, session_prefix, conf=0.30, weights=None, jpeg_quality=90, timeout=10.0):
        try:
            from visionist_client import Visionist
        except ImportError:
            sys.exit("visionist_client is not installed:  pip install visionist-client")
        self.box = Visionist(address, timeout=timeout)       # a short timeout: this is a live stream
        self.prefix = session_prefix
        self.epoch = 0
        self._failed, self._used = False, False              # last call failed / a call has worked in this epoch
        self.jpeg_quality = jpeg_quality
        self.params = {"conf": conf, "classes": DETECT_CLASSES, "save_annotated": False}
        if weights:
            self.params["weights"] = weights

    @property
    def session_id(self):
        return f"{self.prefix}-{self.epoch}" if self.epoch else self.prefix

    @property
    def id_offset(self):
        return self.epoch * ID_STRIDE

    def detect(self, frame):
        if self._failed and self._used:                      # once per outage, not once per failed call
            self.epoch += 1
            self._used = False
        self._failed = False
        try:
            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if not ok:
                raise RuntimeError("cannot JPEG-encode the frame")
            res = self.box.run(data={"images": [jpg.tobytes()]},
                               config={"yolo": {"command": "detect", "session_id": self.session_id,
                                                "parameters": self.params}})
            cfg = res.config.get("yolo", {}) if isinstance(res.config, dict) else {}
            if cfg.get("status") != "done":
                raise RuntimeError(f"yolo box: {cfg or res.config}")
            rec = res.fields["detections"][0]
        except Exception:
            self._failed = True
            raise
        self._used = True
        return rec

    def close(self):
        self.box.close()


def short_error(e):
    """One line for an exception; gRPC errors are otherwise printed as a multi-line dump."""
    code = getattr(e, "code", None)
    if callable(code):
        try:
            return f"{code().name}: {str(e.details()).splitlines()[0]}"
        except Exception:
            pass
    return f"{type(e).__name__}: {e}"


def parse_detections(rec, id_offset=0):
    """The box's record -> a list of dicts.  tid is -1 while the tracker has not confirmed the object."""
    boxes = rec.get("boxes") or []
    ids = rec.get("track_id") or [None] * len(boxes)
    return [dict(box=tuple(float(v) for v in b), class_id=int(c), label=lab, score=float(s),
                 tid=-1 if tid is None else int(tid) + id_offset)
            for b, c, lab, s, tid in zip(boxes, rec.get("class_ids") or [], rec.get("labels") or [],
                                         rec.get("scores") or [], ids)]


# =================================================================================================
# Counting
# =================================================================================================
def anchor_point(box, anchor="bottom"):
    """Position of a detection: x = box centre, y = bottom / centre / top of the box."""
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, {"bottom": y2, "center": (y1 + y2) / 2, "top": y1}[anchor]


class Track:
    """What the counter remembers about one tracked object."""
    __slots__ = ("tid", "labels", "n", "last_t", "last_xy", "last_side", "crossed", "pending", "trail")

    def __init__(self, tid):
        self.tid = tid
        self.labels = Counter()                  # label -> number of detections
        self.n = 0                               # detections so far
        self.last_t, self.last_xy, self.last_side = 0.0, None, None
        self.crossed = set()                     # {(segment index, direction)}
        self.pending = []                        # crossings not yet announced: the track is still too short
        self.trail = deque(maxlen=TRAIL_POINTS)

    @property
    def label(self):
        return self.labels.most_common(1)[0][0]


class CrossingCounter:
    """Counts, per segment and class group, the distinct tracks that cross it in each direction.

    segments: [((xA, yA), (xB, yB)), ...] in image pixels.  'left' and 'right' are as seen on the
    image, standing at A and looking at B.  Same rules as the notebook's crossing_events/count_crossings,
    but fed one frame at a time."""

    def __init__(self, segments, anchor="bottom", min_points=3, track_ttl=TRACK_TTL_S):
        self.segments = [tuple(tuple(map(float, p)) for p in s) for s in segments]
        self._len2 = []
        for a, b in self.segments:
            d2 = (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2
            if d2 == 0:
                raise ValueError("a segment has zero length")
            self._len2.append(d2)
        self.anchor, self.min_points, self.track_ttl = anchor, min_points, track_ttl
        self.tracks = {}                                       # track id -> Track (open tracks)
        self._closed = Counter()                               # (segment, group, direction) -> tracks already closed
        self.last_crossing_t = [None] * len(self.segments)    # for the red flash

    def update(self, t, detections):
        """One frame: ``t`` is its time (epoch s), ``detections`` as from parse_detections().
        Returns the crossings that have just become countable, as dicts with
        segment, direction, t, x, y, track_id, label, group and first (True when this is the first time
        the track crosses this segment in this direction: only those add to the counts)."""
        announced = []
        for d in detections:
            if d["tid"] < 0 or d["label"] not in GROUP_OF:     # unconfirmed by the tracker / not a person or vehicle
                continue
            tr = self.tracks.get(d["tid"])
            if tr is None:
                tr = self.tracks[d["tid"]] = Track(d["tid"])
            x, y = anchor_point(d["box"], self.anchor)
            sides = [(b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]) for a, b in self.segments]
            if tr.last_xy is not None:
                x0, y0 = tr.last_xy
                for k, (a, b) in enumerate(self.segments):
                    s0, s1 = tr.last_side[k], sides[k]
                    if (s0 > 0) == (s1 > 0):                   # > 0: right of A->B on the screen (y points down)
                        continue
                    u = s0 / (s0 - s1)                         # where on the step it meets the line
                    px, py = x0 + u * (x - x0), y0 + u * (y - y0)
                    s = ((px - a[0]) * (b[0] - a[0]) + (py - a[1]) * (b[1] - a[1])) / self._len2[k]
                    if 0.0 <= s <= 1.0:                        # inside the finite segment
                        direction = "left_to_right" if s1 > 0 else "right_to_left"
                        first = (k, direction) not in tr.crossed        # False: this track already counted here
                        tr.crossed.add((k, direction))
                        tr.pending.append(dict(segment=k, direction=direction, x=px, y=py, first=first,
                                               t=tr.last_t + u * (t - tr.last_t)))
            tr.labels[d["label"]] += 1
            tr.n += 1
            tr.trail.append((x, y))
            tr.last_xy, tr.last_side, tr.last_t = (x, y), sides, t
            if tr.n >= self.min_points and tr.pending:
                for ev in tr.pending:
                    ev.update(track_id=tr.tid, label=tr.label, group=GROUP_OF[tr.label])
                    self.last_crossing_t[ev["segment"]] = ev["t"]
                    announced.append(ev)
                tr.pending.clear()
        self._close_old(t)
        return announced

    def break_tracks(self):
        """Measure no step across a gap in the video (the camera stalled and came back): the tracker may
        have given an old id to a different object.  Counted crossings are kept."""
        for tr in self.tracks.values():
            tr.last_xy = None

    def _close_old(self, now):
        """Forget the tracks that have not been seen for track_ttl seconds (the count keeps their crossings)."""
        for tid in [tid for tid, tr in self.tracks.items() if now - tr.last_t > self.track_ttl]:
            tr = self.tracks.pop(tid)
            if tr.n >= self.min_points:
                for k, direction in tr.crossed:
                    self._closed[(k, GROUP_OF[tr.label], direction)] += 1

    def counts(self):
        """Counter {(segment index, group, direction): number of tracks}; a missing key is 0."""
        c = Counter(self._closed)
        for tr in self.tracks.values():
            if tr.n >= self.min_points:
                for k, direction in tr.crossed:
                    c[(k, GROUP_OF[tr.label], direction)] += 1
        return c


# =================================================================================================
# The CSV
# =================================================================================================
class CsvLog:
    """A CSV that is flushed after every write (the detections, and the counted crossings).  An existing
    file is kept under a time-stamped name (or continued, with append=True)."""

    def __init__(self, path, append=False, columns=COLUMNS):
        has_data = os.path.exists(path) and os.path.getsize(path) > 0
        if has_data and not append:
            stem, ext = os.path.splitext(path)
            kept = f"{stem}.{dt.datetime.fromtimestamp(os.path.getmtime(path)):%Y%m%d-%H%M%S}{ext}"
            os.replace(path, kept)
            print(f"previous {path} kept as {kept}")
            has_data = False
        self.path = path
        self._f = open(path, "a", newline="")
        self._w = csv.writer(self._f)
        if not has_data:
            self._w.writerow(columns)
            self._f.flush()

    def write_frame(self, rows):
        self._w.writerows(rows)
        self._f.flush()

    def close(self):
        self._f.close()


def count_rows(events, counts, camera, frame_no):
    """The counts CSV: one row per counted crossing (a track that crosses the same segment the same way
    again is not counted again, so it has no row), in the order they became countable, which can differ
    from the order of ``t`` by a moment.  ``t`` is when the track met the line; the last four columns are
    the totals of that segment after this frame, so the last row of a segment is its count."""
    rows = []
    for ev in events:
        if not ev["first"]:
            continue
        k = ev["segment"]
        stamp = dt.datetime.fromtimestamp(ev["t"], dt.timezone.utc).isoformat(timespec="milliseconds")
        rows.append([stamp, round(ev["t"], 3), camera, frame_no, k + 1, ev["direction"], ev["group"], ev["label"],
                     ev["track_id"], round(ev["x"], 1), round(ev["y"], 1),
                     *(counts[(k, g, d)] for g in ("person", "vehicle") for d in DIRECTIONS)])
    return rows


def csv_rows(rec, detections, camera, frame_no, t, frame_shape):
    """One row per detection, as in the notebook.  A frame with no detections adds no row."""
    stamp = dt.datetime.fromtimestamp(t, dt.timezone.utc).isoformat(timespec="milliseconds")
    width, height = rec.get("width", frame_shape[1]), rec.get("height", frame_shape[0])
    return [[stamp, round(t, 3), camera, frame_no, d["tid"], d["class_id"], d["label"],
             round(d["score"], 3), *d["box"], width, height] for d in detections]


# =================================================================================================
# Drawing
# =================================================================================================
def put_text(img, text, org, scale, color=(255, 255, 255), thick=1, right=False):
    """Text with a dark outline, readable on any background.  org is the baseline-left corner
    (or baseline-right with right=True).  The outline is the same text drawn at the same thickness,
    shifted: some OpenCV builds space the glyphs differently for different thicknesses, which would
    make a thicker outline drift away from the text."""
    (w, _), _ = cv2.getTextSize(text, FONT, scale, thick)
    x, y = int(org[0] - (w if right else 0)), int(org[1])
    for dx in (-thick, 0, thick):
        for dy in (-thick, 0, thick):
            if dx or dy:
                cv2.putText(img, text, (x + dx, y + dy), FONT, scale, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)
    return w


def draw_segment(img, index, a, b, color, k, thick):
    """The arrow A->B, its number at A, and L / R on either side of its middle."""
    pa, pb = (int(round(a[0])), int(round(a[1]))), (int(round(b[0])), int(round(b[1])))
    length = max(1.0, float(np.hypot(b[0] - a[0], b[1] - a[1])))
    cv2.arrowedLine(img, pa, pb, color, thick, cv2.LINE_AA, tipLength=min(0.2, 28.0 * k / length))
    put_text(img, str(index + 1), (a[0] + 8 * k, a[1] - 8 * k), 0.9 * k, color, 2)
    nx, ny = -(b[1] - a[1]) / length, (b[0] - a[0]) / length        # points to the "right" side
    mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    for letter, sign in (("R", 1), ("L", -1)):
        put_text(img, letter, (mx + sign * nx * 26 * k - 7 * k, my + sign * ny * 26 * k + 7 * k),
                 0.7 * k, color, 2)


def draw_overlay(frame, counter, detections, t, camera, frame_no, fps=None, box_ms=None, status=None):
    """The frame with boxes, track trails, the segments and the current counts."""
    img = frame.copy()
    h, w = img.shape[:2]
    k = max(1.0, h / 720.0)                           # the sizes below are chosen for a 720 pixel high image
    th = max(1, int(round(k)))

    for tr in counter.tracks.values():                # trails of the tracks seen in this frame
        if tr.last_t == t and len(tr.trail) > 1:
            pts = np.array(tr.trail, np.float32).round().astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], False, BGR[GROUP_OF.get(tr.label)], th, cv2.LINE_AA)
    for d in detections:                              # boxes
        x1, y1, x2, y2 = (int(round(v)) for v in d["box"])
        color = BGR[GROUP_OF.get(d["label"])]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2 * th)
        name = d["label"] + (f" #{d['tid'] % ID_STRIDE}" if d["tid"] >= 0 else "")    # the box's own id
        (tw, tg), _ = cv2.getTextSize(name, FONT, 0.5 * k, th)
        cv2.rectangle(img, (x1 - th, y1 - tg - int(8 * k)), (x1 + tw + int(6 * k), y1), color, -1)
        cv2.putText(img, name, (x1 + int(3 * k), y1 - int(4 * k)), FONT, 0.5 * k, (0, 0, 0), th, cv2.LINE_AA)
        ax, ay = anchor_point(d["box"], counter.anchor)
        cv2.circle(img, (int(round(ax)), int(round(ay))), int(4 * k), color, -1, cv2.LINE_AA)

    for i, (a, b) in enumerate(counter.segments):     # segments, red for a moment after a crossing
        last = counter.last_crossing_t[i]
        flash = last is not None and 0 <= t - last < FLASH_S
        draw_segment(img, i, a, b, RED if flash else YELLOW, k, (4 if flash else 2) * th)

    # counts: a table in the top-left corner, people in green and vehicles in orange like their boxes
    counts = counter.counts()
    fs, lh = 0.6 * k, int(28 * k)
    labels = [(f"Line {i + 1}  L>R" if j == 0 else "        R>L", i, d)
              for i in range(len(counter.segments)) for j, d in enumerate(DIRECTIONS)]
    label_w = max(cv2.getTextSize(s, FONT, fs, 2)[0][0] for s, _, _ in labels)
    col_w = cv2.getTextSize("VEHICLES", FONT, fs, 2)[0][0]
    x0, y0 = int(10 * k), int(10 * k)
    pw, ph = int(label_w + 2 * col_w + 50 * k), int(lh * (len(labels) + 1) + 12 * k)
    roi = img[y0:y0 + ph, x0:x0 + pw]
    roi[:] = (roi * 0.35).astype(np.uint8)            # darken the panel background
    cx_p, cx_v = x0 + int(label_w + col_w + 20 * k), x0 + int(label_w + 2 * col_w + 40 * k)
    y = y0 + lh
    put_text(img, "PEOPLE", (cx_p, y), fs, BGR["person"], 2, right=True)
    put_text(img, "VEHICLES", (cx_v, y), fs, BGR["vehicle"], 2, right=True)
    for text, i, direction in labels:
        y += lh
        put_text(img, text, (x0 + int(8 * k), y), fs, YELLOW, 2)
        put_text(img, str(counts[(i, "person", direction)]), (cx_p, y), 0.8 * k, (255, 255, 255), 2, right=True)
        put_text(img, str(counts[(i, "vehicle", direction)]), (cx_v, y), 0.8 * k, (255, 255, 255), 2, right=True)

    info = f"{camera}  frame {frame_no}"       # status line at the bottom
    if fps is not None:
        info += f"  {fps:.1f} fps"
    if box_ms is not None:
        info += f"  box {box_ms:.0f} ms"
    put_text(img, info, (10 * k, h - 10 * k), 0.55 * k, (255, 255, 255), th)
    put_text(img, "q = quit", (w - 10 * k, h - 10 * k), 0.55 * k, (255, 255, 255), th, right=True)
    if status:
        put_text(img, status, (w - 10 * k, 30 * k), 0.7 * k, RED, 2, right=True)
    return img


def fit_to_screen(img, max_width):
    """(view, scale): the image shrunk to at most max_width pixels wide."""
    scale = min(1.0, max_width / img.shape[1])
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img, scale


# =================================================================================================
# Clicking the segments
# =================================================================================================
def pick_segments(image, n, max_width=1280, win="VisionIST - click the line segments"):
    """Show the image; every two left-clicks make one segment (A then B).  Returns
    [((xA, yA), (xB, yB)), ...] in image pixels once n segments are done and Enter is pressed.
    Keys: u = undo, r = start over, Enter / Space = done, q / Esc = give up."""
    base, scale = fit_to_screen(image, max_width)
    segments, pending = [], []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(segments) < n:
            pending.append((x / scale, y / scale))
            if len(pending) == 2:
                segments.append(tuple(pending))
                pending.clear()

    def render():
        view = base.copy()
        for i, (a, b) in enumerate(segments):
            draw_segment(view, i, (a[0] * scale, a[1] * scale), (b[0] * scale, b[1] * scale), YELLOW, 1.0, 2)
        for a in pending:
            cv2.circle(view, (int(a[0] * scale), int(a[1] * scale)), 6, YELLOW, -1, cv2.LINE_AA)
        if len(segments) >= n:
            msg = f"{n} segment(s) done - Enter = start, u = undo, r = start over"
        else:
            msg = f"segment {len(segments) + 1}/{n}: click {'B (end)' if pending else 'A (start)'}   u = undo, q = quit"
        put_text(view, msg, (10, 24), 0.65, YELLOW, 2)
        return view

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    try:
        while True:
            cv2.imshow(win, render())
            key = cv2.waitKey(30) & 0xFF
            if key in (13, 10, 32) and len(segments) >= n and not pending:
                return list(segments)
            if key in (ord("u"), 8, 127):
                if pending:
                    pending.pop()
                elif segments:
                    segments.pop()
            elif key == ord("r"):
                segments.clear()
                pending.clear()
            elif key in (ord("q"), 27):
                raise SystemExit("cancelled while clicking the segments")
    finally:
        cv2.destroyWindow(win)
        cv2.waitKey(1)


def save_segments(path, camera, segments, size):
    """Same file as the notebook's segments.json (plus the image size, to catch a mismatch later)."""
    with open(path, "w") as f:
        json.dump({"camera": camera, "size": list(size), "segments": segments}, f)


def load_segments(path, size):
    with open(path) as f:
        data = json.load(f)
    if "size" in data and tuple(data["size"]) != tuple(size):
        sys.exit(f"{path} was drawn on a {data['size'][0]}x{data['size'][1]} image but the stream is "
                 f"{size[0]}x{size[1]}: click the segments again (drop --reuse-segments)")
    return [tuple(map(tuple, s)) for s in data["segments"]]


# =================================================================================================
# The application
# =================================================================================================
def summary_lines(counter):
    counts = counter.counts()
    return [f"line {i + 1}:  " + "   ".join(
        f"{g}s {counts[(i, g, 'left_to_right')]} left>right, {counts[(i, g, 'right_to_left')]} right>left"
        for g in ("person", "vehicle")) for i in range(len(counter.segments))]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Count the people and vehicles that cross a line in an RTSP stream (yolo box).",
        epilog="Click two points per segment on the first image.  Keys in the window: q / Esc = quit.")
    p.add_argument("url", nargs="?", help="RTSP URL (or set $RTSP_URL, which keeps the password out of `ps`)")
    p.add_argument("--yolo", default="localhost:9068", help="address of the yolo box [%(default)s]")
    p.add_argument("--camera", default="cam1", help="camera name written to the CSV [%(default)s]")
    p.add_argument("--segments", type=int, default=1, help="number of line segments to click [%(default)s]")
    p.add_argument("--segments-file", default="segments.json", help="where the clicked segments are saved [%(default)s]")
    p.add_argument("--reuse-segments", action="store_true", help="read the segments from --segments-file instead of clicking")
    p.add_argument("--csv", default="detections.csv", help="detections CSV, updated after every frame [%(default)s]")
    p.add_argument("--counts-csv", default=None,
                   help="counts CSV, one row per counted crossing [<--csv name>_counts.csv]")
    p.add_argument("--append", action="store_true", help="continue existing CSVs (default: keep them under a time-stamped name)")
    p.add_argument("--fps", type=float, default=5.0, help="frames per second sent to the box [%(default)s]")
    p.add_argument("--conf", type=float, default=0.30, help="minimum detection confidence [%(default)s]")
    p.add_argument("--weights", default=None, help="checkpoint for the box (default: the box's COCO model)")
    p.add_argument("--anchor", choices=("bottom", "center", "top"), default="bottom",
                   help="point of the box that is tracked [%(default)s]")
    p.add_argument("--min-points", type=int, default=3, help="ignore tracks with fewer detections [%(default)s]")
    p.add_argument("--duration", type=float, default=0, help="stop after this many seconds (0 = until you quit)")
    p.add_argument("--no-display", action="store_true", help="no windows (needs --reuse-segments)")
    p.add_argument("--display-width", type=int, default=1280, help="largest window width, pixels [%(default)s]")
    p.add_argument("--rtsp-transport", choices=("tcp", "udp"), default="tcp",
                   help="tcp is the safe choice; udp has less delay but drops packets [%(default)s]")
    p.add_argument("--jpeg-quality", type=int, default=90, help="[%(default)s]")
    p.add_argument("--timeout", type=float, default=10.0, help="seconds to wait for the box per frame [%(default)s]")
    args = p.parse_args(argv)
    args.url = args.url or os.environ.get("RTSP_URL")
    if not args.url:
        p.error("give the RTSP URL as an argument or in $RTSP_URL")
    if not args.counts_csv:
        args.counts_csv = os.path.splitext(args.csv)[0] + "_counts.csv"
    if os.path.abspath(args.counts_csv) == os.path.abspath(args.csv):
        p.error("--counts-csv and --csv are the same file")
    if args.no_display and not args.reuse_segments:
        p.error("--no-display needs --reuse-segments: there is nowhere to click")
    if args.reuse_segments and not os.path.exists(args.segments_file):
        p.error(f"--reuse-segments: {args.segments_file} does not exist (run once without it and click the segments)")
    if args.segments < 1:
        p.error("--segments must be at least 1")
    return args


def _terminate(signum, frame):
    raise KeyboardInterrupt


def main(argv=None):
    args = parse_args(argv)
    signal.signal(signal.SIGTERM, _terminate)                       # `kill` / docker stop end the run cleanly
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{args.rtsp_transport}"   # before any capture
    show = not args.no_display
    stem = os.path.splitext(args.csv)[0]

    print(f"opening {mask_url(args.url)} ...")
    try:
        reader = RTSPReader(args.camera, args.url)
    except RuntimeError as e:
        sys.exit(str(e))
    log = clog = detector = counter = None
    n_frames = n_errors = 0
    t_start = time.time()
    try:
        first = reader.latest()[0]
        size = (first.shape[1], first.shape[0])
        print(f"{args.camera}: open, {size[0]}x{size[1]}")
        cv2.imwrite(f"{stem}_{args.camera}.jpg", first)       # the picture the segments are drawn on (as in the notebook)

        if args.reuse_segments:
            segments = load_segments(args.segments_file, size)
        else:
            print(f"click {args.segments} segment(s) on the image: two points each, then Enter")
            segments = pick_segments(first, args.segments, args.display_width)
            save_segments(args.segments_file, args.camera, segments, size)
        for k, (a, b) in enumerate(segments, 1):
            print(f"segment {k}: A=({a[0]:.0f}, {a[1]:.0f})  B=({b[0]:.0f}, {b[1]:.0f})")

        counter = CrossingCounter(segments, args.anchor, args.min_points)
        run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")          # a fresh tracker session for every run
        detector = YoloDetector(args.yolo, f"{run_id}-{args.camera}", args.conf, args.weights,
                                args.jpeg_quality, args.timeout)
        log = CsvLog(args.csv, args.append)
        clog = CsvLog(args.counts_csv, args.append, COUNT_COLUMNS)
        win = "VisionIST - line crossing counter"
        if show:
            cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        print(f"counting; detections go to {args.csv}, counted crossings to {args.counts_csv}; "
              + ("press q in the window to stop" if show else "Ctrl-C to stop"))

        period = 1.0 / args.fps
        last_seq, last_new, stalled = -1, time.time(), False
        prev_t0, fps, box_ms, status, epoch = None, None, None, None, 0
        t_start = time.time()
        while not (args.duration and time.time() - t_start >= args.duration):
            t0 = time.time()
            got = reader.latest()
            if got is None or got[2] == last_seq:                    # nothing new since the last iteration
                if time.time() - last_new > STALL_WARN_S and not stalled:
                    print(f"no new frame from the camera for {STALL_WARN_S:.0f} s (reconnecting)")
                    stalled = True
                    counter.break_tracks()
                if show:
                    if cv2.waitKey(10) & 0xFF in (ord("q"), 27):
                        break
                else:
                    time.sleep(0.01)
                continue
            frame, t, last_seq = got
            last_new, stalled = time.time(), False

            dets, rec = [], None
            try:
                tb = time.time()
                rec = detector.detect(frame)
                box_ms = (time.time() - tb) * 1000
                dets = parse_detections(rec, detector.id_offset)
                status = None
                if detector.epoch != epoch:
                    epoch = detector.epoch
                    print(f"yolo box answering again: new tracker session {detector.session_id}, "
                          f"track ids are now offset by {detector.id_offset}")
            except Exception as e:                                   # one bad frame must not end the run
                n_errors += 1
                status = "yolo box not answering"
                if n_errors <= 3 or n_errors % 50 == 0:
                    print(f"{short_error(e)}  ({n_errors} so far)")
            if rec is not None:
                n_frames += 1
                log.write_frame(csv_rows(rec, dets, args.camera, n_frames, t, frame.shape))
                events = counter.update(t, dets)
                if events:
                    clog.write_frame(count_rows(events, counter.counts(), args.camera, n_frames))
                for ev in events:
                    when = dt.datetime.fromtimestamp(ev["t"]).strftime("%H:%M:%S")
                    print(f"{when}  line {ev['segment'] + 1}: {ev['label']} #{ev['track_id']} "
                          f"{ev['direction']}" + ("" if ev["first"] else "  (again, not counted)"))
            if prev_t0 is not None:
                inst = 1.0 / max(t0 - prev_t0, 1e-6)
                fps = inst if fps is None else 0.8 * fps + 0.2 * inst
            prev_t0 = t0

            if show:
                out = draw_overlay(frame, counter, dets, t, args.camera, n_frames, fps, box_ms, status)
                cv2.imshow(win, fit_to_screen(out, args.display_width)[0])
                if cv2.waitKey(max(1, int(1000 * (period - (time.time() - t0))))) & 0xFF in (ord("q"), 27):
                    break
            else:
                time.sleep(max(0.0, period - (time.time() - t0)))
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        for f in (log, clog):
            if f is not None:
                f.close()
        if detector is not None:
            detector.close()
        reader.close()
        if show:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
    elapsed = time.time() - t_start
    print(f"{elapsed:.0f} s, {n_frames} frames ({n_frames / max(elapsed, 1e-9):.1f}/s), {n_errors} box errors"
          + (f", detections in {log.path}" if log is not None else "")
          + (f", counted crossings in {clog.path}" if clog is not None else ""))
    if counter is not None:
        print("\n".join(summary_lines(counter)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

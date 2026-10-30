# Applications

Scripts that use the boxes of the fleet to do something useful, one file each.
They talk to the boxes through [`visionist_client`](../visionist_client/) like
the notebooks do, and are meant to be read, copied and changed.

```bash
pip install -r requirements.txt            # visionist-client, opencv-python, numpy
cd ../fleet && docker compose up -d yolo   # the box the first application needs (port 9068)
```

| script | what it does | box |
|---|---|---|
| [`line_crossing_counter.py`](line_crossing_counter.py) | counts the people and vehicles that cross a line in an RTSP camera stream, live | yolo |
| [`crossing_counter.html`](crossing_counter.html) | a one-file viewer for what the counter recorded: open the detections CSV of a camera (or a whole folder of them), draw lines on its picture and see the people and vehicles that crossed each line, in both directions, over time | none (open it in a browser) |
| [`rtsp_probe.py`](rtsp_probe.py) | shows what an RTSP camera really sends (codec, fps, key frames, jitter, what time information it carries) and whether several streams can be synchronized | none |

## line_crossing_counter.py

[`notebooks/people_vehicle_crossings.ipynb`](../notebooks/people_vehicle_crossings.ipynb)
as a live application: it reads an RTSP stream, tracks people and vehicles with the
yolo box, counts how many of them cross each line segment in each direction, and shows the
camera image with the annotations and the current counts.

```bash
python line_crossing_counter.py rtsp://user:password@192.168.1.101:554/stream1
python line_crossing_counter.py $URL --segments 2 --fps 5 --csv gate.csv      # two lines: gate.csv and gate_counts.csv
RTSP_URL=$URL python line_crossing_counter.py --reuse-segments --no-display   # unattended
```

**Drawing the lines.** The first image of the stream opens in a window. Click two points for
each segment, A (start) and then B (end); the arrow A→B is what *left* and *right* refer to
(standing at A and looking at B). `u` undoes, `r` starts over, **Enter** starts counting. The
segments are saved to `segments.json` (the notebook's format), so the next run can skip the
clicking with `--reuse-segments`.

**The window.** Boxes and ids, a trail for every track, the segments (red for a second after a
crossing, with `L` and `R` on their two sides), and a table with the people and vehicles that have
crossed each segment, `L>R` and `R>L`. `q` or Esc quits. Crossings are also printed.

**The CSV.** `detections.csv` has the notebook's columns (`timestamp, t, camera, frame, track_id,
class_id, label, score, x1, y1, x2, y2, width, height`), gets its rows after **every** frame and is
flushed, so it is always up to date and survives the script being killed. A frame with no detections adds no row.
An existing file is kept under a time-stamped name, or continued with `--append`. The first image
is saved beside it as `detections_cam1.jpg`, so the notebook's sections 3 and 4 can recount
the CSV offline and give the same numbers.

**The counts CSV.** `detections_counts.csv` (`--counts-csv`, by default the name of `--csv` plus
`_counts`) gets **one row for every counted crossing**, also flushed at once:

| column | meaning |
|---|---|
| `timestamp`, `t` | when the track met the line (UTC, ISO 8601 with milliseconds; epoch seconds) |
| `camera`, `frame` | `--camera`, and the number of the frame in which the crossing became countable |
| `line` | the segment, numbered from 1 as in the window |
| `direction` | `left_to_right` or `right_to_left`, standing at A and looking at B |
| `group`, `label` | `person` or `vehicle`, and the label of the track (`car`, `truck`, ...) |
| `track_id` | the same id as in the detections CSV, so a crossing can be looked up there |
| `x`, `y` | where the track met the line, in image pixels |
| `person_left_to_right`, `person_right_to_left`, `vehicle_left_to_right`, `vehicle_right_to_left` | the totals of that line after this frame; the last row of a line holds its final counts |

A track that crosses the same line the same way again is not counted again and has no row. Rows
are in the order the crossings became countable, which can differ from the order of `t` by a moment
(a crossing made in the first frames of a track is counted once the track has `--min-points`
detections), so sort by `t` before plotting. With `--append` the totals start from 0 again at every
run: add the rows of the runs instead of reading the totals. Counting the rows per
minute or hour, by `line`, `group` and `direction`, gives the counts over time.

**How it counts** (the notebook's rules, applied frame by frame):

- a track's position is the bottom centre of its box (`--anchor bottom|center|top`);
- it crosses a segment when the step between two consecutive positions cuts the *finite* segment;
- a track has one class, the label it has had most often; people and vehicles
  (bicycle, car, motorcycle, airplane, bus, train, truck, boat) are counted separately;
- a track with fewer than `--min-points` (3) detections is ignored; a crossing made early in a track
  is counted as soon as the track is long enough;
- the counts are distinct tracks: one that crosses back and forth counts once in each direction.

**Good to know**

- Frames are taken at `--fps` (5) from the *newest* image of the stream, so a slow box makes it
  skip frames rather than fall behind. Raise `--fps` if fast objects keep getting new ids.
- If a call to the box fails the application keeps going (a red message in the window) and, when
  the box answers again, **starts a new tracker session**: a restarted box numbers its tracks from 1
  again and would otherwise pair new objects with old tracks. Track ids of the new session are
  offset by 1 000 000 000 per restart, in the CSV too, so ids stay unique within a file. Objects that
  cross while the box is away cannot be counted.
- If the camera stalls for 5 s, no step is measured across the gap. The stream reconnects by itself.
- The password of the URL is never printed. Prefer `$RTSP_URL` to the argument, which shows in `ps`.
- `--rtsp-transport udp` has less delay than `tcp` but drops packets; `--duration S` stops after S seconds.
- Run it from a desktop session: the windows need a display (`--no-display` with `--reuse-segments` does not).

## crossing_counter.html

Open it in a browser (it is one file, nothing to install, and nothing leaves your computer). It works on
**one camera at a time**:

1. **Open CSV files…** or **Open folder…** (or drop them on the page). Every detections CSV is listed with its
   camera, its time span (Lisbon time) and its size; click the one to show. Files with no rows and the
   `_counts.csv` files are listed but cannot be opened. A big file takes a few seconds (622 MB: about 4 s).
2. Draw lines on the camera picture (drag; drag a handle to change a line). The picture is the
   `<csv name>_<camera>.jpg` that `line_crossing_counter.py` saves next to the CSV; **Add picture…** puts another
   one, and the page remembers it.
3. Each line gets a card with the vehicles and the people that crossed it *with the arrow* and *against it*,
   per bin of time (10 s to 2 h, depending on how long the recording is), above a timeline of the whole recording.
   Drag across a timeline to zoom.

The lines come from a `segments.json` in the same folder as the CSV (put it there, or select it together with the
CSV), otherwise from the last recording of the same camera. They are the notebook's `segments.json` (**Save lines**
writes the same format; call the file `segments.json` to use the lines with `--reuse-segments`). **Save counts (CSV)** writes the
counts per line and bin. The counting rules are the notebook's, applied to every detection of the file.
The first row of the list is a built-in example (one day of cam1), so the page works with no files.

## rtsp_probe.py

Read this before trying to synchronize cameras: it tells you what the cameras can give. It plays one or
more RTSP streams **at the same time** over TCP (Python 3.9+, standard library only, nothing to install) and
prints, for each:

- the codec, picture size, frame rate (measured and as declared), key-frame interval, bit rate, B-frames, lost packets;
- how steadily the frames arrive (jitter), and how much later key frames arrive;
- **what time information the stream carries**: RTCP sender reports (the standard way to learn the
  wall-clock time of a frame - without them the stream has no absolute time), an RFC 7273 clock declaration
  in the SDP, an RTP header extension, SEI messages, and whether the RTP timeline starts at PLAY;
- how fast each stream's clock runs against this computer's, in ppm, and, with several URLs, against the other streams.

```bash
python rtsp_probe.py rtsp://admin:CODE@host:554/Streaming/channels/101 --seconds 30 --ports
python rtsp_probe.py rtsp://admin:CODE@host:554/Streaming/channels/101 \
                     rtsp://admin:CODE@host:554/Streaming/channels/201 --seconds 300 --json probe.json
RTSP_USER=admin RTSP_PASSWORD=CODE python rtsp_probe.py rtsp://host:554/Streaming/channels/101
```

- Two streams that come from **one clock** show no rate difference; two clocks drift apart by tens of ppm
  (a few ms per minute). The error of that number falls fast with time (about 1/t^1.5): ±10 ppm needs a few minutes.
- With B-frames the RTP time is the display time and the clock and jitter figures are unreliable; the probe says so.
- `--ports` tries the usual camera ports, `--isapi` a few read-only Hikvision-style HTTP GETs, `--sdp` prints the raw SDP.
- It only reads (OPTIONS, DESCRIBE, SETUP, PLAY, TEARDOWN). The password is never printed. The exit status is 1 if a stream failed.

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

## line_crossing_counter.py

[`notebooks/people_vehicle_crossings.ipynb`](../notebooks/people_vehicle_crossings.ipynb)
as a live application: it reads an RTSP stream, tracks people and vehicles with the
yolo box, counts how many of them cross each line segment in each direction, and shows the
camera image with the annotations and the current counts.

```bash
python line_crossing_counter.py rtsp://user:password@192.168.1.101:554/stream1
python line_crossing_counter.py $URL --segments 2 --fps 5 --csv gate.csv      # two lines
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

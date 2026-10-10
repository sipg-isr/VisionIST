# VisionIST
![Visionist](docs/logo.png)

## A thin, box-agnostic client and a browser front end for a fleet of Signal Processing (AI) boxes
A set of independent Dockerized inference services that all speak one gRPC envelope standardized by the [European Platform for AI-on-Demand](http://aiod.eu).

This website provides the python client while the registry of boxes is in [Visionist_Library](http://github.com/sipg-isr/VisionIST_Library). For "old timers" and increased productivity we include a [Matlab Client](http://github.com/sipg-isr/VisionIST_matlab). 

```protobuf
service PipelineService { rpc Process( Envelope ) returns ( Envelope ); }
```

> **The boxes themselves live in
> [VisionIST_Library](https://github.com/sipg-isr/VisionIST_Library).**
> That registry holds one directory per box — Dockerfile, service source,
> manifest — and publishes the images to Docker Hub (`sipgisr/`). This repository
> keeps the client, the webui, the documentation and a reference fleet
> assembled from that registry. Nothing here is built from box source.

## Calling a box

```python
from visionist_client import Visionist
import pathlib

b = Visionist("localhost:9061")                      # any box, by IP:port
res = b.run(data   = {"images": [pathlib.Path("dog.jpg")]},
            config = {"lang_sam": {"command": "segment",
                                   "parameters": {"box_threshold": 0.3},
                                   "text_prompt": ["a dog"]}})
print(res.config)    # namespaced status
print(res.results)   # decoded payload, per the encoding the box declared
```

The client knows no box: it builds an `Envelope`, sends it, and decodes the
reply with whatever codec the box declared. Per-box request shapes are in each
box's README in the Library. Details:
[visionist_client/README.md](visionist_client/README.md).

```bash
pip install visionist-client
```

## Running a fleet

The reference fleet in [`fleet/`](fleet/) is **generated** — every box in the
registry, pulled rather than built:

```bash
cd fleet
docker compose pull
docker compose up -d
docker compose ps
```

For your own selection, regenerate it. You do not need a Library checkout;
the generator reads the published registry index over HTTP:

```bash
python3 tools/make_fleet.py --boxes clip,yolo,moge --out my-fleet --gpu
python3 tools/make_fleet.py --all --runtime cpu --registry dockerhub --out laptop-fleet
cd my-fleet && docker compose pull && docker compose up -d
```

Each box has a permanent host port, issued once in arrival order and
recorded in the registry's
[`ports.json`](https://github.com/sipg-isr/VisionIST_Library/blob/main/registry/ports.json).
A box therefore answers on the same port in every fleet: a partial selection
has gaps rather than renumbered boxes, and adding a box moves nothing.
`fleet/data/fleet.json` records the service-name addresses the webui uses.

**Access**
- webui: **http://localhost:8080** (SPA at `/`, API at `/api/*`)
- boxes: host ports from 9061 up, or by service name on port 8061 inside the
  fleet network

## What is here

| | |
|---|---|
| [`visionist_client/`](visionist_client/) | the box-agnostic Python client (`pip install visionist-client`) |
| [`webui/`](webui/) | browser front end: a declarative layer where box knowledge is YAML, not code |
| [`fleet/`](fleet/) | the generated reference fleet, plus `hello.py` and `supervisor_demo.py` |
| [`tools/make_fleet.py`](tools/make_fleet.py) | build a compose file from the registry |
| [`docs/`](docs/) | architecture, quick start, the envelope reference, codecs |
| [`notebooks/`](notebooks/) | a walkthrough of the fleet with the raw client |
| [`protos/`](protos/) | the envelope, for reference — the source of truth is the Library's `contract/` |

## Key contracts

- **Shared envelope** — one `pipeline.proto` for every box. It is maintained in
  [VisionIST_Library/contract](https://github.com/sipg-isr/VisionIST_Library/tree/main/contract)
  and synced into each box from there.
- **Config dispatch** — the request's `config_json` carries a section named
  after the box; `{"command": "reset"}` is accepted by every standard box.
- **Declared payload encoding** — a box declares `"encoding"` in its reply (a
  codec name, or a `{field: codec}` map) and the client decodes with it:
  `identity` / `json` / `numpy` / `torch` / `zstd_pickle`. Rationale:
  [docs/CODECS.md](docs/CODECS.md).

## Documentation

Start at [docs/index.md](docs/index.md):

- [Architecture overview](docs/Architecture_Overview.md) — what a box is, the
  conventions, GPU memory lifecycle
- [Quick start](docs/Quick_Start_Guide.md) — run a box, call it, write your own
- [gRPC services reference](docs/gRPC_Services_Reference.md) — the envelope
  contract in detail
- [CODECS](docs/CODECS.md) — self-describing payload decoding
- [webui](webui/README.md) — the declarative web layer

## Contributing

A **new box** goes to
[VisionIST_Library](https://github.com/sipg-isr/VisionIST_Library) — see its
CONTRIBUTING.md; `tools/new_box.py` there scaffolds one that already runs.

Changes to the **client**, the **webui** or these **docs** belong here.

## Tests

```bash
# client, no GPU and no boxes needed
python3 visionist_client/tests/fake_box_smoke.py
python3 visionist_client/tests/codec_smoke.py

# webui
cd webui && PYTHONPATH=src python3 -m pytest tests -q
```

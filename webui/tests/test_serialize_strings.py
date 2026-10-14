"""serialize: one rule that bites in practice — string-dtype ndarrays
(numpy codec on arrays of names) must come out as inline lists, not pickle
files."""

import numpy as np

from webui.core import ArtifactStore, serialize_value


def _store():
    return ArtifactStore(ttl=60)


def test_unicode_ndarray_inlines_as_list():
    arr = np.array(["frame_0001.jpg", "frame_0002.jpg", "frame_0003.jpg"])
    out = serialize_value(arr, _store())
    assert out == ["frame_0001.jpg", "frame_0002.jpg", "frame_0003.jpg"]


def test_2d_string_ndarray_inlines_as_nested_list():
    arr = np.array([["a", "b"], ["c", "d"]])
    out = serialize_value(arr, _store())
    assert out == [["a", "b"], ["c", "d"]]


def test_bytes_string_ndarray_inlines_as_str():
    arr = np.array([b"a", b"b"])
    out = serialize_value(arr, _store())
    assert out == ["a", "b"]


def test_numeric_ndarray_still_arrays_or_buffers():
    import json
    small = np.arange(4, dtype=np.float32).reshape(2, 2)
    out = serialize_value(small, _store())
    assert out["kind"] in ("array", "buffer")
    json.dumps(out)          # must stay JSON-safe

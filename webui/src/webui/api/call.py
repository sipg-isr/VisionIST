"""Call routes: upload files, call a box, fetch artifacts."""

from __future__ import annotations

import time

from fastapi import APIRouter, UploadFile, File, Body, Request
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

from ..core import (
    ArtifactMissing, ArtifactStore, CallBuildError, CallRequest,
    Registry, Fleet, build_call, execute, serialize_result,
)
from .errors import WebUIError


class CallBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fleet_id: str
    def_id: str | None = None          # else: entry.def_id, else name match
    data: dict = Field(default_factory=dict)
    parameters: dict = Field(default_factory=dict)
    section: dict = Field(default_factory=dict)
    command: str | None = None
    action: str | None = None
    session_id: str | None = None
    method: str | None = None
    timeout: float = Field(600.0, ge=1.0, le=86400)


def create_call_router(registry: Registry, fleet: Fleet, store: ArtifactStore,
                       max_upload_bytes: int) -> APIRouter:
    r = APIRouter(prefix="/api", tags=["call"])

    @r.post("/upload", status_code=201)
    async def upload(file: UploadFile = File(...)):
        data = await file.read()
        if len(data) == 0:
            raise WebUIError(400, {"message": "empty upload"})
        if len(data) > max_upload_bytes:
            raise WebUIError(413, {"message": f"upload exceeds {max_upload_bytes} bytes"})
        token = store.add(data, file.content_type or "application/octet-stream",
                          extra={"filename": file.filename or ""})
        return {
            "token": token,
            "ref": "@" + token,
            "size": len(data),
            "content_type": file.content_type or "application/octet-stream",
        }

    @r.get("/file/{token}")
    def file(token: str, request: Request):
        """Serve an artifact, with HTTP `Range` / `206 Partial Content`.

        HTML5 `<video>` seeks by sending `Range: bytes=<start>-<end>`;
        a server that only answers full `200` makes the player bounce,
        re-download and can refuse to seek.  Honoring ranges (plus
        `Accept-Ranges` + accurate `Content-Length`) is what makes the
        annotated mp4 scrub in the browser's native player."""
        try:
            art = store.get(token)
        except ArtifactMissing as e:
            raise WebUIError(404, {"token": token}) from e
        extra = {k: v for k, v in art.extra.items() if k in ("dtype", "shape", "pytype", "note")}
        headers = {f"x-artifact-{k}": str(v) for k, v in extra.items()}
        data = art.data
        total = len(data)
        headers["Accept-Ranges"] = "bytes"

        range_hdr = request.headers.get("range")
        if range_hdr and range_hdr.startswith("bytes="):
            # only the first single-range spec is honoured (players send one)
            spec = range_hdr.split("bytes=", 1)[1].split(",")[0].strip()
            try:
                start_s, _, end_s = spec.partition("-")
                if start_s == "":
                    # suffix range `bytes=-N`: the final N bytes
                    length = int(end_s or 0)
                    start, end = max(0, total - length), total - 1
                else:
                    start = int(start_s)
                    end = min(int(end_s), total - 1) if end_s else total - 1
            except (ValueError, TypeError):
                start, end = 0, total - 1
            if start >= total or start > end:
                return Response(status_code=416,
                                headers={"Content-Range": f"bytes */{total}"})
            chunk = data[start:end + 1]
            headers.update({
                "Content-Range": f"bytes {start}-{end}/{total}",
                "Content-Length": str(len(chunk)),
            })
            return Response(chunk, media_type=art.content_type,
                            status_code=206, headers=headers)

        headers["Content-Length"] = str(total)
        return Response(data, media_type=art.content_type, headers=headers)

    @r.post("/call")
    def call(body: CallBody):
        # ---- resolve fleet entry + definition -----------------------------
        try:
            entry = fleet.get(body.fleet_id)
        except KeyError as e:
            raise WebUIError(404, {"message": str(e.args[0])}) from e
        def_id = body.def_id or entry.def_id
        defn = registry.match(def_id) if def_id else None
        if defn is None:
            candidates = registry.match(entry.name) or registry.match(entry.id)
            if candidates is None:
                raise WebUIError(
                    400,
                    {"message": f"no box definition matches {def_id or entry.name!r}",
                     "known": [d.id for d in registry]})
            defn, def_id = candidates, candidates.id

        req = CallRequest(
            data=body.data, parameters=body.parameters, section=body.section,
            command=body.command, action=body.action,
            session_id=body.session_id, method=body.method,
        )

        # ---- build (pure; may raise CallBuildError) ------------------------
        try:
            spec = build_call(defn, req, store)
        except CallBuildError as e:
            raise WebUIError(400, {"message": str(e), **e.detail}) from e
        except ArtifactMissing as e:
            raise WebUIError(400, {"message": f"unknown upload token in data: {e.args[0]}"}) from e

        # ---- execute + serialize ------------------------------------------
        before = set(store.tokens())
        started = time.time()
        try:
            result = execute(spec, entry.addr, defn, timeout=body.timeout)
        except WebUIError:
            raise
        except Exception as e:  # noqa: BLE001 — report, don't crash
            raise WebUIError(
                502,
                {"stage": "grpc", "box": defn.id, "addr": entry.addr,
                 "method": spec.method, "error": f"{type(e).__name__}: {e}"},
            ) from e

        payload = serialize_result(result, store)
        produced = [store.public_view(t) for t in store.tokens() if t not in before]
        payload.update({
            "box": defn.id,
            "addr": entry.addr,
            "method": spec.method,
            "action": spec.action,
            "session_id": spec.session_id,
            "duration_ms": round((time.time() - started) * 1000.0, 1),
            "artifacts": produced,
        })
        return payload

    return r


__all__ = ["create_call_router"]

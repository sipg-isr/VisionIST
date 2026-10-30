#!/usr/bin/env python3
"""What does this RTSP camera really give, and can its streams be synchronized?

Standard library only (Python 3.9+).  Opens each URL with a small RTSP client over TCP, plays it for a
while and reports what the camera sends: the codec, resolution and frame rate, the key-frame interval,
the bit rate, how steadily the frames arrive, and - the part that matters for synchronizing two streams -
what *time* information the stream carries:

  * RTCP sender reports (RTP time <-> NTP wall clock): the standard way to get an absolute capture time.
    A camera that sends none gives you no absolute time at all through RTSP.
  * an RFC 7273 clock declaration in the SDP, an RTP header extension, SEI messages.
  * how the RTP clock of each stream runs against this computer's clock (ppm): two streams from two
    clocks drift apart, two streams from one clock do not.

Give several URLs to probe them at the same time (e.g. the two lenses of one camera, or two cameras): the
last section compares them.

    python rtsp_probe.py rtsp://admin:CODE@host:554/Streaming/channels/101 \\
                         rtsp://admin:CODE@host:554/Streaming/channels/201 --seconds 60
    RTSP_USER=admin RTSP_PASSWORD=CODE python rtsp_probe.py rtsp://host:554/Streaming/channels/101

A drift of tens of ppm needs minutes to be measured: use --seconds 300 for the comparison.
Only reads: it sends OPTIONS, DESCRIBE, SETUP, PLAY and TEARDOWN (and, with --isapi, a few HTTP GETs).
"""

import argparse
import base64
import hashlib
import http.client
import json
import math
import os
import random
import re
import socket
import statistics
import struct
import sys
import threading
import time
from collections import Counter
from urllib.parse import unquote, urlsplit

NTP_TO_UNIX = 2208988800                  # seconds from 1900-01-01 to 1970-01-01
SR_ABSENT = "none"


# =================================================================================================
# small helpers
# =================================================================================================
def mask_url(url):
    return re.sub(r"//[^@/]*@", "//***@", url)


def md5(s):
    return hashlib.md5(s.encode()).hexdigest()


def parse_kv(s):
    """key=value pairs of an authentication challenge."""
    return {m.group(1).lower(): (m.group(3) if m.group(3) is not None else m.group(2))
            for m in re.finditer(r'(\w+)=("([^"]*)"|[^,\s]+)', s)}


def percentile(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def linear_fit(xs, ys):
    """y = a + b x by least squares.  Returns (a, b, se_b, residuals)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    res = [y - (a + b * x) for x, y in zip(xs, ys)]
    se = math.sqrt(sum(r * r for r in res) / max(n - 2, 1) / sxx)
    return a, b, se, res


class BitReader:
    def __init__(self, data):
        self.val, self.n, self.pos = int.from_bytes(data, "big"), 8 * len(data), 0

    def u(self, k):
        if self.pos + k > self.n:
            raise ValueError("ran out of bits")
        v = (self.val >> (self.n - self.pos - k)) & ((1 << k) - 1)
        self.pos += k
        return v

    def ue(self):
        zeros = 0
        while self.u(1) == 0:
            zeros += 1
            if zeros > 32:
                raise ValueError("bad exp-golomb")
        return (1 << zeros) - 1 + (self.u(zeros) if zeros else 0)

    def se(self):
        k = self.ue()
        return (k + 1) // 2 if k % 2 else -(k // 2)


def unescape(nal):
    return re.sub(b"\x00\x00\x03", b"\x00\x00", nal)


def _skip_scaling_list(r, size):
    last = nxt = 8
    for _ in range(size):
        if nxt:
            nxt = (last + r.se() + 256) % 256
        last = nxt or last


def parse_h264_sps(nal):
    """(width, height, profile_idc, level_idc, fps or None) from an H.264 SPS NAL unit."""
    r = BitReader(unescape(nal[1:]))
    profile, _, level = r.u(8), r.u(8), r.u(8)
    r.ue()
    chroma = 1
    if profile in (100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135):
        chroma = r.ue()
        if chroma == 3:
            r.u(1)
        r.ue(); r.ue(); r.u(1)
        if r.u(1):
            for i in range(8 if chroma != 3 else 12):
                if r.u(1):
                    _skip_scaling_list(r, 16 if i < 6 else 64)
    r.ue()
    poc = r.ue()
    if poc == 0:
        r.ue()
    elif poc == 1:
        r.u(1); r.se(); r.se()
        for _ in range(r.ue()):
            r.se()
    r.ue(); r.u(1)
    w_mbs, h_units, frame_mbs_only = r.ue() + 1, r.ue() + 1, r.u(1)
    if not frame_mbs_only:
        r.u(1)
    r.u(1)
    crop = (0, 0, 0, 0)
    if r.u(1):
        crop = (r.ue(), r.ue(), r.ue(), r.ue())
    cux = 1 if chroma in (0, 3) else 2
    cuy = (1 if chroma in (0, 2, 3) else 2) * (2 - frame_mbs_only)
    width = w_mbs * 16 - cux * (crop[0] + crop[1])
    height = (2 - frame_mbs_only) * h_units * 16 - cuy * (crop[2] + crop[3])
    fps = None
    try:                                              # VUI: the frame rate the encoder was set to
        if r.u(1):
            if r.u(1) and r.u(8) == 255:
                r.u(32)
            if r.u(1):
                r.u(1)
            if r.u(1):
                r.u(4)
                if r.u(1):
                    r.u(24)
            if r.u(1):
                r.ue(); r.ue()
            if r.u(1):
                units, scale = r.u(32), r.u(32)
                if units:
                    fps = scale / (2.0 * units)
    except ValueError:
        pass
    return width, height, profile, level, fps


def parse_h265_sps(nal):
    """(width, height, profile_idc, level_idc, None) from an H.265 SPS NAL unit."""
    r = BitReader(unescape(nal[2:]))
    r.u(4)
    sub = r.u(3)
    r.u(1)
    r.u(2); r.u(1)
    profile = r.u(5)
    r.u(32); r.u(4); r.u(32); r.u(11); r.u(1)
    level = r.u(8)
    present = [(r.u(1), r.u(1)) for _ in range(sub)]
    if sub:
        for _ in range(sub, 8):
            r.u(2)
    for p, lv in present:
        if p:
            r.u(88)
        if lv:
            r.u(8)
    r.ue()
    chroma = r.ue()
    if chroma == 3:
        r.u(1)
    width, height = r.ue(), r.ue()
    if r.u(1):
        left, right, top, bottom = r.ue(), r.ue(), r.ue(), r.ue()
        sw = 2 if chroma in (1, 2) else 1
        sh = 2 if chroma == 1 else 1
        width -= sw * (left + right)
        height -= sh * (top + bottom)
    return width, height, profile, level, None


# =================================================================================================
# RTSP client (TCP interleaved)
# =================================================================================================
class RtspError(Exception):
    pass


class Response:
    def __init__(self, status, reason, headers, body):
        self.status, self.reason, self.headers, self.body = status, reason, headers, body

    def h(self, name, default=None):
        v = self.headers.get(name.lower())
        return v[0] if v else default


class RtspClient:
    def __init__(self, url, user=None, password=None, timeout=8.0):
        p = urlsplit(url)
        self.host, self.port = p.hostname, p.port or 554
        self.user = unquote(p.username) if p.username else user
        self.password = unquote(p.password) if p.password is not None else password
        self.uri = f"rtsp://{self.host}:{self.port}{p.path or '/'}" + (f"?{p.query}" if p.query else "")
        self.cseq, self.session, self.session_timeout = 0, None, 60
        self.auth, self.nc = None, 0
        self.auth_scheme = None
        self.on_interleaved = None
        self.sock = socket.create_connection((self.host, self.port), timeout)
        self.sock.settimeout(timeout)
        self.buf = bytearray()

    # -- authentication ---------------------------------------------------------------------------
    def _authorization(self, method, uri):
        if not self.auth or self.user is None:
            return None
        if self.auth[0] == "basic":
            return "Basic " + base64.b64encode(f"{self.user}:{self.password or ''}".encode()).decode()
        p = self.auth[1]
        ha1 = md5(f"{self.user}:{p['realm']}:{self.password or ''}")
        ha2 = md5(f"{method}:{uri}")
        qops = [q.strip() for q in p.get("qop", "").split(",")]
        head = (f'Digest username="{self.user}", realm="{p["realm"]}", nonce="{p["nonce"]}", uri="{uri}"')
        if "auth" in qops:
            self.nc += 1
            nc, cnonce = f"{self.nc:08x}", "%08x" % random.getrandbits(32)
            resp = md5(f"{ha1}:{p['nonce']}:{nc}:{cnonce}:auth:{ha2}")
            head += f', response="{resp}", qop=auth, nc={nc}, cnonce="{cnonce}"'
        else:
            resp = md5(ha1 + ":" + p["nonce"] + ":" + ha2)
            head += f', response="{resp}"'
        if "opaque" in p:
            head += f', opaque="{p["opaque"]}"'
        return head

    def _learn_auth(self, resp):
        challenges = resp.headers.get("www-authenticate", [])
        digest = next((c for c in challenges if c.lower().startswith("digest")), None)
        basic = next((c for c in challenges if c.lower().startswith("basic")), None)
        if digest:
            self.auth, self.auth_scheme = ("digest", parse_kv(digest[6:])), "Digest (realm %r)" % parse_kv(digest[6:]).get("realm")
        elif basic:
            self.auth, self.auth_scheme = ("basic", None), "Basic"
        else:
            return False
        self.nc = 0
        return True

    # -- wire -------------------------------------------------------------------------------------
    def _fill(self, need):
        while len(self.buf) < need:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise RtspError("the camera closed the connection")
            self.buf += chunk

    def read_message(self):
        """('rtp', channel, bytes, arrival_ns) or ('rtsp', Response)."""
        self._fill(1)
        if self.buf[0] == 0x24:                                       # '$' interleaved frame
            self._fill(4)
            ch, ln = self.buf[1], struct.unpack("!H", self.buf[2:4])[0]
            self._fill(4 + ln)
            t = time.monotonic_ns()
            data = bytes(self.buf[4:4 + ln])
            del self.buf[:4 + ln]
            return ("rtp", ch, data, t)
        while b"\r\n\r\n" not in self.buf:
            self._fill(len(self.buf) + 1)
        end = self.buf.index(b"\r\n\r\n") + 4
        head = bytes(self.buf[:end]).decode("latin-1")
        lines = head.split("\r\n")
        m = re.match(r"RTSP/1\.0 (\d+) ?(.*)", lines[0])
        if not m:
            raise RtspError(f"unexpected data from the camera: {lines[0][:60]!r}")
        headers = {}
        for ln in lines[1:]:
            if ":" in ln:
                k, v = ln.split(":", 1)
                headers.setdefault(k.strip().lower(), []).append(v.strip())
        length = int((headers.get("content-length") or ["0"])[0])
        self._fill(end + length)
        body = bytes(self.buf[end:end + length])
        del self.buf[:end + length]
        return ("rtsp", Response(int(m.group(1)), m.group(2), headers, body))

    def request(self, method, uri=None, headers=None):
        uri = uri or self.uri
        for attempt in range(2):
            self.cseq += 1
            h = {"CSeq": str(self.cseq), "User-Agent": "rtsp_probe/1.0"}
            if self.session:
                h["Session"] = self.session
            auth = self._authorization(method, uri)
            if auth:
                h["Authorization"] = auth
            h.update(headers or {})
            self.sock.sendall((f"{method} {uri} RTSP/1.0\r\n" + "".join(f"{k}: {v}\r\n" for k, v in h.items()) + "\r\n").encode())
            while True:
                msg = self.read_message()
                if msg[0] == "rtp":
                    if self.on_interleaved:
                        self.on_interleaved(*msg[1:])
                    continue
                resp = msg[1]
                break
            if resp.status == 401 and attempt == 0 and self.user is not None and self._learn_auth(resp):
                continue
            return resp

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


# =================================================================================================
# SDP
# =================================================================================================
def parse_sdp(text):
    sess, media, cur = {"attrs": []}, [], None
    for line in text.replace("\r", "").split("\n"):
        if len(line) < 2 or line[1] != "=":
            continue
        k, v = line[0], line[2:]
        if k == "m":
            f = v.split()
            cur = {"kind": f[0], "port": f[1], "proto": f[2], "pt": f[3] if len(f) > 3 else "", "attrs": [], "bw": None}
            media.append(cur)
        elif k == "a":
            (cur or sess)["attrs"].append(v)
        elif k == "b" and cur is not None:
            cur["bw"] = v
        elif k in "sic" and cur is None:
            sess[k] = v
    for m in media:
        for a in m["attrs"]:
            if a.startswith("rtpmap:"):
                mm = re.match(r"rtpmap:(\d+) ([\w.\-]+)/(\d+)(?:/(\d+))?", a)
                if mm and mm.group(1) == m["pt"]:
                    m["codec"], m["clock"] = mm.group(2).upper(), int(mm.group(3))
                    m["channels"] = int(mm.group(4)) if mm.group(4) else None
            elif a.startswith("fmtp:"):
                m["fmtp"] = a.split(" ", 1)[1] if " " in a else ""
            elif a.startswith("control:"):
                m["control"] = a[8:]
            elif a.startswith("framerate:"):
                m["framerate"] = a[10:]
    return sess, media


def sdp_parameter_sets(media):
    """The SPS found in the SDP, as (codec, nal bytes) or None."""
    fm = media.get("fmtp", "")
    try:
        if media.get("codec") == "H264":
            m = re.search(r"sprop-parameter-sets=([^;\s]+)", fm)
            if m:
                return "H264", base64.b64decode(m.group(1).split(",")[0])
        if media.get("codec") in ("H265", "HEVC"):
            m = re.search(r"sprop-sps=([^;\s]+)", fm)
            if m:
                return "H265", base64.b64decode(m.group(1))
    except Exception:
        pass
    return None


# =================================================================================================
# one stream
# =================================================================================================
class Track:
    def __init__(self, media, index):
        self.media, self.index = media, index
        self.kind = media["kind"]
        self.codec = media.get("codec", "?")
        self.clock = media.get("clock") or 90000
        self.rtp_ch = self.rtcp_ch = None
        self.n_packets = self.bytes = self.gaps = 0
        self.last_seq = None
        self.ssrcs = set()
        self.ext = Counter()                      # RTP header extension profile ids
        self.ext_sample = None
        self.nals = Counter()
        self.sei_sample = []
        self.sps = None                           # (w, h, profile, level, fps)
        self.sps_source = None
        self.cur = None                           # frame being assembled
        self.frames = []                          # (rtp_ts_unwrapped, t_first_ns, t_last_ns, bytes, key)
        self.ts_prev, self.ts_wrap = None, 0
        self.ts0 = None
        self.rtp_info = None
        self.sr = []                              # (host_wall_ns, host_mono_ns, ntp_unix, rtp_ts, packets, octets)
        self.cname = None
        self.bye = False


class Stream:
    def __init__(self, label, url, user, password, seconds):
        self.label, self.url, self.user, self.password, self.seconds = label, url, user, password, seconds
        self.error = None
        self.info = {}
        self.sdp_text = ""
        self.tracks = []
        self.t_play = None
        self.t_end = None
        self.client = None

    # -- the whole conversation ---------------------------------------------------------------------
    def run(self, start_barrier):
        try:
            t0 = time.monotonic()
            self.client = c = RtspClient(self.url, self.user, self.password)
            self.info["connect_ms"] = round((time.monotonic() - t0) * 1000)
            r = c.request("OPTIONS")
            self.info["options_status"] = r.status
            self.info["methods"] = r.h("public", "")
            self.info["server"] = r.h("server", "")
            r = c.request("DESCRIBE", headers={"Accept": "application/sdp"})
            if r.status == 401:
                raise RtspError("401 Unauthorized: wrong user/password (for an EZVIZ camera: the 6-letter verification code, or "
                                "the encryption password if you changed it in the app)")
            if r.status != 200:
                raise RtspError(f"DESCRIBE answered {r.status} {r.reason} (wrong path, or RTSP not enabled in the app?)")
            self.info["auth"] = c.auth_scheme or "none"
            self.sdp_text = r.body.decode("utf-8", "replace")
            self.info["content_base"] = r.h("content-base") or r.h("content-location")
            sess, media = parse_sdp(self.sdp_text)
            self.sdp_session = sess
            base = (self.info["content_base"] or c.uri).rstrip("/")
            ch = 0
            for i, m in enumerate(media):
                if m["kind"] not in ("video", "audio"):
                    continue
                t = Track(m, i)
                ctl = m.get("control", "")
                uri = ctl if ctl.startswith("rtsp://") else (c.uri if ctl in ("", "*") else base + "/" + ctl)
                rr = c.request("SETUP", uri, {"Transport": f"RTP/AVP/TCP;unicast;interleaved={ch}-{ch + 1}"})
                if rr.status != 200:
                    raise RtspError(f"SETUP of the {t.kind} track answered {rr.status} {rr.reason}")
                tr = rr.h("transport", "")
                mm = re.search(r"interleaved=(\d+)-(\d+)", tr)
                t.rtp_ch, t.rtcp_ch = (int(mm.group(1)), int(mm.group(2))) if mm else (ch, ch + 1)
                ms = re.search(r"ssrc=([0-9A-Fa-f]+)", tr)
                t.setup_ssrc = ms.group(1) if ms else None
                sh = rr.h("session", "")
                c.session = sh.split(";")[0].strip()
                mt = re.search(r"timeout=(\d+)", sh)
                if mt:
                    c.session_timeout = int(mt.group(1))
                self.tracks.append(t)
                ch += 2
            if not self.tracks:
                raise RtspError("the SDP has no audio or video track")
            by_ch = {}
            for t in self.tracks:
                by_ch[t.rtp_ch], by_ch[t.rtcp_ch] = (t, "rtp"), (t, "rtcp")
            for t in self.tracks:                         # parameter sets from the SDP
                ps = sdp_parameter_sets(t.media)
                if ps:
                    self._set_sps(t, ps[0], ps[1], "SDP")
            c.on_interleaved = lambda ch_, data, ts: self._on_data(by_ch, ch_, data, ts)
            try:
                start_barrier.wait(timeout=60)              # all streams start playing together...
            except threading.BrokenBarrierError:
                pass                                        # ...unless another one failed: then play alone
            r = c.request("PLAY", headers={"Range": "npt=0.000-"})
            if r.status != 200:
                raise RtspError(f"PLAY answered {r.status} {r.reason}")
            self.t_play = time.monotonic_ns()
            self.info["play_range"] = r.h("range")
            ri = r.h("rtp-info", "")
            self.info["rtp_info"] = ri
            for mm in re.finditer(r"url=([^;,]+)(?:;seq=(\d+))?(?:;rtptime=(\d+))?", ri):
                for t in self.tracks:
                    ctl = t.media.get("control", "")
                    if ctl and mm.group(1).endswith(ctl.split("/")[-1]):
                        t.rtp_info = (int(mm.group(2)) if mm.group(2) else None, int(mm.group(3)) if mm.group(3) else None)
            deadline = time.monotonic() + self.seconds
            next_rr, next_keep = time.monotonic() + 2, time.monotonic() + max(5, c.session_timeout / 2)
            c.sock.settimeout(1.0)
            while time.monotonic() < deadline:
                try:
                    msg = c.read_message()
                except socket.timeout:
                    msg = None
                if msg and msg[0] == "rtp":
                    self._on_data(by_ch, msg[1], msg[2], msg[3])
                now = time.monotonic()
                if now >= next_rr:                          # be a good RTCP citizen: some cameras want receiver reports
                    for t in self.tracks:
                        if t.ssrcs:
                            self._send_rr(t)
                    next_rr = now + 2
                if now >= next_keep:
                    c.cseq += 1
                    c.sock.sendall(f"OPTIONS {c.uri} RTSP/1.0\r\nCSeq: {c.cseq}\r\nSession: {c.session}\r\n\r\n".encode())
                    next_keep = now + max(5, c.session_timeout / 2)
            self.t_end = time.monotonic_ns()
            try:
                c.cseq += 1
                c.sock.sendall(f"TEARDOWN {c.uri} RTSP/1.0\r\nCSeq: {c.cseq}\r\nSession: {c.session}\r\n\r\n".encode())
            except OSError:
                pass
        except Exception as e:                                # report, never crash the other streams
            self.error = self._explain(e)
            try:
                start_barrier.abort()
            except Exception:
                pass
        finally:
            if self.client:
                self.client.close()

    def _explain(self, e):
        port = urlsplit(self.url).port or 554
        if isinstance(e, socket.gaierror):
            return f"cannot resolve the host name ({e}): check the address, or the DNS service of the camera"
        if isinstance(e, ConnectionRefusedError):
            return f"connection refused: nothing listens on port {port} (RTSP not enabled in the app? wrong port?)"
        if isinstance(e, (socket.timeout, TimeoutError)):
            return (f"timed out: the host is unreachable, port {port} is not forwarded/blocked, or the camera is asleep "
                    f"(or, after the connection, it stopped answering)")
        return f"{type(e).__name__}: {e}"

    # -- packets --------------------------------------------------------------------------------------
    def _send_rr(self, t):
        ssrc = next(iter(t.ssrcs))
        hi = t.last_seq or 0
        rr = struct.pack("!BBHI", 0x81, 201, 7, 0x7f7f7f7f) + struct.pack("!IIIIII", ssrc, 0, hi, 0, 0, 0)
        try:
            self.client.sock.sendall(b"$" + bytes([t.rtcp_ch]) + struct.pack("!H", len(rr)) + rr)
        except OSError:
            pass

    def _set_sps(self, t, codec, nal, source):
        if t.sps:
            return
        try:
            t.sps = parse_h264_sps(nal) if codec == "H264" else parse_h265_sps(nal)
            t.sps_source = source
        except Exception:
            pass

    def _on_data(self, by_ch, ch, data, t_ns):
        hit = by_ch.get(ch)
        if not hit:
            return
        t, what = hit
        if what == "rtcp":
            self._on_rtcp(t, data, t_ns)
        else:
            self._on_rtp(t, data, t_ns)

    def _on_rtcp(self, t, data, t_ns):
        wall = time.time_ns()
        i = 0
        while i + 4 <= len(data):
            b0, pt, ln = struct.unpack("!BBH", data[i:i + 4])
            end = i + 4 + 4 * ln
            if pt == 200 and end - i >= 28:                    # sender report
                ssrc, nsec, nfrac, rtp, pk, octs = struct.unpack("!IIIIII", data[i + 4:i + 28])
                unix = nsec - NTP_TO_UNIX + nfrac / 2 ** 32
                t.sr.append((wall, t_ns, unix, rtp, pk, octs))
            elif pt == 202:                                    # SDES: CNAME
                m = re.search(rb"\x01([\x01-\x7f])(.{1,64}?)\x00", data[i + 8:end], re.S)
                if m and t.cname is None:
                    t.cname = m.group(2)[:int(m.group(1)[0])].decode("latin-1", "replace")
            elif pt == 203:
                t.bye = True
            i = end if end > i else len(data)

    def _on_rtp(self, t, d, t_ns):
        if len(d) < 12:
            return
        b0, b1, seq, ts, ssrc = struct.unpack("!BBHII", d[:12])
        if b0 >> 6 != 2:
            return
        hl = 12 + 4 * (b0 & 0x0F)
        if b0 & 0x10 and len(d) >= hl + 4:                       # header extension
            prof, words = struct.unpack("!HH", d[hl:hl + 4])
            t.ext[prof] += 1
            if t.ext_sample is None:
                t.ext_sample = d[hl + 4:hl + 4 + min(4 * words, 24)].hex()
            hl += 4 + 4 * words
        end = len(d) - d[-1] if (b0 & 0x20 and d) else len(d)
        payload = d[hl:end]
        t.ssrcs.add(ssrc)
        t.n_packets += 1
        t.bytes += len(payload)
        if t.last_seq is not None and seq != (t.last_seq + 1) & 0xFFFF:
            t.gaps += 1
        t.last_seq = seq
        # unwrap the RTP timestamp
        if t.ts_prev is not None and ts < t.ts_prev and t.ts_prev - ts > 2 ** 31:
            t.ts_wrap += 1
        t.ts_prev = ts
        uts = ts + (t.ts_wrap << 32)
        if t.ts0 is None:
            t.ts0 = uts
        key = False
        if t.kind == "video":
            key = self._scan_nals(t, payload)
        cur = t.cur
        if cur is None or cur["ts"] != uts:
            self._close_frame(t)
            cur = t.cur = {"ts": uts, "t0": t_ns, "t1": t_ns, "bytes": 0, "key": False}
        cur["t1"] = t_ns
        cur["bytes"] += len(payload)
        cur["key"] = cur["key"] or key
        if b1 & 0x80:                                               # marker: last packet of the frame
            self._close_frame(t)

    def _close_frame(self, t):
        if t.cur:
            c = t.cur
            t.frames.append((c["ts"], c["t0"], c["t1"], c["bytes"], c["key"]))
            t.cur = None

    def _scan_nals(self, t, p):
        """Counts NAL unit types; True when the packet starts or carries a key picture."""
        if not p:
            return False
        key = False
        h265 = t.codec in ("H265", "HEVC")
        units = []
        if h265:
            typ = (p[0] >> 1) & 0x3F
            if typ == 48:                                          # aggregation packet
                i = 2
                while i + 2 <= len(p):
                    n = struct.unpack("!H", p[i:i + 2])[0]
                    units.append(p[i + 2:i + 2 + n])
                    i += 2 + n
            elif typ == 49 and len(p) > 3:                         # fragmentation unit
                if p[2] & 0x80:
                    units.append(bytes([(p[0] & 0x81) | ((p[2] & 0x3F) << 1), p[1]]))
            else:
                units.append(p)
            for u in units:
                if len(u) < 2:
                    continue
                n = (u[0] >> 1) & 0x3F
                t.nals[n] += 1
                key = key or 16 <= n <= 21
                if n == 33:
                    self._set_sps(t, "H265", bytes(u), "stream")
                if n in (39, 40) and len(t.sei_sample) < 3:
                    t.sei_sample.append(u[:28].hex())
        else:
            typ = p[0] & 0x1F
            if typ == 24:                                          # STAP-A
                i = 1
                while i + 2 <= len(p):
                    n = struct.unpack("!H", p[i:i + 2])[0]
                    units.append(p[i + 2:i + 2 + n])
                    i += 2 + n
            elif typ == 28 and len(p) > 2:                         # FU-A
                if p[1] & 0x80:
                    units.append(bytes([(p[0] & 0xE0) | (p[1] & 0x1F)]))
            else:
                units.append(p)
            for u in units:
                if not u:
                    continue
                n = u[0] & 0x1F
                t.nals[n] += 1
                key = key or n == 5
                if n == 7:
                    self._set_sps(t, "H264", bytes(u), "stream")
                if n == 6 and len(t.sei_sample) < 3:
                    t.sei_sample.append(u[:28].hex())
        return key


# =================================================================================================
# analysis and report
# =================================================================================================
def analyze_track(s, t):
    """Numbers for one track, relative to this computer's monotonic clock."""
    s0 = s.t_play
    out = {"kind": t.kind, "codec": t.codec, "clock_hz": t.clock, "packets": t.n_packets, "seq_gaps": t.gaps}
    span = (s.t_end - s0) / 1e9 if s.t_end else 0
    out["seconds"] = round(span, 1)
    if t.n_packets == 0:
        return out
    out["kbit_per_s"] = round(t.bytes * 8 / max(span, 1e-9) / 1000, 1)
    fr = t.frames
    out["frames"] = len(fr)
    if t.sps:
        out["resolution"] = f"{t.sps[0]}x{t.sps[1]}"
        out["profile_level"] = f"{t.sps[2]}/{t.sps[3]}"
        out["sps_fps"] = round(t.sps[4], 3) if t.sps[4] else None
        out["sps_source"] = t.sps_source
    out["rtp_ts0"] = t.ts0 & 0xFFFFFFFF
    out["rtp_info"] = t.rtp_info
    out["ssrc"] = ["%08x" % x for x in sorted(t.ssrcs)]
    out["header_extension"] = {("0x%04x" % k): v for k, v in t.ext.items()} or None
    out["header_extension_sample"] = t.ext_sample
    out["nal_types"] = dict(sorted(t.nals.items())) or None
    out["sei_sample"] = t.sei_sample or None
    if len(fr) >= 5:
        ts = [f[0] for f in fr]
        back = [a - b for a, b in zip(ts, ts[1:]) if b < a]             # timestamps that go backwards = B-frame reordering
        uniq = sorted(set(ts))
        steps = [(b - a) / t.clock for a, b in zip(uniq, uniq[1:])]
        total = (uniq[-1] - uniq[0]) / t.clock
        out["repeated_timestamps"] = len(ts) - len(uniq)
        out["fps"] = round((len(uniq) - 1) / total, 3) if total > 0 else None       # delivered, gaps included
        if steps:
            out["nominal_fps"] = round(1 / statistics.median(steps), 3)                     # from the usual frame step
            out["frame_step_ms"] = {"min": round(min(steps) * 1000, 2), "median": round(statistics.median(steps) * 1000, 2),
                                    "max": round(max(steps) * 1000, 2)}
        step = statistics.median(steps) if steps else None
        out["b_frame_reordering"] = ({"backward_steps": len(back), "depth_frames": round(max(back) / t.clock / step) if step else None}
                                     if back else None)
        keys = [i for i, f in enumerate(fr) if f[4]]
        if len(keys) >= 2:
            gaps = [b - a for a, b in zip(keys, keys[1:])]
            out["key_every_frames"] = statistics.median(gaps)
            out["key_every_s"] = round(statistics.median([abs(fr[b][0] - fr[a][0]) / t.clock for a, b in zip(keys, keys[1:])]), 2)
            ksz = [fr[i][3] for i in keys]
            psz = [f[3] for f in fr if not f[4]]
            out["key_frame_kB"] = round(statistics.mean(ksz) / 1000, 1)
            out["other_frame_kB"] = round(statistics.mean(psz) / 1000, 1) if psz else None
        elif keys:
            out["key_every_frames"] = None
        out["key_frames"] = len(keys)
        # Arrival time of each frame against its RTP time: the slope is the clock ratio, the spread the jitter.
        # With B-frames the RTP time is the display time, which does not grow steadily in arrival order: keep only
        # the frames that move the timeline forward (I and P), whose delay against their display time is constant.
        sel, top = [], None
        for f in fr:
            if top is None or f[0] > top:
                sel.append(f)
                top = f[0]
        xs = [(f[0] - sel[0][0]) / t.clock for f in sel]
        ys = [(f[1] - s0) / 1e9 for f in sel]
        a, b, se, res = linear_fit(xs, ys)
        floor = min(y - b * x for x, y in zip(xs, ys))
        ares = sorted(abs(r) for r in res)
        # b = computer seconds per camera second; a positive ppm below = the camera's clock runs FAST against this computer
        out["clock_vs_computer_ppm"] = round((1 / b - 1) * 1e6, 1)
        out["clock_vs_computer_ppm_se"] = round(se / b ** 2 * 1e6, 1)
        out["arrival_jitter_ms"] = {"std": round(statistics.pstdev(res) * 1000, 1), "p95": round(percentile(ares, 0.95) * 1000, 1),
                                    "max": round(ares[-1] * 1000, 1)}
        ksel = [i for i, f in enumerate(sel) if f[4]]
        if ksel and len(ksel) < len(sel):
            kr = statistics.median([res[i] for i in ksel])
            pr = statistics.median([r for i, r in enumerate(res) if not sel[i][4]])
            out["key_frame_extra_delay_ms"] = round((kr - pr) * 1000, 1)
        out["first_frame_arrival_s"] = round(ys[0], 3)
        out["_fit"] = (a, b, floor, sel[0][0], t.clock)
    # RTCP
    sr = t.sr
    out["rtcp_sender_reports"] = len(sr)
    out["rtcp_cname"] = t.cname
    if sr:
        rate = (len(sr) - 1) / ((sr[-1][1] - sr[0][1]) / 1e9) if len(sr) > 1 else None
        out["rtcp_sr_per_s"] = round(rate, 3) if rate else None
        offs = sorted((u - w / 1e9) * 1000 for (w, _, u, _, _, _) in sr)       # NTP of the camera minus our clock, ms
        out["camera_ntp_minus_computer_ms"] = {"median": round(statistics.median(offs), 1), "spread": round(offs[-1] - offs[0], 1)}
        dn = sr[-1][2] - sr[0][2]
        if len(sr) >= 4 and dn >= 30:
            dr = ((sr[-1][3] - sr[0][3]) & 0xFFFFFFFF) / t.clock
            out["rtcp_rtp_vs_ntp_ppm"] = round((dr / dn - 1) * 1e6, 1)
        else:
            out["rtcp_rtp_vs_ntp_ppm"] = None
    return out


def fmt_url(url):
    return mask_url(url)


def report(streams, results):
    L = []
    P = L.append
    for s, res in zip(streams, results):
        P("=" * 100)
        P(f"[{s.label}] {fmt_url(s.url)}")
        P("=" * 100)
        if s.error:
            P(f"  FAILED: {s.error}")
            continue
        i = s.info
        P(f"  connect {i.get('connect_ms')} ms | server: {i.get('server') or '?'} | auth: {i.get('auth')}")
        P(f"  methods: {i.get('methods') or '?'}")
        P(f"  PLAY range: {i.get('play_range')} | RTP-Info: {i.get('rtp_info') or '-'}")
        sd = [a for a in s.sdp_session["attrs"]]
        refclk = [a for a in sd if a.startswith(("ts-refclk", "mediaclk"))]
        for t in s.tracks:
            refclk += [a for a in t.media["attrs"] if a.startswith(("ts-refclk", "mediaclk"))]
        P(f"  SDP clock declarations (RFC 7273 ts-refclk/mediaclk): {', '.join(refclk) if refclk else 'none'}")
        for t, r in zip(s.tracks, res):
            P(f"  -- {t.kind} track: {t.codec} {t.clock} Hz, channels {t.rtp_ch}/{t.rtcp_ch}, SDP fmtp: {(t.media.get('fmtp') or '-')[:110]}")
            if not r.get("packets"):
                P("     no RTP packets received")
                continue
            if t.kind == "video":
                P(f"     picture: {r.get('resolution', '?')} profile/level {r.get('profile_level', '?')} "
                  f"(from {r.get('sps_source', 'nowhere: no SPS seen')}), encoder fps in SPS: {r.get('sps_fps') or 'not stated'}, "
                  f"SDP framerate: {t.media.get('framerate', 'not stated')}")
            P(f"     measured: {r.get('fps', '?')} fps delivered (nominal {r.get('nominal_fps', '?')}), {r.get('kbit_per_s')} kbit/s, {r.get('frames')} frames in {r.get('seconds')} s, "
              f"frame step ms {r.get('frame_step_ms')}, repeated timestamps {r.get('repeated_timestamps')}")
            bf = r.get("b_frame_reordering")
            if bf:
                P(f"     B-FRAMES: timestamps go backwards {bf['backward_steps']} times (reordering depth {bf['depth_frames']} frames): the RTP time is the "
                  f"display time, the encoder holds frames back, so the capture-to-arrival delay is longer. The clock and jitter figures below are "
                  f"NOT reliable with B-frames (RTP carries no decode time): switch B-frames/smart-codec features off if the camera lets you")
            if t.kind == "video":
                P(f"     key frames: {r.get('key_frames')} (every {r.get('key_every_frames')} frames = {r.get('key_every_s')} s), "
                  f"mean size key {r.get('key_frame_kB')} kB vs other {r.get('other_frame_kB')} kB; NAL types {r.get('nal_types')}")
            P(f"     delivery vs this computer: camera clock {r.get('clock_vs_computer_ppm'):+} ppm (+-{r.get('clock_vs_computer_ppm_se')}; + = camera runs fast), "
              f"arrival jitter ms {r.get('arrival_jitter_ms')}, key frames arrive {r.get('key_frame_extra_delay_ms')} ms later")
            P(f"     packets {r['packets']}, sequence gaps {r['seq_gaps']}, SSRC {r.get('ssrc')}, first RTP timestamp {r.get('rtp_ts0')} "
              f"({r.get('rtp_ts0', 0) / t.clock:.2f} s{' - the timeline starts at PLAY' if r.get('rtp_ts0', 0) / t.clock < 5 else ' - a random origin'}), "
              f"RTP-Info (seq, rtptime) {r.get('rtp_info') if r.get('rtp_info') and any(x is not None for x in r['rtp_info']) else 'not given'}")
            P(f"     time information in the stream: RTP header extension {r.get('header_extension') or 'none'}"
              f"{' (first bytes ' + r['header_extension_sample'] + ')' if r.get('header_extension_sample') else ''}, "
              f"SEI {('sample ' + r['sei_sample'][0]) if r.get('sei_sample') else 'none seen'}")
            n = r["rtcp_sender_reports"]
            if n:
                P(f"     RTCP: {n} sender reports ({r.get('rtcp_sr_per_s')}/s), CNAME {r.get('rtcp_cname')}; camera NTP clock minus this computer: "
                  f"{r['camera_ntp_minus_computer_ms']['median']} ms (spread {r['camera_ntp_minus_computer_ms']['spread']} ms); "
                  f"RTP rate vs NTP {str(r.get('rtcp_rtp_vs_ntp_ppm')) + ' ppm' if r.get('rtcp_rtp_vs_ntp_ppm') is not None else 'not measurable (needs 4+ reports over 30+ s)'}")
            else:
                P(f"     RTCP: NO sender reports in {r.get('seconds')} s -> this stream carries no absolute (wall-clock) time")
    # ---- comparison -------------------------------------------------------------------------------------
    vids = []
    for s, res in zip(streams, results):
        if s.error:
            continue
        for t, r in zip(s.tracks, res):
            if t.kind == "video" and r.get("_fit"):
                vids.append((s, t, r))
                break
    if len(vids) >= 2:
        P("")
        P("=" * 100)
        P("COMPARISON of the video streams")
        P("=" * 100)
        if any(r.get("b_frame_reordering") for _, _, r in vids):
            P("  NOTE: a stream uses B-frames, so the clock-rate and jitter figures of that stream are unreliable (see above).")
        for k in range(len(vids)):
            for m in range(k + 1, len(vids)):
                (sa, ta, ra), (sb, tb, rb) = vids[k], vids[m]
                dppm = rb["clock_vs_computer_ppm"] - ra["clock_vs_computer_ppm"]
                se = math.hypot(ra["clock_vs_computer_ppm_se"], rb["clock_vs_computer_ppm_se"])
                P(f"  [{sa.label}] vs [{sb.label}]")
                P(f"    clock [{sb.label}] runs {dppm:+.1f} ppm (+-{se:.1f}) against clock [{sa.label}]; the two timelines slip apart by "
                  f"{abs(dppm) * 0.06:.2f} ms per minute ({abs(dppm) * 3.6:.0f} ms per hour)")
                if abs(dppm) > 3 * se and abs(dppm) > 5:
                    P("      -> the two clocks run at different rates: an offset measured once will drift; it has to be tracked, "
                      "or both streams need an absolute time")
                elif se > 15:
                    P(f"      -> too imprecise to tell (+-{se:.0f} ppm): to resolve ~10 ppm repeat with a longer --seconds (error falls as ~1/t^1.5)")
                else:
                    P(f"      -> no difference beyond the measurement error (+-{se:.1f} ppm): compatible with ONE clock (e.g. both lenses of one device)")
                r0a, r0b = ra["rtp_ts0"] / ta.clock, rb["rtp_ts0"] / tb.clock
                P(f"    first RTP timestamps: {ra['rtp_ts0']} vs {rb['rtp_ts0']} ({r0a:.2f} s vs {r0b:.2f} s)"
                  f"{'  -> both timelines start at PLAY: independent origins, timestamps cannot be compared across streams' if r0a < 5 and r0b < 5 else ''}")
                srs = (ra["rtcp_sender_reports"], rb["rtcp_sender_reports"])
                if all(srs):
                    d = rb["camera_ntp_minus_computer_ms"]["median"] - ra["camera_ntp_minus_computer_ms"]["median"]
                    P(f"    both send RTCP sender reports: their NTP clocks differ by {d:+.1f} ms -> absolute timestamps are usable")
                else:
                    P("    at least one stream sends no RTCP sender report -> there is no absolute time to align them with; "
                      "use arrival time plus a one-time calibration (see the notes)")
                P(f"    arrival jitter (std, ms): {ra['arrival_jitter_ms']['std']} and {rb['arrival_jitter_ms']['std']} "
                  f"-> a floor on how well arrival times can pair the frames, on top of the frame period "
                  f"({ra.get('frame_step_ms', {}).get('median')} / {rb.get('frame_step_ms', {}).get('median')} ms)")
    return "\n".join(L)


# =================================================================================================
# ports and ISAPI
# =================================================================================================
def check_ports(host, ports, timeout=2.0):
    out = {}

    def one(p):
        try:
            with socket.create_connection((host, p), timeout):
                out[p] = True
        except OSError:
            out[p] = False
    th = [threading.Thread(target=one, args=(p,)) for p in ports]
    [x.start() for x in th]
    [x.join() for x in th]
    return dict(sorted(out.items()))


def http_get(host, port, path, user, password, timeout=8.0):
    """GET with Basic/Digest authentication.  Returns (status, body text)."""
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request("GET", path)
        r = conn.getresponse()
        body = r.read()
        if r.status == 401 and user is not None:
            ch = r.getheader("WWW-Authenticate", "")
            if ch.lower().startswith("digest"):
                p = parse_kv(ch[6:])
                ha1, ha2 = md5(f"{user}:{p['realm']}:{password}"), md5(f"GET:{path}")
                if "auth" in [q.strip() for q in p.get("qop", "").split(",")]:
                    nc, cn = "00000001", "%08x" % random.getrandbits(32)
                    resp = md5(f"{ha1}:{p['nonce']}:{nc}:{cn}:auth:{ha2}")
                    auth = (f'Digest username="{user}", realm="{p["realm"]}", nonce="{p["nonce"]}", uri="{path}", response="{resp}", '
                            f'qop=auth, nc={nc}, cnonce="{cn}"')
                else:
                    resp = md5(ha1 + ":" + p["nonce"] + ":" + ha2)
                    auth = f'Digest username="{user}", realm="{p["realm"]}", nonce="{p["nonce"]}", uri="{path}", response="{resp}"'
            else:
                auth = "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()
            conn.close()
            conn = http.client.HTTPConnection(host, port, timeout=timeout)
            conn.request("GET", path, headers={"Authorization": auth})
            r = conn.getresponse()
            body = r.read()
        return r.status, body.decode("utf-8", "replace")
    finally:
        conn.close()


# =================================================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description="Probe RTSP streams: parameters, timing, and what time information they carry.")
    ap.add_argument("urls", nargs="+", help="one or more rtsp:// URLs (all are played at the same time)")
    ap.add_argument("--seconds", type=float, default=20, help="how long to play [%(default)s]; use 300+ to measure clock drift")
    ap.add_argument("--user", default=os.environ.get("RTSP_USER"), help="user, if not in the URL (or $RTSP_USER)")
    ap.add_argument("--password", default=os.environ.get("RTSP_PASSWORD"), help="password, if not in the URL (or $RTSP_PASSWORD)")
    ap.add_argument("--ports", action="store_true", help="also check which common camera ports accept a TCP connection")
    ap.add_argument("--isapi", action="store_true", help="also try a few read-only Hikvision-style ISAPI GETs on --http-port")
    ap.add_argument("--http-port", type=int, default=80)
    ap.add_argument("--json", metavar="FILE", help="write the numbers to FILE")
    ap.add_argument("--sdp", action="store_true", help="print the raw SDP of every stream")
    args = ap.parse_args(argv)

    streams = [Stream(f"{i + 1}", u, args.user, args.password, args.seconds) for i, u in enumerate(args.urls)]
    barrier = threading.Barrier(len(streams))
    threads = [threading.Thread(target=s.run, args=(barrier,), daemon=True) for s in streams]
    print(f"probing {len(streams)} stream(s) for {args.seconds:.0f} s ...", flush=True)
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    results = [[analyze_track(s, t) for t in s.tracks] for s in streams]
    print(report(streams, results))
    if args.sdp:
        for s in streams:
            print(f"\n--- SDP of [{s.label}] ---\n{s.sdp_text}")
    host = urlsplit(args.urls[0]).hostname
    extra = {}
    if args.ports:
        ports = check_ports(host, [80, 443, 554, 8000, 8080, 8443, 8554, 9010, 9020])
        extra["ports"] = ports
        print("\nTCP ports on", host, ":", ", ".join(f"{p} {'open' if o else 'closed'}" for p, o in ports.items()))
    if args.isapi:
        user = urlsplit(args.urls[0]).username or args.user
        pw = urlsplit(args.urls[0]).password or args.password
        extra["isapi"] = {}
        for path in ("/ISAPI/System/deviceInfo", "/ISAPI/System/time", "/ISAPI/System/time/ntpServers", "/ISAPI/Streaming/channels"):
            try:
                st, body = http_get(host, args.http_port, path, unquote(user) if user else None, unquote(pw) if pw else None)
            except Exception as e:
                st, body = None, f"{type(e).__name__}: {e}"
            extra["isapi"][path] = {"status": st, "body": body[:1500]}
            print(f"\nGET {path} -> {st}\n{body[:600]}")
    if args.json:
        clean = [[{k: v for k, v in r.items() if not k.startswith("_")} for r in res] for res in results]
        json.dump({"urls": [mask_url(u) for u in args.urls], "seconds": args.seconds,
                   "streams": [{"label": s.label, "error": s.error, "info": s.info, "tracks": c} for s, c in zip(streams, clean)],
                   **extra}, open(args.json, "w"), indent=1, default=str)
        print(f"\nnumbers written to {args.json}")
    return 1 if any(s.error for s in streams) else 0


if __name__ == "__main__":
    sys.exit(main())

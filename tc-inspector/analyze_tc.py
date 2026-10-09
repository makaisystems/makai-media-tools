#!/usr/bin/env python3
# Copyright (c) 2026 Makai Systems LLC
# SPDX-License-Identifier: BUSL-1.1
# Free to run as a standalone tool; selling it or incorporating it into a
# commercial product requires a license. See LICENSE for the full terms.
# For commercial licensing, contact licensing@makaisystems.com.
"""Analyze audio/video files for embedded timecode and recommend fixes.

Checks, per file:
  * Format, duration, RIFF/RF64 chunk layout (WAV)
  * BWF bext TimeReference (what Premiere, Pro Tools, Resolve use as a WAV's start TC)
  * iXML timecode rate / flag / timestamp and track names
  * Container timecode in video/QuickTime/MXF files (via ffprobe)
  * LTC (SMPTE linear timecode) on any audio channel: decoded TC, measured frame rate,
    drop-frame flag, continuity, coverage, level, reverse playback
  * Hints in the filename (LTC_HHMMSSFF, 30_NDF, 48000x16, 5mins)
and then cross-checks them, explaining what an NLE will do with the file and how to fix it.

WAV/BWF/RF64 (8/16/24/32-bit int, 32/64-bit float) are read natively. Anything else
(AIFF, A-law/u-law WAV, MOV, MP4, MXF, ...) is decoded with ffmpeg/ffprobe if installed.

Frame rates: 23.976, 24, 25, 29.97, 29.97df, 30, 47.952, 48, 50, 59.94, 59.94df, 60.
LTC itself tops out at 30 frames per second (SMPTE 12M), so 48-60 fps projects use LTC at
half rate, one LTC frame per pair of video frames; the script matches them up that way.

Usage:
  python3 analyze_tc.py file.wav [more files ...] [--fps 59.94df] [--limit 600] [--json]
  python3 analyze_tc.py --fix file.wav          # writes file.tcfix.wav, then re-checks it
  python3 analyze_tc.py --fix --in-place file.wav
Each report ends with what --fix would do. Fixes use the LTC as the reference when present.
Exit code: 0 = no errors (after fixing, with --fix), 1 = ERROR findings remain, 2 = unreadable file / fix failed.
"""
import argparse
import array
import json
import math
import mmap
import os
import re
import shutil
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET
from fractions import Fraction

# label -> (actual rate, nominal frame count per TC second, drop frame)
RATES = {
    "23.976": (Fraction(24000, 1001), 24, False),
    "24": (Fraction(24), 24, False),
    "25": (Fraction(25), 25, False),
    "29.97": (Fraction(30000, 1001), 30, False),
    "29.97df": (Fraction(30000, 1001), 30, True),
    "30": (Fraction(30), 30, False),
    "47.952": (Fraction(48000, 1001), 48, False),
    "48": (Fraction(48), 48, False),
    "50": (Fraction(50), 50, False),
    "59.94": (Fraction(60000, 1001), 60, False),
    "59.94df": (Fraction(60000, 1001), 60, True),
    "60": (Fraction(60), 60, False),
}
SYNC = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]
SYNC_REV = SYNC[::-1]
PROBE_SECONDS = 10


# ---------------------------------------------------------------- timecode math

def drop_count(nominal):
    return nominal // 15  # 2 frames/minute at 30, 4 at 60


def tc_to_frames(h, m, s, f, nominal, drop):
    frames = (h * 3600 + m * 60 + s) * nominal + f
    if drop:
        total_min = h * 60 + m
        frames -= drop_count(nominal) * (total_min - total_min // 10)
    return frames


def frames_to_tc(frames, nominal, drop):
    frames = int(frames)
    if drop:
        dc = drop_count(nominal)
        per_min, per_10 = nominal * 60 - dc, nominal * 600 - 9 * dc
        d, mod = divmod(frames, per_10)
        frames += 9 * dc * d + (dc * ((mod - dc) // per_min) if mod > dc else 0)
    return ((frames // (nominal * 3600)) % 24, (frames // (nominal * 60)) % 60,
            (frames // nominal) % 60, frames % nominal)


def fmt_tc(tc, drop=False):
    return "%02d:%02d:%02d%s%02d" % (tc[0], tc[1], tc[2], ";" if drop else ":", tc[3])


def parse_tc(s):
    m = re.match(r"^(\d{1,2})[:;.](\d\d)[:;.](\d\d)([:;.])(\d\d)$", s.strip())
    if not m:
        return None, False
    return tuple(int(m.group(i)) for i in (1, 2, 3, 5)), m.group(4) in ";."


def label_seconds(tc, nominal):
    """A timecode label as a number of 'timecode seconds' (labels, not wall-clock time)."""
    return Fraction(tc[0] * 3600 + tc[1] * 60 + tc[2]) + Fraction(tc[3], nominal)


def rate_key_for(rate, drop):
    for k, (r, _, d) in RATES.items():
        if r == rate and d == drop:
            return k
    return None


def samples_to_tc(samples, sr, rate_key):
    rate, nominal, drop = RATES[rate_key]
    frames = Fraction(samples) * rate / sr
    return fmt_tc(frames_to_tc(math.floor(frames), nominal, drop), drop)


def tc_samples(frame_count, rate, sr):
    return round(Fraction(frame_count) / rate * sr)


def is_pair_rate(high_key, ltc_key):
    """True if high_key is a 48-60 fps rate carried by LTC at ltc_key (frame pairs)."""
    hr, _, hd = RATES[high_key]
    lr, _, ld = RATES[ltc_key]
    return hr == 2 * lr and hd == ld


# ---------------------------------------------------------------- sources

class Source:
    """A file's metadata plus lazily-extracted audio channels."""

    def __init__(self, path):
        self.path = path
        self.kind = "wav"
        self.container = None
        self.chunks = []
        self.bext = None
        self.ixml = None
        self.container_tc = None   # {"tc": str, "rate_key": str|None, "start": float}
        self.streams = []          # [{"label", "sr", "bits", "float", "channels", "frames", "get", "start"}]


def load(path, limit_s=None):
    with open(path, "rb") as fh:
        head = fh.read(12)
    if head[:4] in (b"RIFF", b"RF64", b"BW64") and head[8:12] == b"WAVE":
        src = load_wav(path, limit_s)
        if src.streams:
            return src
        return load_ffmpeg(path, limit_s, src)  # WAV with a codec we don't decode natively
    return load_ffmpeg(path, limit_s, None)


def load_wav(path, limit_s):
    src = Source(path)
    fh = open(path, "rb")
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    src.container = mm[:4].decode()
    ds64_data, fmt, data_at, data_size = None, None, None, None
    p = 12
    while p + 8 <= len(mm):
        cid, size = mm[p:p + 4], struct.unpack("<I", mm[p + 4:p + 8])[0]
        at = p + 8
        if cid == b"ds64":
            ds64_data = struct.unpack("<Q", mm[at + 8:at + 16])[0]
        if cid == b"data" and size == 0xFFFFFFFF and ds64_data is not None:
            size = ds64_data
        size = min(size, len(mm) - at)
        src.chunks.append((cid.decode("latin1"), size))
        if cid == b"fmt ":
            body = mm[at:at + size]
            tag, ch, sr, _, blk, bits = struct.unpack("<HHIIHH", body[:16])
            if tag == 0xFFFE and len(body) >= 26:
                tag = struct.unpack("<H", body[24:26])[0]
            fmt = {"tag": tag, "channels": ch, "sr": sr, "block": blk, "bits": bits}
        elif cid == b"data":
            data_at, data_size = at, size
        elif cid == b"bext" and size >= 346:
            src.bext = parse_bext(mm[at:at + size])
        elif cid == b"iXML":
            src.ixml = parse_ixml(mm[at:at + size])
        p = at + size + (size & 1)
    if fmt is None or data_at is None:
        raise ValueError("missing fmt or data chunk")

    tag, bits, nch, sr, blk = fmt["tag"], fmt["bits"], fmt["channels"], fmt["sr"], fmt["block"]
    native = (tag == 1 and bits in (8, 16, 24, 32)) or (tag == 3 and bits in (32, 64))
    src.format = {"channels": nch, "sr": sr, "bits": bits, "float": tag == 3, "codec_tag": tag}
    if not native:
        return src
    width = bits // 8
    n = data_size // blk
    if limit_s:
        n = min(n, int(limit_s * sr))

    def get(k):
        raw = bytearray(n * width)
        base = data_at + k * width
        for j in range(width):
            raw[j::width] = mm[base + j:base + j + n * blk:blk]
        if tag == 3:
            a, fs = array.array("f" if bits == 32 else "d"), 1.0
            a.frombytes(raw)
        elif bits == 8:  # unsigned 8-bit
            a, fs = array.array("b"), 127
            a.frombytes(bytes(raw).translate(bytes((i - 128) & 0xFF for i in range(256))))
        elif bits == 16:
            a, fs = array.array("h"), 32767
            a.frombytes(raw)
        elif bits == 24:
            wide = bytearray(4 * n)  # widen to 32-bit; values scaled by 256
            wide[1::4], wide[2::4], wide[3::4] = raw[0::3], raw[1::3], raw[2::3]
            a, fs = array.array("i"), 0x7FFFFF00
            a.frombytes(wide)
        else:
            a, fs = array.array("i"), 0x7FFFFFFF
            a.frombytes(raw)
        if sys.byteorder != "little" and a.itemsize > 1:
            a.byteswap()
        return a, fs

    src.streams.append({"label": "", "sr": sr, "channels": nch, "frames": n, "get": get, "start": 0.0})
    return src


def parse_bext(body):
    txt = lambda a, b: body[a:b].split(b"\0")[0].decode("latin1").strip()
    return {
        "description": txt(0, 256), "originator": txt(256, 288),
        "originator_reference": txt(288, 320), "date": txt(320, 330), "time": txt(330, 338),
        "time_reference": struct.unpack("<Q", body[338:346])[0],
        "version": struct.unpack("<H", body[346:348])[0] if len(body) >= 348 else None,
        "coding_history": body[602:].split(b"\0")[0].decode("latin1").strip() if len(body) > 602 else "",
    }


def parse_ixml(body):
    out = {"raw_ok": False}
    try:
        root = ET.fromstring(body.split(b"\0")[0].decode("utf-8", "replace"))
    except ET.ParseError:
        return out
    out["raw_ok"] = True
    g = lambda path: (root.findtext(path) or "").strip() or None
    out["timecode_rate"] = g("SPEED/TIMECODE_RATE")
    out["timecode_flag"] = g("SPEED/TIMECODE_FLAG")
    hi, lo = g("SPEED/TIMESTAMP_SAMPLES_SINCE_MIDNIGHT_HI"), g("SPEED/TIMESTAMP_SAMPLES_SINCE_MIDNIGHT_LO")
    if hi is not None and lo is not None:
        out["timestamp"] = (int(hi) << 32) | int(lo)
    out["tracks"] = [(t.findtext("CHANNEL_INDEX"), t.findtext("NAME")) for t in root.findall("TRACK_LIST/TRACK")]
    return out


def load_ffmpeg(path, limit_s, src):
    if not (shutil.which("ffprobe") and shutil.which("ffmpeg")):
        raise ValueError("not a natively supported WAV and ffmpeg/ffprobe are not installed")
    src = src or Source(path)
    src.kind = "ffmpeg" if src.container is None else "wav"
    pr = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path],
                        capture_output=True, text=True)
    if pr.returncode:
        raise ValueError("ffprobe could not read the file: " + pr.stderr.strip()[:200])
    info = json.loads(pr.stdout)
    streams, fmtinfo = info.get("streams", []), info.get("format", {})
    if src.container is None:
        src.container = fmtinfo.get("format_name", "?")

    # Container timecode (QuickTime tmcd, MXF, MP4, ...): format tag or any stream's tag.
    tc_str = (fmtinfo.get("tags") or {}).get("timecode")
    for s in streams:
        tc_str = tc_str or (s.get("tags") or {}).get("timecode")
    video = next((s for s in streams if s.get("codec_type") == "video"
                  and not (s.get("disposition") or {}).get("attached_pic")), None)
    tmcd = next((s for s in streams if s.get("codec_tag_string") == "tmcd"), None)
    if tc_str:
        tc, drop = parse_tc(tc_str)
        rate_key = None
        for s in (video, tmcd):
            if s and s.get("r_frame_rate") not in (None, "0/0"):
                rate_key = rate_key_for(Fraction(s["r_frame_rate"]), drop) or rate_key_for(
                    Fraction(s.get("avg_frame_rate", "0/1") or "0/1"), drop)
                if rate_key:
                    break
        src.container_tc = {"tc": tc_str, "parsed": tc, "drop": drop, "rate_key": rate_key,
                            "start": float((video or {}).get("start_time") or 0)}
    if video:
        src.video = {"codec": video.get("codec_name"), "rate": video.get("r_frame_rate"),
                     "size": f"{video.get('width')}x{video.get('height')}",
                     "start": float(video.get("start_time") or 0)}

    audio = [s for s in streams if s.get("codec_type") == "audio"]
    if not audio:
        raise ValueError("no audio streams")
    first = audio[0]
    if src.container in ("RIFF", "RF64", "BW64") and not hasattr(src, "format"):
        src.format = {}
    if not hasattr(src, "format"):
        src.format = {"channels": sum(int(s.get("channels", 0)) for s in audio), "sr": int(first["sample_rate"]),
                      "bits": int(first.get("bits_per_raw_sample") or first.get("bits_per_sample") or 0),
                      "float": "flt" in (first.get("sample_fmt") or ""), "codec": first.get("codec_name")}
    else:
        src.format["codec"] = first.get("codec_name")
    for i, s in enumerate(audio):
        nch, sr = int(s.get("channels", 1)), int(s["sample_rate"])
        cmd = ["ffmpeg", "-v", "error", "-i", path, "-map", f"0:a:{i}"]
        if limit_s:
            cmd += ["-t", str(limit_s)]
        cmd += ["-f", "s32le", "-acodec", "pcm_s32le", "-"]
        raw = subprocess.run(cmd, capture_output=True).stdout
        a = array.array("i")
        a.frombytes(raw[:len(raw) // 4 * 4])
        if sys.byteorder != "little":
            a.byteswap()
        get = (lambda a, nch: lambda k: (a[k::nch], 0x7FFFFFFF))(a, nch)
        label = f"a{i}." if len(audio) > 1 or src.kind == "ffmpeg" else ""
        src.streams.append({"label": label, "sr": sr, "channels": nch, "frames": len(a) // max(1, nch),
                            "get": get, "start": float(s.get("start_time") or 0)})
    return src


# ---------------------------------------------------------------- LTC decoding

def decode_ltc(x, sr, peak):
    """Return list of (frame_start_sample, (h,m,s,f), drop_flag) plus a reverse-sync count."""
    h = 0.25 * peak
    edges, state = [], 0
    for i, v in enumerate(x):
        if v > h:
            if state != 1:
                edges.append(i)
                state = 1
        elif v < -h:
            if state != -1:
                edges.append(i)
                state = -1
    if len(edges) < 400:
        return [], 0
    # The last bit has no closing transition; close it where the signal ends.
    end = next(i for i in range(len(x) - 1, -1, -1) if abs(x[i]) > h) + 1
    if end > edges[-1]:
        edges.append(end)
    iv = [b - a for a, b in zip(edges, edges[1:])]
    lo, hi = 0.75 * sr / 2400, 1.25 * sr / 1918  # full-bit length range, 23.976-30 frames/s (+ varispeed)
    longs = sorted(v for v in iv if lo <= v <= hi)
    if not longs:
        return [], 0
    L = longs[len(longs) // 2]
    bits, pos = [], []
    i = 0
    while i < len(iv):
        if 0.75 * L <= iv[i] <= 1.4 * L:
            bits.append(0); pos.append(edges[i]); i += 1
        elif 0.3 * L <= iv[i] < 0.75 * L and i + 1 < len(iv) and 0.3 * L <= iv[i + 1] < 0.75 * L:
            bits.append(1); pos.append(edges[i]); i += 2
        else:
            bits.append(None); pos.append(edges[i]); i += 1  # undecodable gap
    v = lambda b, st, w: sum(b[st + k] << k for k in range(w))
    frames, rev = [], 0
    for j in range(len(bits) - 79):
        if bits[j + 64:j + 80] == SYNC:
            b = bits[j:j + 80]
            if None in b:
                continue
            tc = (v(b, 48, 4) + 10 * v(b, 56, 2), v(b, 32, 4) + 10 * v(b, 40, 3),
                  v(b, 16, 4) + 10 * v(b, 24, 3), v(b, 0, 4) + 10 * v(b, 8, 2))
            if tc[0] < 24 and tc[1] < 60 and tc[2] < 60 and tc[3] < 30:
                frames.append((pos[j], tc, b[10]))
        elif bits[j:j + 16] == SYNC_REV:
            rev += 1
    return frames, rev


def find_ltc(x, sr, peak):
    """Probe a few short windows first so long non-LTC channels are skipped quickly."""
    n, w = len(x), PROBE_SECONDS * sr
    if n > 4 * w:
        rev_total = 0
        for a in (0, n // 2 - w // 2, n - w):
            frames, rev = decode_ltc(x[a:a + w], sr, peak)
            rev_total += rev
            if len(frames) >= 5:
                break
        else:
            return [], rev_total
    return decode_ltc(x, sr, peak)


def analyze_ltc(frames, sr):
    nominal = max(f[1][3] for f in frames) + 1
    nominal = min((24, 25, 30), key=lambda r: abs(r - nominal))
    drop = sum(f[2] for f in frames) * 2 > len(frames)
    cnt = [(s, tc_to_frames(*tc, nominal, drop)) for s, tc, _ in frames]
    # Use the longest continuous run to measure the rate.
    runs, cur = [], [cnt[0]]
    for a, b in zip(cnt, cnt[1:]):
        if b[1] - a[1] >= 1 and abs((b[0] - a[0]) / (b[1] - a[1]) - sr / nominal) < 0.1 * sr / nominal:
            cur.append(b)
        else:
            runs.append(cur); cur = [b]
    runs.append(cur)
    best = max(runs, key=len)
    if len(best) > 1:
        measured = (best[-1][1] - best[0][1]) * sr / (best[-1][0] - best[0][0])
    else:
        measured = nominal
    cands = [(k, r) for k, (r, nom, d) in RATES.items() if nom == nominal and d == drop]
    key, rate = min(cands, key=lambda kr: abs(float(kr[1]) - measured))
    ppm = (measured / float(rate) - 1) * 1e6
    spf = sr / measured
    jumps, dropouts = [], 0
    for a, b in zip(cnt, cnt[1:]):
        expect = round((b[0] - a[0]) / spf)
        if b[1] - a[1] != expect:
            jumps.append((a[0] / sr, fmt_tc(frames_to_tc(a[1], nominal, drop), drop),
                          fmt_tc(frames_to_tc(b[1], nominal, drop), drop)))
        elif expect > 1:
            dropouts += expect - 1
    s0, f0 = best[0]
    at_zero = f0 - Fraction(s0) / Fraction(spf).limit_denominator(10 ** 6)
    whole = math.floor(at_zero + Fraction(1, 1000))
    zero_tc = frames_to_tc(whole, nominal, drop)
    first, last = cnt[0], cnt[-1]
    return {
        "frames_decoded": len(frames), "nominal": nominal, "drop": drop,
        "rate_key": key, "rate": rate, "measured_fps": measured, "ppm": ppm,
        "first_tc": fmt_tc(frames_to_tc(first[1], nominal, drop), drop), "first_at_s": first[0] / sr,
        "last_tc": fmt_tc(frames_to_tc(last[1], nominal, drop), drop), "last_end_s": (last[0] + spf) / sr,
        "frames_at_zero": at_zero, "tc_at_zero": fmt_tc(zero_tc, drop),
        "label_s_at_zero": label_seconds(zero_tc, nominal) + (at_zero - whole) / nominal,
        "subframe_samples": round(float(at_zero - whole) * spf),
        "jumps": jumps, "dropouts": dropouts,
        "expected_bext": tc_samples(at_zero, rate, sr),
    }


# ---------------------------------------------------------------- filename hints

FPS_RE = r"23\.976|23\.98|29\.97|47\.952|47\.95|59\.94|24|25|30|48|50|60"


def filename_hints(name):
    stem = os.path.splitext(os.path.basename(name))[0]
    h = {}
    m = re.search(r"(?<!\d)([0-2]\d)([0-5]\d)([0-5]\d)([0-5]\d)(?!\d)", stem)
    if m and int(m.group(1)) < 24:
        h["tc"] = tuple(int(x) for x in m.groups())
    m = re.search(rf"(?i)(?<![\d.])({FPS_RE})(?![\d.])(?:[ _-]*(NDF|DF))?[ _-]*(?:FPS|P)(?![A-Z])", stem) or \
        re.search(rf"(?i)(?<![\d.])({FPS_RE})(?![\d.])[ _-]*(NDF|DF)(?![A-Z])", stem)
    if m:
        h["fps"] = {"23.98": "23.976", "47.95": "47.952"}.get(m.group(1), m.group(1))
        if m.group(2):
            h["drop"] = m.group(2).upper() == "DF"
    m = re.search(r"(44100|48000|88200|96000|192000)\s*x\s*(8|16|24|32|64)", stem)
    if m:
        h["sample_rate"], h["bits"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"(?i)(\d+(?:\.\d+)?)\s*min", stem)
    if m:
        h["minutes"] = float(m.group(1))
    return h


# ---------------------------------------------------------------- analysis

def analyze(path, fps_override=None, limit_s=None):
    src = load(path, limit_s)
    rep = {"file": path, "kind": src.kind, "container": src.container, "chunks": src.chunks,
           "format": dict(src.format), "bext": src.bext, "ixml": src.ixml,
           "container_tc": src.container_tc, "video": getattr(src, "video", None),
           "hints": filename_hints(path), "channels": [], "findings": [], "limited_to_s": limit_s}
    st0 = src.streams[0]
    rep["format"]["duration_s"] = st0["frames"] / st0["sr"]
    rep["format"]["sr"] = st0["sr"]

    for st in src.streams:
        sr = st["sr"]
        for k in range(st["channels"]):
            x, fs = st["get"](k)
            n = len(x)
            mx, mn = (max(x), min(x)) if n else (0, 0)
            peak = max(mx, -mn)
            ch = {"index": f"{st['label']}{k + 1}", "sr": sr, "stream_start": st["start"],
                  "peak_dbfs": 20 * math.log10(peak / fs) if peak else None}
            if peak:
                thr = peak / 1000
                first = next(i for i in range(n) if abs(x[i]) > thr)
                last = next(i for i in range(n - 1, -1, -1) if abs(x[i]) > thr)
                ch["signal_from_s"], ch["signal_to_s"] = first / sr, last / sr
                clip_lvl = fs * 0.9999
                ch["clipped_samples"] = (x.count(mx) if mx >= clip_lvl else 0) + (x.count(mn) if -mn >= clip_lvl else 0)
                frames, rev = find_ltc(x, sr, peak)
                ch["reverse_sync_words"] = rev
                if len(frames) >= 10:
                    ch["ltc"] = analyze_ltc(frames, sr)
            rep["channels"].append(ch)
            del x

    findings(rep, fps_override)
    return rep


def declared_rate(rep, fps_override):
    """Frame rate the file claims (or the user gave), before looking at LTC."""
    if fps_override:
        return fps_override, "--fps"
    ctc = rep["container_tc"]
    if ctc and ctc.get("rate_key"):
        return ctc["rate_key"], "container"
    ix = rep["ixml"] or {}
    if ix.get("timecode_rate"):
        try:
            k = rate_key_for(Fraction(ix["timecode_rate"]), (ix.get("timecode_flag") or "").upper() == "DF")
            if k:
                return k, "iXML"
        except (ValueError, ZeroDivisionError):
            pass
    hints = rep["hints"]
    if "fps" in hints:
        k = hints["fps"] + ("df" if hints.get("drop") and hints["fps"] in ("29.97", "59.94") else "")
        if k in RATES:
            return k, "filename"
    return None, None


def findings(rep, fps_override):
    F = rep["findings"]
    add = lambda sev, msg, fix=None: F.append({"severity": sev, "message": msg, "fix": fix})
    fmt = rep["format"]
    sr, dur = fmt["sr"], fmt["duration_s"]
    bext, ixml, hints, ctc = rep["bext"], rep["ixml"] or {}, rep["hints"], rep["container_tc"]
    is_wav = rep["container"] in ("RIFF", "RF64", "BW64")
    ltcs = [c for c in rep["channels"] if "ltc" in c]
    ltc = ltcs[0]["ltc"] if ltcs else None
    fname = os.path.basename(rep["file"])

    decl, decl_src = declared_rate(rep, fps_override)
    # Rate used to show TC values: the declared one if it agrees with the LTC (incl. frame pairs), else the LTC.
    if ltc and not fps_override and decl and decl != ltc["rate_key"] and not is_pair_rate(decl, ltc["rate_key"]):
        show = ltc["rate_key"]
    else:
        show = decl or (ltc and ltc["rate_key"])
    rep["interpret_rate"] = show
    pairs = bool(ltc and show and is_pair_rate(show, ltc["rate_key"]))

    if rep["limited_to_s"]:
        add("INFO", f"Only the first {rep['limited_to_s']:g}s were analyzed (--limit).")
    if pairs:
        add("INFO", f"{show} fps project with {ltc['rate_key']} fps LTC: each LTC frame covers a pair of "
                    f"{show} frames (standard for LTC above 30 fps).")

    if bext:
        tr = bext["time_reference"]
        rep["bext_tc"] = samples_to_tc(tr, sr, show) if show else None
        secs = tr / sr
        rep["bext_clock"] = "%02d:%02d:%06.3f" % (secs // 3600, secs % 3600 // 60, secs % 60)
    if ctc and ctc["parsed"]:
        nom = RATES[ctc["rate_key"]][1] if ctc.get("rate_key") else None
        rep["container_tc_label_s"] = label_seconds(ctc["parsed"], nom) if nom else None

    # --- what the NLE will do
    if bext:
        where = rep["bext_tc"] or rep["bext_clock"]
        add("INFO", f"Premiere/Pro Tools/Resolve will place this clip at {where} (from bext TimeReference "
                    f"{bext['time_reference']}{'' if show else ', frame rate unknown'}).")
    elif ctc:
        add("INFO", f"Premiere/Resolve will use the embedded container timecode {ctc['tc']}"
                    f"{' @ ' + ctc['rate_key'] if ctc.get('rate_key') else ''}.")
    elif is_wav:
        add("ERROR" if (ltc or "tc" in hints) else "WARN",
            "No bext chunk: NLEs will treat this clip as starting at 00:00:00:00 and cannot sync it by timecode.",
            "Add one: bwfmetaedit --Timereference=<samples> \"%s\"   (or rerun with --fix)" % fname)
    elif rep.get("video"):
        add("ERROR" if ltc else "WARN",
            "No embedded timecode in this file: NLEs will start it at 00:00:00:00 (or its creation time).",
            f"ffmpeg -i \"{fname}\" -map 0 -c copy -timecode \"{ltc['tc_at_zero']}\" out.mov   (stamps the LTC start; "
            "or rerun with --fix)" if ltc else None)
    else:
        bits = fmt.get("bits") if fmt.get("bits") in (16, 24, 32) else 24
        add("ERROR" if ltc else "WARN",
            f"{rep['container']} has no standard place for a timecode stamp: NLEs will start it at 00:00:00:00. "
            "Convert to Broadcast WAV to carry one.",
            f"ffmpeg -i \"{fname}\" -c:a pcm_s{bits}le -write_bext 1 out.wav && "
            f"bwfmetaedit --Timereference={ltc['expected_bext']} out.wav   (sets start to {ltc['tc_at_zero']}; "
            "or rerun with --fix)" if ltc else None)

    # --- LTC vs bext
    if ltc and is_wav:
        exp = ltc["expected_bext"]
        target = samples_to_tc(exp, sr, show) if show else ltc["tc_at_zero"]
        if bext:
            diff = bext["time_reference"] - exp
            if abs(Fraction(diff) * ltc["rate"] / sr) >= Fraction(1, 2):
                add("ERROR",
                    f"bext start ({rep['bext_tc']}) does not match the LTC on channel {ltcs[0]['index']} "
                    f"({ltc['tc_at_zero']} at sample 0): off by {describe_offset(diff / sr, ltc)}. "
                    "Syncing by timecode in the NLE will land this clip in the wrong place.",
                    f"bwfmetaedit --Timereference={exp} \"{fname}\"   (sets start to {target}; or rerun with --fix)")
            elif diff:
                add("INFO", f"bext and LTC agree to within {abs(diff)} samples (< half an LTC frame).")
            else:
                add("OK", f"bext start matches the LTC exactly ({target}).")
        elif F[-1]["severity"] == "ERROR":
            F[-1]["fix"] = f"bwfmetaedit --Timereference={exp} \"{fname}\"   (sets start to {target}; or rerun with --fix)"

    # --- LTC vs container timecode (camera files fed LTC on an audio input)
    if ltc and ctc and rep.get("container_tc_label_s") is not None:
        ch = ltcs[0]
        # LTC label at the first video frame = label at audio sample 0 + time between them.
        shift = Fraction(ctc["start"] - ch["stream_start"]).limit_denominator(10 ** 6)
        ltc_at_video = ltc["label_s_at_zero"] + shift * ltc["rate"] / ltc["nominal"]
        d = rep["container_tc_label_s"] - ltc_at_video
        frames = d * ltc["nominal"]
        if abs(frames) >= Fraction(1, 2):
            add("ERROR" if abs(frames) >= 2 else "WARN",
                f"Container timecode {ctc['tc']} does not match the LTC on channel {ch['index']} "
                f"(which reads {fmt_tc(frames_to_tc(math.floor(ltc_at_video * ltc['nominal'] + Fraction(1, 1000)), ltc['nominal'], False), ltc['drop'])} "
                f"at the first video frame): off by {float(frames):+.1f} LTC frames. The camera's clock and the "
                "fed LTC disagree; decide which one is the reference and re-stamp the other.",
                "ffmpeg -i \"%s\" -map 0 -c copy -timecode <correct TC> out.%s   (or rerun with --fix --trust-ltc "
                "to re-stamp from the LTC)" % (fname, fname.rsplit('.', 1)[-1]))
        else:
            add("OK", f"Container timecode {ctc['tc']} matches the LTC on channel {ch['index']}.")
        if ctc.get("rate_key") and ctc["rate_key"] != ltc["rate_key"] and not is_pair_rate(ctc["rate_key"], ltc["rate_key"]):
            add("WARN", f"Video timecode runs at {ctc['rate_key']} but the LTC is {ltc['rate_key']}; they will drift "
                        "apart over the clip.")

    # --- filename hints
    if "tc" in hints:
        htc = hints["tc"]
        label = "%02d:%02d:%02d:%02d" % htc
        hint_nom = RATES[show][1] if show else (ltc["nominal"] if ltc else None)
        if ltc and hint_nom and htc[3] < hint_nom:
            if abs(label_seconds(htc, hint_nom) - ltc["label_s_at_zero"]) * ltc["nominal"] >= Fraction(1, 2):
                add("WARN", f"Filename suggests start {label} but the LTC reads {ltc['tc_at_zero']} at sample 0.")
            else:
                add("OK", f"Filename start TC {label} matches the LTC.")
        if bext and show and htc[3] < RATES[show][1]:
            rate, nom, dr = RATES[show]
            want = tc_samples(tc_to_frames(*htc, nom, dr), rate, sr)
            if want != bext["time_reference"] and not (ltc and abs(ltc["expected_bext"] - want) < sr / 60):
                add("WARN", f"Filename suggests start {label} but bext says {rep['bext_tc']}.",
                    None if ltc else f"bwfmetaedit --Timereference={want} \"{fname}\"")
    if "fps" in hints and ltc:
        hk = hints["fps"] + ("df" if hints.get("drop") and hints["fps"] in ("29.97", "59.94") else "")
        if hk in RATES and hk != ltc["rate_key"] and not is_pair_rate(hk, ltc["rate_key"]) and \
                RATES[hk][0] != ltc["rate"]:
            add("WARN", f"Filename says {hints['fps']} fps but the LTC measures {ltc['measured_fps']:.3f} fps "
                        f"({ltc['rate_key']}).")
        if "drop" in hints and hints["drop"] != ltc["drop"]:
            add("WARN", f"Filename says {'DF' if hints['drop'] else 'NDF'} but the LTC drop-frame flag is "
                        f"{'set' if ltc['drop'] else 'clear'}.")
    if "sample_rate" in hints and fmt.get("bits"):
        if hints["sample_rate"] != sr or hints["bits"] != fmt["bits"]:
            add("INFO", f"Filename says {hints['sample_rate']}x{hints['bits']} but the file is "
                        f"{sr}x{fmt['bits']}{' float' if fmt.get('float') else ''} (often the spec of the original "
                        "LTC source, not this export).")
    if "minutes" in hints and not rep["limited_to_s"] and abs(hints["minutes"] * 60 - dur) > 5:
        add("INFO", f"Filename says {hints['minutes']:g} min but the file is {dur // 60:.0f}:{dur % 60:04.1f}.")

    # --- iXML
    if is_wav and not rep["ixml"]:
        if ltc or bext:
            add("INFO", "No iXML chunk, so the frame rate / drop-frame flag isn't declared in the file. "
                        "Premiere shows the bext start using the clip's interpreted timebase; fine if your "
                        "sequence matches, but confirm the rate on set/in the edit.")
    elif rep["ixml"]:
        if not ixml.get("raw_ok"):
            add("WARN", "iXML chunk present but not valid XML.")
        if ixml.get("timestamp") is not None and bext and ixml["timestamp"] != bext["time_reference"]:
            add("WARN", f"iXML timestamp ({ixml['timestamp']}) disagrees with bext ({bext['time_reference']}). "
                        "Different apps may place the clip differently.")
        if decl_src == "iXML" and ltc and decl != ltc["rate_key"] and not is_pair_rate(decl, ltc["rate_key"]):
            add("WARN", f"iXML declares {decl} fps but the LTC is {ltc['rate_key']}.")

    # --- LTC health
    if len(ltcs) > 1:
        add("INFO", "LTC found on channels " + ", ".join(str(c["index"]) for c in ltcs) + "; using the first.")
    for c in rep["channels"]:
        if c.get("reverse_sync_words", 0) > 10 and "ltc" not in c:
            add("WARN", f"Channel {c['index']} looks like LTC recorded/played in reverse.")
    if ltc:
        ch = ltcs[0]
        if abs(ltc["ppm"]) > 100:
            add("WARN", f"LTC runs at {ltc['measured_fps']:.4f} fps, {ltc['ppm']:+.0f} ppm off {ltc['rate_key']}. "
                        "Possible varispeed, pull-up/down, or the file's sample rate was changed.")
        if ltc["drop"] and ltc["rate_key"] != "29.97df":
            add("WARN", "LTC drop-frame flag is set but the rate isn't 29.97.")
        if ltc["rate_key"] in ("24", "30"):
            alt = {"24": "23.976", "30": "29.97"}[ltc["rate_key"]]
            alt += f" (or {'47.952' if alt == '23.976' else '59.94'})"
            add("INFO", f"LTC is true {ltc['rate_key']} fps. If cameras/slates ran at {alt} instead, they drift "
                        f"apart by ~{dur / 1001:.2f}s over this {dur:.0f}s file (3.6s per hour). Make sure everything "
                        f"was jammed at the matching integer rate.")
        if ltc["jumps"]:
            first = ltc["jumps"][0]
            add("WARN", f"{len(ltc['jumps'])} LTC discontinuities (first at {first[0]:.2f}s: {first[1]} -> {first[2]}). "
                        "The audio was edited after the LTC was laid down, or the source wasn't continuous.")
        else:
            add("OK", f"LTC is continuous ({ltc['frames_decoded']} frames, {ltc['first_tc']} - {ltc['last_tc']}).")
        if ltc["dropouts"]:
            add("WARN", f"{ltc['dropouts']} LTC frames couldn't be decoded (dropouts or distortion).")
        if ltc["first_at_s"] > 0.1:
            add("INFO", f"LTC starts {ltc['first_at_s']:.2f}s into the file.")
        if dur - ltc["last_end_s"] > 0.1:
            add("INFO", f"LTC stops {dur - ltc['last_end_s']:.2f}s before the end of the file (tail has no timecode).")
        if ltc["subframe_samples"]:
            add("INFO", f"LTC frame boundaries are {ltc['subframe_samples']} samples off the file's sample 0 "
                        "(sub-frame offset; fine unless you need sample-accurate alignment).")
        pk = ch["peak_dbfs"]
        if pk is not None and pk < -30:
            add("WARN", f"LTC level is low ({pk:.1f} dBFS); some readers may not lock.")
        elif pk is not None and pk > -6:
            add("INFO", f"LTC peaks at {pk:.1f} dBFS. Fine in a file, but when playing it into a camera/slate "
                        "input, turn it down to about -18 to -12 dBFS to avoid distortion and bleed into other channels.")

    # --- general
    for c in rep["channels"]:
        if "ltc" not in c and c.get("clipped_samples"):
            add("INFO", f"Channel {c['index']} has {c['clipped_samples']} samples at full scale (possible clipping).")
        if c["peak_dbfs"] is None:
            add("INFO", f"Channel {c['index']} is silent.")
    if sr not in (48000, 96000):
        add("WARN", f"Sample rate is {sr} Hz; video workflows expect 48 kHz.")
    if is_wav and rep["kind"] == "ffmpeg":
        add("INFO", f"Audio codec '{fmt.get('codec')}' was decoded with ffmpeg; most NLEs expect plain PCM WAV.")
    known = {"fmt ", "data", "bext", "iXML", "JUNK", "junk", "LIST", "ds64", "fact", "cue ", "PAD ", "FLLR",
             "axml", "_PMX", "chna", "smpl", "inst", "acid", "minf", "elm1", "regn", "umid"}
    for cid, size in rep["chunks"]:
        if cid == "LGWV":
            add("INFO", f"LGWV chunk ({size} bytes) is Logic Pro project data; other apps ignore it.")
        elif cid not in known:
            add("INFO", f"Unrecognized chunk '{cid}' ({size} bytes).")

    if not ltc and not bext and not ctc:
        add("WARN", "No timecode found anywhere (no bext, no container TC, no LTC). "
                    "The clip can only be synced by waveform or by hand.")


def describe_offset(secs, ltc):
    a = abs(secs)
    hms = "%d:%02d:%06.3f" % (a // 3600, a % 3600 // 60, a % 60)
    return f"{hms} ({a * float(ltc['rate']):.1f} LTC frames), bext is {'later' if secs > 0 else 'earlier'}"


# ---------------------------------------------------------------- fixing

def video_tc_from_ltc(rep, ltc, ch, rate_key):
    """TC label the LTC carries at the first video frame, expressed at the video's rate."""
    shift = Fraction(rep["video"]["start"] - ch["stream_start"]).limit_denominator(10 ** 6)
    label_s = ltc["label_s_at_zero"] + shift * ltc["rate"] / ltc["nominal"]
    _, nom, drop = RATES[rate_key]
    return fmt_tc(frames_to_tc(math.floor(label_s * nom + Fraction(1, 1000)), nom, False), drop)


def build_plan(rep, trust_ltc=False):
    """What --fix would do: a dict describing the change, or None. Never touches files."""
    fmt, sr = rep["format"], rep["format"]["sr"]
    bext, ixml, hints, ctc = rep["bext"], rep["ixml"], rep["hints"], rep["container_tc"]
    ltcs = [c for c in rep["channels"] if "ltc" in c]
    ltc, ch = (ltcs[0]["ltc"], ltcs[0]) if ltcs else (None, None)
    show = rep.get("interpret_rate")
    is_wav = rep["container"] in ("RIFF", "RF64", "BW64")

    if rep["limited_to_s"]:
        return {"skip": "analysis was limited with --limit; rerun on the whole file to fix it"}

    if rep.get("video"):
        if not ltc:
            return None
        vkey = ctc.get("rate_key") if ctc else None
        if not vkey:
            vf = Fraction(rep["video"]["rate"]) if rep["video"].get("rate") not in (None, "0/0") else None
            vkey = vf and (rate_key_for(vf, ltc["drop"]) or rate_key_for(vf, False))
        if not vkey or not (RATES[vkey][0] == ltc["rate"] or is_pair_rate(vkey, ltc["rate_key"])):
            return {"skip": f"video rate {vkey or rep['video'].get('rate')} doesn't match the {ltc['rate_key']} LTC; "
                            "can't stamp LTC timecode onto it safely"}
        tc = video_tc_from_ltc(rep, ltc, ch, vkey)
        errs = [f for f in rep["findings"] if f["severity"] == "ERROR" and "Container timecode" in f["message"]]
        if ctc and not errs:
            return None
        if ctc and not trust_ltc:
            return {"skip": "camera timecode and LTC disagree; add --trust-ltc to re-stamp the video from the LTC"}
        return {"action": "video", "tc": tc, "rate_key": vkey,
                "describe": f"{'re-stamp' if ctc else 'add'} container timecode {tc} @ {vkey} from the LTC "
                            "(streams copied, not re-encoded)"}

    # Audio. Decide the start time to write.
    target, why = None, None
    if ltc:
        target, why = ltc["expected_bext"], "LTC"
    elif "tc" in hints and show and hints["tc"][3] < RATES[show][1] and not bext:
        rate, nom, dr = RATES[show]
        target, why = tc_samples(tc_to_frames(*hints["tc"], nom, dr), rate, sr), "filename"
    elif bext:
        target = bext["time_reference"]

    changes = []
    set_bext = None
    if target is not None:
        if not bext:
            set_bext = target
            changes.append(f"add bext start {samples_to_tc(target, sr, show) if show else target} (from {why})")
        elif ltc and abs(Fraction(bext["time_reference"] - target) * ltc["rate"] / sr) >= Fraction(1, 2):
            set_bext = target
            changes.append(f"set bext start {rep['bext_tc']} -> {samples_to_tc(target, sr, show)} (from LTC)")
    final = set_bext if set_bext is not None else (bext or {}).get("time_reference")
    ixml_key = show if final is not None else None
    if ixml_key:
        ix = ixml or {}
        want_rate = f"{RATES[ixml_key][0].numerator}/{RATES[ixml_key][0].denominator}"
        want_flag = "DF" if RATES[ixml_key][2] else "NDF"
        if not ixml:
            changes.append(f"add iXML declaring {ixml_key} fps {want_flag}")
        elif ix.get("timestamp") != final or ix.get("timecode_rate") != want_rate or \
                (ix.get("timecode_flag") or "").upper() != want_flag:
            changes.append(f"update iXML timecode to {ixml_key} {want_flag}, timestamp {final}")
        else:
            ixml_key = None
    if not changes:
        return None
    plan = {"action": "wav" if is_wav else "to_bwf", "time_ref": set_bext, "final_time_ref": final,
            "ixml_key": ixml_key, "describe": "; ".join(changes)}
    if not is_wav:
        plan["describe"] = f"convert {rep['container']} to Broadcast WAV, then " + plan["describe"]
    return plan


def new_bext(time_ref):
    now = __import__("datetime").datetime.now()
    return struct.pack("<256s32s32s10s8sQH64s190s", b"", b"analyze_tc.py", b"",
                       now.strftime("%Y-%m-%d").encode(), now.strftime("%H:%M:%S").encode(),
                       time_ref, 1, b"", b"")


def set_ixml_tag(text, tag, value):
    if re.search(rf"<{tag}>.*?</{tag}>", text, re.S):
        return re.sub(rf"<{tag}>.*?</{tag}>", f"<{tag}>{value}</{tag}>", text, count=1, flags=re.S)
    if "</SPEED>" not in text:
        text = text.replace("</BWFXML>", "<SPEED></SPEED></BWFXML>", 1)
    return text.replace("</SPEED>", f"<{tag}>{value}</{tag}></SPEED>", 1)


def make_ixml(old, time_ref, rate_key, sr):
    text = old.split(b"\0")[0].decode("utf-8", "replace") if old else ""
    if "</BWFXML>" not in text:
        text = '<?xml version="1.0" encoding="UTF-8"?><BWFXML><IXML_VERSION>2.10</IXML_VERSION></BWFXML>'
    rate, _, drop = RATES[rate_key]
    for tag, value in (("TIMECODE_RATE", f"{rate.numerator}/{rate.denominator}"),
                       ("TIMECODE_FLAG", "DF" if drop else "NDF"),
                       ("TIMESTAMP_SAMPLES_SINCE_MIDNIGHT_HI", time_ref >> 32),
                       ("TIMESTAMP_SAMPLES_SINCE_MIDNIGHT_LO", time_ref & 0xFFFFFFFF),
                       ("TIMESTAMP_SAMPLE_RATE", sr)):
        text = set_ixml_tag(text, tag, value)
    return text.encode("utf-8")


def write_fixed_wav(src_path, dst_path, time_ref, final_ref, ixml_key, sr):
    """Copy a WAV, setting/adding bext TimeReference and iXML. All other chunks are kept byte-for-byte."""
    with open(src_path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm, \
            open(dst_path, "wb") as out:
        riff = mm[:4]
        spans, p, ds64_data = [], 12, None
        while p + 8 <= len(mm):
            cid, size = mm[p:p + 4], struct.unpack("<I", mm[p + 4:p + 8])[0]
            real = size
            if cid == b"ds64":
                ds64_data = struct.unpack("<Q", mm[p + 16:p + 24])[0]
            if cid == b"data" and size == 0xFFFFFFFF and ds64_data is not None:
                real = ds64_data
            real = min(real, len(mm) - p - 8)
            spans.append((cid, size, p + 8, real))
            p += 8 + real + (real & 1)
        has_bext = any(c == b"bext" for c, *_ in spans)
        has_ixml = any(c == b"iXML" for c, *_ in spans)

        out.write(riff + b"\0\0\0\0WAVE")
        ds64_at = None

        def put(cid, body, size_field=None):
            out.write(cid + struct.pack("<I", len(body) if size_field is None else size_field))
            out.write(body)
            if len(body) % 2:
                out.write(b"\0")

        for cid, size, at, real in spans:
            if cid == b"data":
                if time_ref is not None and not has_bext:
                    put(b"bext", new_bext(time_ref))
                if ixml_key and not has_ixml:
                    put(b"iXML", make_ixml(None, final_ref, ixml_key, sr))
                out.write(cid + struct.pack("<I", size))
                for off in range(at, at + real, 1 << 26):
                    out.write(mm[off:min(at + real, off + (1 << 26))])
                if real % 2:
                    out.write(b"\0")
            elif cid == b"bext" and time_ref is not None:
                body = bytearray(mm[at:at + real])
                body[338:346] = struct.pack("<Q", time_ref)
                put(cid, bytes(body))
            elif cid == b"iXML" and ixml_key:
                put(cid, make_ixml(mm[at:at + real], final_ref, ixml_key, sr))
            else:
                if cid == b"ds64":
                    ds64_at = out.tell() + 8
                put(cid, mm[at:at + real], size)
        total = out.tell()
        if riff == b"RIFF":
            if total - 8 > 0xFFFFFFFF:
                raise ValueError("fixed file would exceed 4 GB; convert to RF64 first")
            out.seek(4)
            out.write(struct.pack("<I", total - 8))
        else:
            out.seek(4)
            out.write(struct.pack("<I", 0xFFFFFFFF))
            if ds64_at is not None:
                out.seek(ds64_at)
                out.write(struct.pack("<Q", total - 8))


def apply_fix(rep, plan, in_place):
    path = rep["file"]
    stem, ext = os.path.splitext(path)
    if plan["action"] == "to_bwf":
        dst = stem + ".tcfix.wav"
    else:
        dst = path if in_place else stem + ".tcfix" + ext
    work = dst + ".tmp" if dst == path else dst
    if os.path.exists(work) or (dst != path and os.path.exists(dst)):
        raise ValueError(f"{dst} already exists; move or delete it first")
    sr = rep["format"]["sr"]
    try:
        if plan["action"] == "wav":
            write_fixed_wav(path, work, plan["time_ref"], plan["final_time_ref"], plan["ixml_key"], sr)
        elif plan["action"] == "to_bwf":
            bits = rep["format"].get("bits") if rep["format"].get("bits") in (16, 24, 32) else 24
            tmp = work + ".conv.wav"
            r = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-map", "0:a", "-c:a", f"pcm_s{bits}le",
                                "-rf64", "auto", tmp], capture_output=True, text=True)
            if r.returncode:
                raise ValueError("ffmpeg conversion failed: " + r.stderr.strip()[:300])
            try:
                write_fixed_wav(tmp, work, plan["time_ref"], plan["final_time_ref"], plan["ixml_key"], sr)
            finally:
                os.remove(tmp)
        elif plan["action"] == "video":
            fmt_name = "mov" if ext.lower() in (".mov", ".qt") else None
            cmd = ["ffmpeg", "-v", "error", "-i", path, "-map", "0", "-map", "-0:d?", "-c", "copy",
                   "-timecode", plan["tc"]] + (["-f", fmt_name] if fmt_name else []) + [work]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode:
                raise ValueError("ffmpeg re-stamp failed: " + r.stderr.strip()[:300])
        if work != dst:
            os.replace(work, dst)
    except BaseException:
        if os.path.exists(work) and work != path:
            os.remove(work)
        raise
    return dst


# ---------------------------------------------------------------- output

def print_report(rep):
    f, b, ix, ctc = rep["format"], rep["bext"], rep["ixml"], rep["container_tc"]
    dur = f["duration_s"]
    print("=" * 78)
    print(os.path.basename(rep["file"]))
    print("=" * 78)
    codec = f" {f['codec']}" if f.get("codec") else ""
    print(f"Format    {rep['container']}{codec} | {f.get('channels')} ch | {f['sr']} Hz | {f.get('bits') or '?'}-bit"
          f"{' float' if f.get('float') else ''} | {int(dur // 60)}:{dur % 60:06.3f}"
          + (" (analyzed)" if rep["limited_to_s"] else ""))
    if rep.get("video"):
        v = rep["video"]
        print(f"Video     {v['codec']} {v['size']} @ {v['rate']}")
    if rep["chunks"]:
        print("Chunks    " + ", ".join(f"{c.strip()}({s})" for c, s in rep["chunks"]))
    if b:
        print(f"bext      TimeReference {b['time_reference']} = {rep['bext_clock']}"
              + (f" = {rep['bext_tc']} @ {rep['interpret_rate']}" if rep.get("bext_tc") else ""))
        print(f"          originator '{b['originator']}', {b['date']} {b['time']}, v{b['version']}"
              + (f", desc '{b['description']}'" if b["description"] else ""))
    elif rep["container"] in ("RIFF", "RF64", "BW64"):
        print("bext      none")
    if ctc:
        print(f"TC track  {ctc['tc']}" + (f" @ {ctc['rate_key']}" if ctc.get("rate_key") else ""))
    if ix:
        print(f"iXML      rate {ix.get('timecode_rate')} {ix.get('timecode_flag') or ''}, timestamp {ix.get('timestamp')}"
              + (", tracks: " + ", ".join(f"{i}={n}" for i, n in ix["tracks"]) if ix.get("tracks") else ""))
    elif rep["container"] in ("RIFF", "RF64", "BW64"):
        print("iXML      none")
    if rep["hints"]:
        print("Filename  " + ", ".join(f"{k}={v}" for k, v in rep["hints"].items()))
    for c in rep["channels"]:
        pk = "silent" if c["peak_dbfs"] is None else f"peak {c['peak_dbfs']:.1f} dBFS"
        span = f", signal {c['signal_from_s']:.2f}-{c['signal_to_s']:.2f}s" if "signal_from_s" in c else ""
        print(f"Ch {c['index']:<6} {pk}{span}")
        if "ltc" in c:
            l = c["ltc"]
            print(f"          LTC {l['rate_key']} ({l['measured_fps']:.4f} fps measured, {l['ppm']:+.0f} ppm), "
                  f"{'DF' if l['drop'] else 'NDF'}, {l['frames_decoded']} frames")
            print(f"          TC at sample 0: {l['tc_at_zero']}   first {l['first_tc']} @ {l['first_at_s']:.3f}s,"
                  f" last {l['last_tc']} ending @ {l['last_end_s']:.2f}s")
    print("-" * 78)
    order = {"ERROR": 0, "WARN": 1, "INFO": 2, "OK": 3}
    for x in sorted(rep["findings"], key=lambda x: order[x["severity"]]):
        print(f"[{x['severity']:<5}] {x['message']}")
        if x["fix"]:
            print(f"        fix: {x['fix']}")
    plan = rep.get("plan")
    if plan:
        print("-" * 78)
        if "skip" in plan:
            print(f"AUTO-FIX  not available: {plan['skip']}")
        elif rep.get("fixed_to"):
            print(f"FIXED     {plan['describe']}")
            print(f"          wrote {rep['fixed_to']}")
            if rep.get("fix_check"):
                print(f"          re-check: {rep['fix_check']}")
        else:
            print(f"AUTO-FIX  --fix would: {plan['describe']}")
    errors = [x for x in rep["findings"] if x["severity"] == "ERROR"]
    has_tc = rep["bext"] or rep["container_tc"] or any("ltc" in c for c in rep["channels"])
    print("-" * 78)
    if rep.get("fixed_to") and rep.get("fix_check", "").startswith("OK"):
        print("VERDICT: fixed; " + ("the original was updated in place." if rep["fixed_to"] == rep["file"]
                                    else f"use {os.path.basename(rep['fixed_to'])} (original unchanged)."))
        print()
        return
    print("VERDICT: " + ("timecode needs fixing before syncing by TC in Premiere." if errors
                         else "OK to sync by timecode." if has_tc
                         else "no timecode; sync by waveform or manually."))
    print()


def to_json(rep):
    def conv(o):
        if isinstance(o, Fraction):
            return float(o)
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        return o
    return conv(rep)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+")
    p.add_argument("--fps", choices=RATES, help="project frame rate (default: from container, iXML, filename, or LTC)")
    p.add_argument("--limit", type=float, metavar="SECONDS", help="only analyze the first SECONDS of audio")
    p.add_argument("--json", action="store_true", help="print machine-readable JSON instead of a report")
    p.add_argument("--fix", action="store_true",
                   help="write a fixed copy next to each file (name.tcfix.ext), then re-check it")
    p.add_argument("--in-place", action="store_true", help="with --fix: replace the original file instead of copying")
    p.add_argument("--trust-ltc", action="store_true",
                   help="with --fix: re-stamp video files whose camera timecode disagrees with their LTC")
    a = p.parse_args()
    if (a.in_place or a.trust_ltc) and not a.fix:
        p.error("--in-place and --trust-ltc only apply with --fix")
    reports, status = [], 0
    for path in a.files:
        try:
            rep = analyze(path, a.fps, a.limit)
        except (OSError, ValueError, struct.error) as e:
            print(f"{path}: cannot analyze: {e}", file=sys.stderr)
            status = max(status, 2)
            continue
        rep["plan"] = build_plan(rep, a.trust_ltc)
        failed = any(x["severity"] == "ERROR" for x in rep["findings"])
        if a.fix and rep["plan"] and "action" in rep["plan"]:
            try:
                rep["fixed_to"] = apply_fix(rep, rep["plan"], a.in_place)
                after = analyze(rep["fixed_to"], a.fps)
                errs = [x["message"] for x in after["findings"] if x["severity"] == "ERROR"]
                rep["fix_check"] = ("OK, no errors remain" if not errs
                                    else f"{len(errs)} error(s) remain: " + errs[0])
                failed = bool(errs)
            except (OSError, ValueError) as e:
                rep["fix_error"] = str(e)
                print(f"{path}: fix failed: {e}", file=sys.stderr)
                status = max(status, 2)
        if failed:
            status = max(status, 1)
        if a.json:
            reports.append(to_json(rep))
        else:
            print_report(rep)
    if a.json:
        print(json.dumps(reports, indent=2))
    sys.exit(status)


if __name__ == "__main__":
    main()

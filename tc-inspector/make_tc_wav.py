#!/usr/bin/env python3
# Copyright (c) 2026 Makai Systems LLC
# SPDX-License-Identifier: BUSL-1.1
# Free to run as a standalone tool; selling it or incorporating it into a
# commercial product requires a license. See LICENSE for the full terms.
# For commercial licensing, contact licensing@makaisystems.com.
"""Generate a polyphonic Broadcast WAV (BWF) file with embedded timecode.

Timecode is carried three ways:
  * bext chunk TimeReference (samples since midnight) - read by Pro Tools, Resolve, Premiere, etc.
  * iXML chunk (timecode rate, drop/non-drop flag, timestamp, track names)
  * an LTC (SMPTE linear timecode) audio track

Pure standard library; no numpy required.
"""
import argparse
import array
import math
import struct
import sys
from datetime import datetime
from fractions import Fraction

RATES = {
    "23.976": (Fraction(24000, 1001), 24, False),
    "24": (Fraction(24), 24, False),
    "25": (Fraction(25), 25, False),
    "29.97": (Fraction(30000, 1001), 30, False),
    "29.97df": (Fraction(30000, 1001), 30, True),
    "30": (Fraction(30), 30, False),
}

FULL_SCALE = (1 << 23) - 1


def db(v):
    return 10 ** (v / 20)


# ---------------------------------------------------------------- timecode math

def tc_to_frames(tc, nominal, drop):
    h, m, s, f = (int(x) for x in tc.replace(";", ":").split(":"))
    frames = ((h * 3600 + m * 60 + s) * nominal) + f
    if drop:
        total_min = h * 60 + m
        frames -= 2 * (total_min - total_min // 10)
    return frames


def frames_to_tc(frames, nominal, drop):
    if drop:
        d, mod = divmod(frames, 17982)
        frames += 18 * d + (2 * ((mod - 2) // 1798) if mod > 1 else 0)
    f = frames % nominal
    s = (frames // nominal) % 60
    m = (frames // (nominal * 60)) % 60
    h = (frames // (nominal * 3600)) % 24
    return h, m, s, f


# ---------------------------------------------------------------- LTC encoding

def ltc_bits(h, m, s, f, drop, nominal):
    bits = [0] * 80

    def put(start, value, width):
        for i in range(width):
            bits[start + i] = (value >> i) & 1

    put(0, f % 10, 4)
    put(8, f // 10, 2)
    bits[10] = 1 if drop else 0
    put(16, s % 10, 4)
    put(24, s // 10, 3)
    put(32, m % 10, 4)
    put(40, m // 10, 3)
    put(48, h % 10, 4)
    put(56, h // 10, 2)
    for i, b in enumerate((0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1)):
        bits[64 + i] = b
    # Polarity-correction bit keeps the number of zeros in each word even.
    pc = 59 if nominal == 25 else 27
    if bits.count(0) % 2:
        bits[pc] = 1
    return bits


def render_ltc(n, sr, rate, start_frame, nominal, drop, amp):
    out = [0.0] * n
    spf = Fraction(sr) / rate  # samples per frame (may be fractional)
    level = 1.0
    frame = 0
    while True:
        frame_start = spf * frame
        if frame_start >= n:
            break
        bits = ltc_bits(*frames_to_tc(start_frame + frame, nominal, drop), drop, nominal)
        for i, bit in enumerate(bits):
            edges = [frame_start + spf * Fraction(2 * i, 160),
                     frame_start + spf * Fraction(2 * i + 1, 160),
                     frame_start + spf * Fraction(2 * i + 2, 160)]
            a, mid, b = (min(n, round(e)) for e in edges)
            level = -level  # transition at every bit boundary
            if bit:
                for k in range(a, mid):
                    out[k] = level * amp
                level = -level  # extra mid-bit transition for a 1
                for k in range(mid, b):
                    out[k] = level * amp
            else:
                for k in range(a, b):
                    out[k] = level * amp
        frame += 1
    return out


# ---------------------------------------------------------------- test signals

def sine(n, sr, freq, amp):
    w = 2 * math.pi * freq / sr
    return [amp * math.sin(w * i) for i in range(n)]


def pips(n, sr, rate, start_frame, nominal, drop, freq, amp):
    """1-frame tone burst at the top of every timecode second."""
    out = [0.0] * n
    spf = Fraction(sr) / rate
    w = 2 * math.pi * freq / sr
    frame = 0
    while spf * frame < n:
        if frames_to_tc(start_frame + frame, nominal, drop)[3] == 0:
            a, b = round(spf * frame), min(n, round(spf * (frame + 1)))
            for k in range(a, b):
                out[k] = amp * math.sin(w * (k - a))
        frame += 1
    return out


def log_sweep(n, sr, f0, f1, period_s, amp):
    out = [0.0] * n
    T = period_s
    k = math.log(f1 / f0)
    for i in range(n):
        t = (i / sr) % T
        out[i] = amp * math.sin(2 * math.pi * f0 * T / k * (math.exp(t * k / T) - 1))
    return out


# ---------------------------------------------------------------- RIFF writing

def chunk(cid, data):
    pad = b"\0" if len(data) % 2 else b""
    return cid + struct.pack("<I", len(data)) + data + pad


def fmt_chunk(channels, sr, bits):
    block = channels * bits // 8
    sub_pcm = bytes.fromhex("0100000000001000800000AA00389B71")
    return chunk(b"fmt ", struct.pack(
        "<HHIIHHHHI16s", 0xFFFE, channels, sr, sr * block, block, bits,
        22, bits, 0, sub_pcm))  # WAVE_FORMAT_EXTENSIBLE, channel mask 0 = discrete tracks


def bext_chunk(desc, time_ref, now, sr, bits, channels):
    coding = f"A=PCM,F={sr},W={bits},M=multitrack,T=make_tc_wav.py;CH={channels}\r\n".encode()
    data = struct.pack(
        "<256s32s32s10s8sIIH64s5h180s",
        desc.encode()[:256], b"make_tc_wav.py", b"TEFFT-TC-TEST",
        now.strftime("%Y-%m-%d").encode(), now.strftime("%H:%M:%S").encode(),
        time_ref & 0xFFFFFFFF, time_ref >> 32, 2, b"",
        0x7FFF, 0x7FFF, 0x7FFF, 0x7FFF, 0x7FFF,  # loudness values: not set
        b"") + coding
    return chunk(b"bext", data)


def ixml_chunk(project, scene, take, rate_key, rate, drop, sr, bits, time_ref, names, start_tc):
    tracks = "".join(
        f"<TRACK><CHANNEL_INDEX>{i}</CHANNEL_INDEX><INTERLEAVE_INDEX>{i}</INTERLEAVE_INDEX>"
        f"<NAME>{name}</NAME></TRACK>"
        for i, name in enumerate(names, 1))
    speed = f"{rate.numerator}/{rate.denominator}"
    master = speed
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<BWFXML><IXML_VERSION>2.10</IXML_VERSION>"
        f"<PROJECT>{project}</PROJECT><SCENE>{scene}</SCENE><TAKE>{take}</TAKE>"
        f"<NOTE>Start TC {start_tc} @ {rate_key}</NOTE>"
        "<SPEED>"
        f"<MASTER_SPEED>{master}</MASTER_SPEED><CURRENT_SPEED>{master}</CURRENT_SPEED>"
        f"<TIMECODE_RATE>{speed}</TIMECODE_RATE>"
        f"<TIMECODE_FLAG>{'DF' if drop else 'NDF'}</TIMECODE_FLAG>"
        f"<FILE_SAMPLE_RATE>{sr}</FILE_SAMPLE_RATE><AUDIO_BIT_DEPTH>{bits}</AUDIO_BIT_DEPTH>"
        f"<DIGITIZER_SAMPLE_RATE>{sr}</DIGITIZER_SAMPLE_RATE>"
        f"<TIMESTAMP_SAMPLE_RATE>{sr}</TIMESTAMP_SAMPLE_RATE>"
        f"<TIMESTAMP_SAMPLES_SINCE_MIDNIGHT_HI>{time_ref >> 32}</TIMESTAMP_SAMPLES_SINCE_MIDNIGHT_HI>"
        f"<TIMESTAMP_SAMPLES_SINCE_MIDNIGHT_LO>{time_ref & 0xFFFFFFFF}</TIMESTAMP_SAMPLES_SINCE_MIDNIGHT_LO>"
        "</SPEED>"
        f"<TRACK_LIST><TRACK_COUNT>{len(names)}</TRACK_COUNT>{tracks}</TRACK_LIST>"
        "</BWFXML>")
    return chunk(b"iXML", xml.encode())


def interleave_24(channels):
    n = len(channels[0])
    ints = array.array("i", bytes(4 * n * len(channels)))
    for c, samples in enumerate(channels):
        ints[c::len(channels)] = array.array(
            "i", (max(-FULL_SCALE, min(FULL_SCALE, round(s * FULL_SCALE))) for s in samples))
    if sys.byteorder != "little":
        ints.byteswap()
    raw = ints.tobytes()
    out = bytearray(len(raw) // 4 * 3)
    out[0::3], out[1::3], out[2::3] = raw[0::4], raw[1::4], raw[2::4]
    return bytes(out)


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-o", "--output", default="tc_multitrack.wav")
    p.add_argument("--start", default="01:00:00:00", help="start timecode HH:MM:SS:FF (default 01:00:00:00)")
    p.add_argument("--fps", default="24", choices=RATES)
    p.add_argument("--sr", type=int, default=48000)
    p.add_argument("--duration", type=float, default=30.0, help="seconds")
    a = p.parse_args()

    rate, nominal, drop = RATES[a.fps]
    sr, bits = a.sr, 24
    n = int(a.duration * sr)
    start_frame = tc_to_frames(a.start, nominal, drop)
    # Samples since midnight: real elapsed time of start_frame frames at the actual rate.
    time_ref = round(Fraction(start_frame) / rate * sr)

    print(f"Rendering {a.duration}s @ {sr} Hz / {bits}-bit, TC {a.start} @ {a.fps} ...")
    tracks = [
        ("REF_1K", sine(n, sr, 1000, db(-20))),
        ("PIP", pips(n, sr, rate, start_frame, nominal, drop, 1000, db(-12))),
        ("TONE_A3", sine(n, sr, 220.00, db(-24))),
        ("TONE_C#4", sine(n, sr, 277.18, db(-24))),
        ("TONE_E4", sine(n, sr, 329.63, db(-24))),
        ("TONE_A4", sine(n, sr, 440.00, db(-24))),
        ("SWEEP", log_sweep(n, sr, 20, 20000, 10, db(-20))),
        ("LTC", render_ltc(n, sr, rate, start_frame, nominal, drop, db(-12))),
    ]
    names = [t[0] for t in tracks]
    data = interleave_24([t[1] for t in tracks])

    now = datetime.now()
    body = (b"WAVE"
            + fmt_chunk(len(tracks), sr, bits)
            + bext_chunk(f"Multitrack timecode test. Start {a.start} @ {a.fps} fps", time_ref, now, sr, bits, len(tracks))
            + ixml_chunk("TEFFT", "TC_TEST", "1", a.fps, rate, drop, sr, bits, time_ref, names, a.start)
            + chunk(b"data", data))
    with open(a.output, "wb") as fh:
        fh.write(b"RIFF" + struct.pack("<I", len(body)) + body)

    end = frames_to_tc(start_frame + int(a.duration * rate), nominal, drop)
    sep = ";" if drop else ":"
    print(f"Wrote {a.output}: {len(tracks)} tracks, TimeReference={time_ref} samples")
    print("Tracks: " + ", ".join(f"{i}={nm}" for i, nm in enumerate(names, 1)))
    print(f"TC range: {a.start} -> {end[0]:02d}:{end[1]:02d}:{end[2]:02d}{sep}{end[3]:02d}")


if __name__ == "__main__":
    main()

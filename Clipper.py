#!/usr/bin/env python3
"""Paste a YouTube link -> get the best 30-60s clips. 100% free, runs locally."""
import argparse, re, subprocess, tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import yt_dlp
from faster_whisper import WhisperModel

HOOKS = {
    "secret", "mistake", "never", "always", "truth", "actually", "crazy", "insane",
    "biggest", "worst", "best", "why", "how", "wow", "unbelievable", "shocking",
    "important", "problem", "lesson", "learned", "warning", "stop", "change", "money",
    "free", "hack", "trick", "story", "believe", "love", "hate", "fear",
}


def download(url, out_dir):
    opts = {
        "format": "bv*[height<=1080]+ba/b[height<=1080]",
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / "source.%(ext)s"),
        "quiet": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    return Path(info["requested_downloads"][0]["filepath"]), info.get("title", "video")


def audio_energy(video):
    """RMS loudness for each second of audio."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "quiet", "-i", str(video), "-f", "s16le", "-ac", "1", "-ar", "16000", "-"],
        capture_output=True, check=True,
    ).stdout
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768
    n = len(a) // 16000
    return np.sqrt((a[: n * 16000].reshape(n, 16000) ** 2).mean(axis=1)) if n else np.zeros(1)


@lru_cache(maxsize=2)
def get_model(model_size):
    return WhisperModel(model_size, compute_type="int8")  # runs fine on CPU


def transcribe(video, model_size):
    model = get_model(model_size)
    segs, info = model.transcribe(str(video), vad_filter=True)
    out = [(s.start, s.end, s.text.strip()) for s in segs]
    if not out:  # music / no speech: fall back to fixed blocks
        out = [(t, t + 5, "") for t in range(0, int(info.duration), 5)]
    return out


def candidates(segs, energy, min_len, max_len):
    base = energy.mean() + 1e-9
    cands = []
    for i, (start, _, _) in enumerate(segs):
        for j in range(i, len(segs)):
            end = segs[j][1]
            dur = end - start
            if dur > max_len:
                break
            if dur < min_len:
                continue
            text = " ".join(s[2] for s in segs[i : j + 1])
            words = re.findall(r"[a-z']+", text.lower())
            e = energy[int(start) : int(end) + 1]
            cands.append({
                "start": start, "end": end,
                "energy": float(e.mean() / base) if len(e) else 0,
                "rate": len(words) / dur,
                "hooks": sum(w in HOOKS for w in words) + text.count("?") + text.count("!"),
                "ends_clean": text.rstrip().endswith((".", "?", "!")),
                "text": text,
            })
    for key in ("energy", "rate", "hooks"):
        m = max((c[key] for c in cands), default=1) or 1
        for c in cands:
            c[key + "_n"] = c[key] / m
    for c in cands:
        c["score"] = 0.4 * c["energy_n"] + 0.3 * c["rate_n"] + 0.3 * c["hooks_n"] + (0.1 if c["ends_clean"] else 0)
    return sorted(cands, key=lambda c: c["score"], reverse=True)


def pick(cands, n):
    chosen = []
    for c in cands:
        if all(c["end"] <= x["start"] or c["start"] >= x["end"] for x in chosen):
            chosen.append(c)
        if len(chosen) == n:
            break
    return sorted(chosen, key=lambda c: c["start"])


def cut(video, c, out, vertical):
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{c['start']:.2f}", "-i", str(video), "-t", f"{c['end'] - c['start']:.2f}"]
    if vertical:
        cmd += ["-vf", "crop=ih*9/16:ih,scale=1080:1920"]
    subprocess.run(cmd + ["-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", str(out)], check=True)


def main():
    p = argparse.ArgumentParser(description="Auto-clip the best parts of a YouTube video.")
    p.add_argument("url")
    p.add_argument("-n", "--num-clips", type=int, default=5)
    p.add_argument("--min", type=int, default=30, help="min clip seconds")
    p.add_argument("--max", type=int, default=60, help="max clip seconds")
    p.add_argument("--model", default="base", help="whisper: tiny/base/small/medium")
    p.add_argument("--vertical", action="store_true", help="crop to 9:16 for Shorts/TikTok")
    p.add_argument("-o", "--out", default="clips")
    a = p.parse_args()

    out_dir = Path(a.out)
    out_dir.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        print("Downloading...")
        video, title = download(a.url, Path(tmp))
        print("Analysing audio...")
        energy = audio_energy(video)
        print("Transcribing (first run downloads the model)...")
        segs = transcribe(video, a.model)
        best = pick(candidates(segs, energy, a.min, a.max), a.num_clips)
        for k, c in enumerate(best, 1):
            f = out_dir / f"clip_{k}_{int(c['start'])}s-{int(c['end'])}s.mp4"
            cut(video, c, f, a.vertical)
            print(f"[{k}] {f.name}  score={c['score']:.2f}  \"{c['text'][:80]}...\"")
    print(f"Done. Clips saved in ./{out_dir}")


if __name__ == "__main__":
    main()

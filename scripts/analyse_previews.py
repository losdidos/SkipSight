"""Analyse the 30s Deezer preview of every track ourselves (librosa) -> data/processed/track_audio_preview.csv

Resumable: tracks already in the output are skipped. Run after/while enrich_metadata.py --source deezer runs.
The preview URL is a time-limited token, so the track is re-fetched right before downloading.
The mp3 is analysed in memory/temp and deleted; only the numbers are kept.
NOTE: the preview is a ~30s excerpt from the middle of the song, so these describe the sound, not the intro.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "processed" / "track_audio_preview.csv"
DEEZER = ROOT / "data" / "raw" / "deezer.csv"
SR = 22050


def get_preview(dz_id: int) -> bytes | None:
    for attempt in range(4):
        try:
            r = requests.get(f"https://api.deezer.com/track/{dz_id}", timeout=30)
            if r.status_code == 200:
                url = r.json().get("preview")
                if not url:
                    return None
                a = requests.get(url, timeout=60)
                return a.content if a.status_code == 200 and len(a.content) > 10_000 else None
        except requests.RequestException:
            pass
        time.sleep(3 * (attempt + 1))
    return None


def analyse(raw: bytes) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fh:
        fh.write(raw)
    try:
        y, sr = librosa.load(fh.name, sr=SR, mono=True)
    finally:
        os.unlink(fh.name)
    f: dict = {"pv_seconds": round(len(y) / sr, 2)}
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(20):
        f[f"pv_mfcc{i + 1}_mean"] = mfcc[i].mean()
        f[f"pv_mfcc{i + 1}_std"] = mfcc[i].std()
    for name, arr in {
        "centroid": librosa.feature.spectral_centroid(y=y, sr=sr)[0],
        "bandwidth": librosa.feature.spectral_bandwidth(y=y, sr=sr)[0],
        "rolloff": librosa.feature.spectral_rolloff(y=y, sr=sr)[0],
        "flatness": librosa.feature.spectral_flatness(y=y)[0],
        "zcr": librosa.feature.zero_crossing_rate(y)[0],
    }.items():
        f[f"pv_{name}_mean"], f[f"pv_{name}_std"] = arr.mean(), arr.std()
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    f["pv_contrast_mean"] = contrast.mean()
    rms = librosa.feature.rms(y=y)[0]
    f["pv_rms_mean"], f["pv_rms_std"] = rms.mean(), rms.std()
    f["pv_dynamic_range_db"] = float(20 * np.log10((np.percentile(rms, 95) + 1e-9) / (np.percentile(rms, 5) + 1e-9)))
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    f["pv_onset_mean"] = onset.mean()
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset, sr=sr)
    f["pv_tempo"] = float(np.atleast_1d(tempo)[0])
    f["pv_onset_rate"] = len(librosa.onset.onset_detect(onset_envelope=onset, sr=sr)) / (len(y) / sr)
    ibi = np.diff(beats)
    f["pv_beat_regularity"] = float(1 - min(1, ibi.std() / ibi.mean())) if len(ibi) > 2 and ibi.mean() > 0 else 0.0
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i, v in enumerate(chroma.mean(axis=1)):
        f[f"pv_chroma{i}"] = v
    harm, perc = librosa.effects.hpss(y)
    f["pv_harmonic_ratio"] = float(np.sum(harm**2) / (np.sum(harm**2) + np.sum(perc**2) + 1e-9))
    return f


def work(args: tuple) -> dict:
    uri, tid = args
    row = {"uri": uri, "pv_ok": 0}
    raw = get_preview(int(tid))
    if raw:
        try:
            row.update(analyse(raw))
            row["pv_ok"] = 1
        except Exception as exc:
            print("analysis failed", uri, exc, flush=True)
    return row


def main() -> None:
    from multiprocessing import Pool

    if not DEEZER.exists():
        sys.exit("run enrich_metadata.py --source deezer first")
    dz = pd.read_csv(DEEZER, dtype={"dz_track_id": "Int64"})
    dz = dz[dz.dz_found == 1].dropna(subset=["dz_track_id"])
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["uri"])
    todo = dz[~dz.uri.isin(done.uri)]
    print(f"{len(done)} analysed, {len(todo)} to do", flush=True)
    rows = []
    with Pool(max(2, (os.cpu_count() or 4) - 1)) as pool:
        for n, row in enumerate(pool.imap_unordered(work, list(zip(todo.uri, todo.dz_track_id)), chunksize=2), 1):
            rows.append(row)
            if n % 40 == 0 or n == len(todo):
                done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
                done.to_csv(OUT, index=False)
                rows = []
                print(f"  {len(done)} analysed", flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()

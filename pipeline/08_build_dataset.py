"""Step 08 - Build the final, analysis-ready tables.

Input : data/interim/tracks_enriched.csv, data/interim/plays_labeled.csv
Output: data/output/tracks.csv   one row per song: API data + derived song columns
        data/output/plays.csv    one row per play: label + context + age columns

Derived song columns
- tempo_folded   : tempo moved into one octave [70, 140) BPM. Tempo detectors often report half or
                   double the real tempo (11 % of songs disagree with Deezer by exactly 2x), so
                   70 and 140 BPM are treated as the same tempo.
- tempo_check    : does ReccoBeats' tempo agree with Deezer's BPM? same / octave / different / unknown
- release_year, age_at_release (my age when the song came out; negative = before I was born)
- genre_* flags  : 1 if the song has that genre. Genres come from the APIs as LISTS OF EXACT NAMES
                   (Deezer album genres, MusicBrainz artist genres), so we test list membership:
                   "dance" can never match "dancehall". Only genres with >= MIN_SONGS songs get a column.
- title_* flags  : version words in the title, matched as whole words (\b): "live" doesn't match "Oliver".

Derived play columns
- my_age_at_play, song_age_at_play (years between release and this play; 0 = it was new then)
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from common import CONFIG, data_path
from matching import norm

MIN_SONGS = 30
TITLE_FLAGS = {
    "remix": r"\bremix(?:ed)?\b", "sped_up": r"\bsped\s*up\b|\bspeed\s*up\b|\bnightcore\b",
    "slowed": r"\bslowed\b", "live": r"\blive\b", "acoustic": r"\bacoustic\b",
    "remaster": r"\bremaster(?:ed)?\b", "feat": r"\b(?:feat|ft|featuring)\b\.?|\(with\b",
    "edit": r"\b(?:radio|extended|club) (?:edit|mix|version)\b",
}


def fold_tempo(bpm: pd.Series) -> pd.Series:
    folded = bpm.where(bpm > 0)
    for _ in range(4):
        folded = folded.where(folded < 140, folded / 2).where(folded >= 70, folded * 2)
    return folded.round(1)


def tempo_check(rb: pd.Series, dz: pd.Series) -> pd.Series:
    ratio = rb / dz.where(dz > 0)
    same = (ratio - 1).abs() < 0.03
    octave = ((ratio - 2).abs() < 0.06) | ((ratio - 0.5).abs() < 0.015)
    return pd.Series(np.select([ratio.isna(), same, octave], ["unknown", "same", "octave"], "different"), index=rb.index)


def genre_lists(tracks: pd.DataFrame) -> pd.Series:
    """Union of Deezer album genres and MusicBrainz artist genres, as a set of exact names per song."""
    def split(value) -> set[str]:
        return {norm(g).replace(" ", "_") for g in str(value).split("|") if g} if isinstance(value, str) else set()
    deezer = tracks["dz_album_genres"].map(split)
    musicbrainz = tracks.get("mb_genres", pd.Series(index=tracks.index, dtype=object)).map(split)
    return pd.Series([d | m for d, m in zip(deezer, musicbrainz)], index=tracks.index)


def main() -> None:
    tracks = pd.read_csv(data_path("interim", "tracks_enriched.csv"))
    birth_year = CONFIG["listener"]["birth_year"]

    tracks["tempo_folded"] = fold_tempo(tracks["tempo"])
    tracks["tempo_check"] = tempo_check(tracks["tempo"], tracks["dz_bpm"])
    tracks["age_at_release"] = tracks["release_year"] - birth_year

    genres = genre_lists(tracks)
    counts = pd.Series([g for gs in genres for g in gs]).value_counts()
    kept = counts[counts >= MIN_SONGS].index
    flags = pd.DataFrame({f"genre_{g}": genres.map(lambda gs, g=g: int(g in gs)) for g in kept}, index=tracks.index)
    tracks["genres"] = genres.map(lambda gs: "|".join(sorted(gs)))
    for name, pattern in TITLE_FLAGS.items():
        tracks[f"title_{name}"] = tracks["track_name"].str.contains(pattern, flags=re.I, regex=True).astype(int)
    tracks = pd.concat([tracks, flags], axis=1)
    tracks.to_csv(data_path("output", "tracks.csv"), index=False)

    plays = pd.read_csv(data_path("interim", "plays_labeled.csv"))
    play_year = pd.to_datetime(plays["start_local"]).dt.year
    release = plays["track_id"].map(tracks.set_index("track_id")["release_year"])
    plays["my_age_at_play"] = play_year - birth_year
    plays["song_age_at_play"] = (play_year - release).clip(lower=0)
    plays.to_csv(data_path("output", "plays.csv"), index=False)

    print(f"tracks.csv: {len(tracks):,} songs, {tracks.shape[1]} columns; plays.csv: {len(plays):,} plays")
    print("tempo check vs Deezer:", tracks["tempo_check"].value_counts().to_dict())
    print(f"{len(kept)} genre flags (genres on >= {MIN_SONGS} songs):", ", ".join(kept[:25]), "...")
    print("title flags:", {k: int(tracks[f'title_{k}'].sum()) for k in TITLE_FLAGS})


if __name__ == "__main__":
    main()

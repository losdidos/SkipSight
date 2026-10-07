"""Step 02 - One row per song, with its length measured from my own history.

Input : data/interim/plays.csv
Output: data/interim/tracks.csv

A play that ended with `trackdone` ran to the end, so its ms_played is about the song's length.
We take the median over those plays (the median ignores a rare odd play, e.g. one with a seek).
Songs I never finished get no length here.
Step 06 prefers Spotify's own length for the exact track id and falls back to this one
(scrubbing forward inside a song can make a `trackdone` play shorter than the song).
"""

from __future__ import annotations

import pandas as pd

from common import data_path


def main() -> None:
    plays = pd.read_csv(data_path("interim", "plays.csv"))
    finished = plays[plays["reason_end"] == "trackdone"]
    tracks = plays.groupby("track_id").agg(
        track_name=("track_name", "first"),
        artist_name=("artist_name", "first"),
        album_name=("album_name", "first"),
        n_plays=("play_id", "size"),
    )
    tracks["n_finished"] = finished.groupby("track_id").size()
    tracks["n_finished"] = tracks["n_finished"].fillna(0).astype(int)
    tracks["length_ms_history"] = finished.groupby("track_id")["ms_played"].median()
    tracks = tracks.reset_index().sort_values("n_plays", ascending=False)
    tracks.to_csv(data_path("interim", "tracks.csv"), index=False)
    print(f"{len(tracks):,} songs, length measured from my finished plays for {tracks['length_ms_history'].notna().mean():.0%}")


if __name__ == "__main__":
    main()

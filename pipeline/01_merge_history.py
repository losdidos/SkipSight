"""Step 01 - Merge the Spotify extended streaming history into one table.

Input : data/input/Streaming_History_Audio_*.json   (your own Spotify export)
Output: data/interim/plays.csv                       (one row per play, oldest first)

What happens:
- Only music plays are kept. Podcasts and audiobooks have no `spotify_track_uri`.
- Private fields (IP address, country, user agent) are dropped on purpose.
- The column names are shortened (master_metadata_track_name -> track_name, ...).
"""

from __future__ import annotations

import json

import pandas as pd

from common import data_path

RENAME = {
    "master_metadata_track_name": "track_name",
    "master_metadata_album_artist_name": "artist_name",
    "master_metadata_album_album_name": "album_name",
}
KEEP = [
    "ts", "platform", "ms_played", "track_id", "track_name", "artist_name", "album_name",
    "reason_start", "reason_end", "shuffle", "skipped", "offline", "incognito_mode",
]


def main() -> None:
    files = sorted(data_path("input").glob("Streaming_History_Audio_*.json"))
    if not files:
        raise SystemExit(
            f"No Streaming_History_Audio_*.json files found in {data_path('input')}.\n"
            "Request 'Extended streaming history' at spotify.com/account/privacy and put the files there."
        )

    raw = pd.DataFrame([item for path in files for item in json.loads(path.read_text(encoding="utf-8"))])
    music = raw[raw["spotify_track_uri"].notna()].rename(columns=RENAME)
    music["track_id"] = music["spotify_track_uri"].str.rsplit(":", n=1).str[-1]

    plays = music[KEEP].sort_values("ts", kind="stable").reset_index(drop=True)
    plays.insert(0, "play_id", range(len(plays)))
    plays.to_csv(data_path("interim", "plays.csv"), index=False)

    print(f"{len(files)} files -> {len(plays):,} music plays ({len(raw) - len(music):,} podcast/other rows dropped)")
    print(f"{plays['track_id'].nunique():,} different songs, {plays['artist_name'].nunique():,} artists")
    print(f"from {plays['ts'].min()} to {plays['ts'].max()}")


if __name__ == "__main__":
    main()

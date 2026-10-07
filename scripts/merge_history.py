"""Merge the Spotify extended streaming history JSON files into one CSV.

Keeps only music plays and the columns the pipeline needs. Private fields
(ip_addr, conn_country, offline timestamps) are dropped on purpose.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COLUMNS = [
    "ts", "platform", "ms_played", "spotify_track_uri", "track_name", "artist_name",
    "album_name", "reason_start", "reason_end", "shuffle", "skipped", "offline", "incognito_mode",
]
SOURCE_KEYS = {
    "track_name": "master_metadata_track_name",
    "artist_name": "master_metadata_album_artist_name",
    "album_name": "master_metadata_album_album_name",
}


def merge(input_dir: Path, output: Path) -> tuple[int, int]:
    rows, dropped = [], 0
    for path in sorted(input_dir.glob("Streaming_History_Audio_*.json")):
        for item in json.loads(path.read_text(encoding="utf-8")):
            uri = item.get("spotify_track_uri")
            if not uri:
                dropped += 1  # podcasts/audiobooks have no track URI
                continue
            row = {column: item.get(SOURCE_KEYS.get(column, column)) for column in COLUMNS}
            row["spotify_track_uri"] = uri.rsplit(":", 1)[-1]
            rows.append(row)
    rows.sort(key=lambda row: row["ts"])
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows), dropped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "spotify_history_full.csv")
    args = parser.parse_args()
    kept, dropped = merge(args.input_dir, args.output)
    print(f"Wrote {kept} track plays to {args.output} (skipped {dropped} non-track rows)")


if __name__ == "__main__":
    main()

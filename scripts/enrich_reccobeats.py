"""Map Spotify track IDs to audio features via ReccoBeats (unofficial, keyless API).

Only Spotify track IDs from a capped, deduplicated sample are sent. Tracks without
a result are written with empty feature values; nothing is guessed.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from itertools import islice
from pathlib import Path

API_URL = "https://api.reccobeats.com/v1/audio-features"
SEARCH_URL = "https://api.reccobeats.com/v1/track/search"
USER_AGENT = "SkipSight/0.1 (local feature enrichment test)"
BATCH_SIZE = 20
FEATURES = [
    "tempo",
    "energy",
    "danceability",
    "valence",
    "acousticness",
    "instrumentalness",
    "liveness",
    "speechiness",
    "loudness",
    "key",
    "mode",
]
OUTPUT_FIELDS = ["spotify_track_id", "found", "source", "isrc", *FEATURES]


def spotify_id(value: str) -> str:
    """Accept a bare ID, a spotify:track:<id> URI, or an open.spotify.com URL."""
    return value.strip().split("?")[0].rstrip("/").replace(":", "/").rsplit("/", 1)[-1]


def normalize(text: str) -> str:
    return re.sub(r"[^\w]", "", unicodedata.normalize("NFKD", text).casefold())


def clean_title(title: str) -> str:
    """Drop feature/remaster/version suffixes that differ between catalogues."""
    title = re.sub(r"\s*[\(\[][^\)\]]*(feat|ft\.|with|remaster|version|edit|mix|from)[^\)\]]*[\)\]]", "", title, flags=re.I)
    title = re.sub(r"\s+-\s+(\d{4}\s+)?(remaster|single|radio|live|mono|stereo|album|.*(mix|remix|version|edit)).*$", "", title, flags=re.I)
    return title.strip()


def unique_tracks(path: Path, scan_rows: int, count: int) -> list[dict]:
    seen: dict[str, dict] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in islice(csv.DictReader(handle), scan_rows):
            track_id = spotify_id(row.get("spotify_track_uri", ""))
            if track_id:
                seen.setdefault(
                    track_id,
                    {"id": track_id, "title": row.get("track_name", ""), "artist": row.get("artist_name", "")},
                )
    tracks = list(seen.values())
    if len(tracks) <= count:
        return tracks
    step = (len(tracks) - 1) / (count - 1)
    return [tracks[round(index * step)] for index in range(count)]


def get_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code == 429 and attempt < 2:
                time.sleep(5 * (attempt + 1))
                continue
            raise
    raise RuntimeError("unreachable")


def search_matches(title: str, artist: str) -> list[dict]:
    """Search by title only (artist in the query returns nothing), then filter by artist."""
    cleaned = clean_title(title)
    wanted_title, wanted_artist = normalize(cleaned), normalize(artist)
    matches: list[dict] = []
    for page in range(2):
        url = SEARCH_URL + "?" + urllib.parse.urlencode({"searchText": cleaned, "page": page})
        content = get_json(url).get("content", [])
        for item in content:
            names = normalize("".join(a.get("name", "") for a in item.get("artists", [])))
            if normalize(clean_title(item.get("trackTitle", ""))) == wanted_title and wanted_artist in names:
                matches.append(item)
        if len(content) < 25:
            break
        time.sleep(0.4)
    return matches


def search_candidate_ids(title: str, artist: str) -> list[str]:
    ids = (spotify_id(item.get("href", "")) for item in search_matches(title, artist))
    return [track_id for track_id in ids if track_id]


def find_duration_ms(track: dict) -> int | None:
    """Track length from search results; prefers the exact Spotify ID, else a title+artist match."""
    if not track["title"] or not track["artist"]:
        return None
    matches = [m for m in search_matches(track["title"], track["artist"]) if m.get("durationMs")]
    exact = [m for m in matches if spotify_id(m.get("href", "")) == track["id"]]
    chosen = exact or matches
    return int(chosen[0]["durationMs"]) if chosen else None


def fetch_batch(ids: list[str]) -> dict[str, dict]:
    url = API_URL + "?" + urllib.parse.urlencode({"ids": ",".join(ids)}, safe=",")
    found = {}
    for item in get_json(url).get("content", []):
        track_id = spotify_id(item.get("href", ""))
        if track_id:
            found[track_id] = item
    return found


def enrich(tracks: list[dict], name_fallback: bool = True, with_duration: bool = False) -> list[dict]:
    """Return one row per track: found/source/isrc plus audio features (empty if unmatched).

    With with_duration, also adds duration_ms (one extra title search per track).
    """
    ids = [track["id"] for track in tracks]
    by_id: dict[str, dict] = {}
    for start in range(0, len(ids), BATCH_SIZE):
        by_id.update(fetch_batch(ids[start : start + BATCH_SIZE]))
        time.sleep(0.5)

    # Fallback: the same song often has several Spotify IDs; find one ReccoBeats knows.
    by_name: dict[str, dict] = {}
    if name_fallback:
        for track in tracks:
            if track["id"] in by_id or not track["title"] or not track["artist"]:
                continue
            if re.search(r"remix", track["title"], flags=re.I):
                continue  # a remix's tempo can differ from the original's
            candidates = search_candidate_ids(track["title"], track["artist"])
            time.sleep(0.4)
            if candidates:
                features = fetch_batch(candidates[:BATCH_SIZE])
                if features:
                    by_name[track["id"]] = next(iter(features.values()))
                time.sleep(0.4)

    rows = []
    for track in tracks:
        item, source = by_id.get(track["id"]), "id"
        if item is None:
            item, source = by_name.get(track["id"]), "name"
        row = {"spotify_track_id": track["id"], "found": int(item is not None)}
        if item:
            row["source"] = source
            row["isrc"] = item.get("isrc", "")
            row.update({name: item.get(name, "") for name in FEATURES})
        if with_duration:
            row["duration_ms"] = find_duration_ms(track)
            time.sleep(0.4)
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("spotify_history.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/track_audio_features.csv"))
    parser.add_argument("--scan-rows", type=int, default=1000)
    parser.add_argument("--tracks", type=int, default=50)
    parser.add_argument(
        "--no-name-fallback",
        action="store_true",
        help="Do not send track titles/artists for a name search on ID misses.",
    )
    args = parser.parse_args()

    tracks = unique_tracks(args.input, args.scan_rows, args.tracks)
    rows = enrich(tracks, name_fallback=not args.no_name_fallback)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    by_id = sum(1 for row in rows if row.get("source") == "id")
    by_name = sum(1 for row in rows if row.get("source") == "name")
    print(f"Queried {total} unique tracks.")
    print(f"  matched by Spotify ID: {by_id} ({by_id / total:.0%})")
    print(f"  recovered by name search: {by_name}")
    print(f"  total with features: {by_id + by_name} ({(by_id + by_name) / total:.0%})")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()

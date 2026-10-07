"""Step 03 - Ask ReccoBeats about every song (audio features, ISRC, length, artists).

Input : data/interim/tracks.csv
Output: data/cache/reccobeats_track.jsonl     Spotify id -> track info (title, artists, durationMs, isrc)
        data/cache/reccobeats_features.jsonl  Spotify id -> audio features (tempo, energy, valence, ...)
        data/cache/reccobeats_search.jsonl    title -> search results (only for songs not found by id)

ReccoBeats is a free API that still serves the audio features Spotify removed from its own API.

1. Look up every song by its exact Spotify id. This is the safest lookup: no guessing involved.
2. The same song often exists under several Spotify ids (single, album, compilation).
   For songs not found by id, search by title and keep candidates with the same title + artist.
   Their features are fetched too. Step 06 decides strictly (length check) which one to trust.
"""

from __future__ import annotations

import pandas as pd

from apis import Cache, get_json, progress
from common import data_path
from matching import artist_ok, clean_title, title_ok

BASE = "https://api.reccobeats.com/v1"
BATCH = 40


def spotify_id(href) -> str:
    return str(href or "").rstrip("/").rsplit("/", 1)[-1]


def fetch_by_ids(endpoint: str, cache: Cache, ids: list[str]) -> None:
    todo = [i for i in dict.fromkeys(ids) if i not in cache]
    print(f"{endpoint}: {len(ids) - len(todo):,} cached, {len(todo):,} to fetch", flush=True)
    for start in range(0, len(todo), BATCH):
        batch = todo[start:start + BATCH]
        answer = get_json(f"{BASE}/{endpoint}", {"ids": ",".join(batch)}, pause=0.3) or {}
        found = {spotify_id(item.get("href")): item for item in answer.get("content", [])}
        for track_id in batch:
            cache.put(track_id, found.get(track_id))
        progress(endpoint, min(start + BATCH, len(todo)), len(todo), every=BATCH * 10)


def search(cache: Cache, title: str) -> list[dict]:
    query = clean_title(title)
    if query not in cache:
        results = []
        for page in range(2):
            answer = get_json(f"{BASE}/track/search", {"searchText": query, "page": page}, pause=0.3) or {}
            results += answer.get("content", [])
            if len(answer.get("content", [])) < 25:
                break
        cache.put(query, results)
    return cache.get(query)


def main() -> None:
    tracks = pd.read_csv(data_path("interim", "tracks.csv"))
    track_cache, feature_cache = Cache("reccobeats_track"), Cache("reccobeats_features")
    search_cache = Cache("reccobeats_search")

    fetch_by_ids("track", track_cache, tracks["track_id"].tolist())
    missing = tracks[[track_cache.get(i) is None for i in tracks["track_id"]]]
    print(f"found by Spotify id: {len(tracks) - len(missing):,}/{len(tracks):,}; searching by title for {len(missing):,}")

    candidate_ids = []
    for n, song in enumerate(missing.itertuples(), 1):
        for item in search(search_cache, song.track_name):
            names = [a.get("name") for a in item.get("artists", [])]
            if title_ok(song.track_name, item.get("trackTitle")) and artist_ok(song.artist_name, names):
                candidate_ids.append(spotify_id(item.get("href")))
        progress("search", n, len(missing))
    print(f"{len(candidate_ids):,} same-title-and-artist candidates found by search")

    fetch_by_ids("audio-features", feature_cache, tracks["track_id"].tolist() + candidate_ids)


if __name__ == "__main__":
    main()

"""Step 04 - Ask Deezer about every song (release date, BPM, genres, popularity, similar artists).

Input : data/interim/tracks.csv, data/cache/reccobeats_track.jsonl (for the ISRC)
Output: data/cache/deezer_*.jsonl

1. If ReccoBeats gave the song's ISRC (a worldwide id for one recording), look it up directly.
2. Otherwise (or if Deezer doesn't know the ISRC) search for artist + title, and fetch the
   candidates with the same title + artist. Step 06 checks their length.
3. For every Deezer song found: its album (genres, release date), its artist (number of fans)
   and the artist's related artists (for the artist-similarity analysis in Phase 2).
"""

from __future__ import annotations

import pandas as pd

from apis import Cache, get_json, progress
from common import data_path
from matching import artist_ok, length_ok, search_text, title_ok

BASE = "https://api.deezer.com"
PAUSE = 0.12  # Deezer allows 50 requests per 5 seconds
BULKY = {"available_countries", "tracks", "share", "md5_image", "contributors_raw"}


def slim(data):
    """Drop the big fields we never use, so the cache stays small."""
    if not isinstance(data, dict):
        return data
    return {k: v for k, v in data.items() if k not in BULKY and not k.startswith(("picture_", "cover_"))}


def cached_get(cache: Cache, key, url: str, params: dict | None = None):
    if key not in cache:
        cache.put(key, slim(get_json(url, params, pause=PAUSE)))
    return cache.get(key)


def main() -> None:
    tracks = pd.read_csv(data_path("interim", "tracks.csv"))
    reccobeats = Cache("reccobeats_track")
    by_isrc, searches, full = Cache("deezer_isrc"), Cache("deezer_search"), Cache("deezer_track")

    found_ids = set()
    for n, song in enumerate(tracks.itertuples(), 1):
        exact = reccobeats.get(song.track_id) or {}
        ref_ms = exact.get("durationMs") or song.length_ms_history
        hit = cached_get(by_isrc, exact["isrc"], f"{BASE}/track/isrc:{exact['isrc']}") if exact.get("isrc") else None
        if hit:
            found_ids.add(hit["id"])
            if hit["id"] not in full:
                full.put(hit["id"], hit)  # an ISRC answer is already the full track
            names = [hit["artist"]["name"]] + [c["name"] for c in hit.get("contributors", [])]
            if artist_ok(song.artist_name, names) and length_ok(ref_ms, hit["duration"] * 1000) is not False:
                continue  # good ISRC hit; otherwise also search, so step 06 has alternatives
        query = search_text(song.artist_name, song.track_name)
        answer = cached_get(searches, query, f"{BASE}/search", {"q": query, "limit": 10}) or {}
        candidates = [c for c in answer.get("data", [])
                      if title_ok(song.track_name, c["title"]) and artist_ok(song.artist_name, [c["artist"]["name"]])]
        for candidate in candidates[:3]:
            if cached_get(full, candidate["id"], f"{BASE}/track/{candidate['id']}"):
                found_ids.add(candidate["id"])
        progress("songs", n, len(tracks), every=250)

    found = [full.get(i) for i in found_ids if full.get(i)]
    album_ids = {t["album"]["id"] for t in found}
    artist_ids = {t["artist"]["id"] for t in found}
    print(f"{len(found):,} Deezer songs -> {len(album_ids):,} albums, {len(artist_ids):,} artists", flush=True)

    albums, artists, related = Cache("deezer_album"), Cache("deezer_artist"), Cache("deezer_related")
    for n, album_id in enumerate(album_ids, 1):
        cached_get(albums, album_id, f"{BASE}/album/{album_id}")
        progress("albums", n, len(album_ids), every=250)
    for n, artist_id in enumerate(artist_ids, 1):
        cached_get(artists, artist_id, f"{BASE}/artist/{artist_id}")
        cached_get(related, artist_id, f"{BASE}/artist/{artist_id}/related", {"limit": 20})
        progress("artists", n, len(artist_ids), every=250)


if __name__ == "__main__":
    main()

"""Step 05 - Ask MusicBrainz about every song by ISRC (first release date, exact artist, genres).

Input : ISRCs from data/cache/reccobeats_track.jsonl and data/cache/deezer_track.jsonl
Output: data/cache/musicbrainz_isrc.jsonl    ISRC -> recordings (title, length, first-release-date, artists)
        data/cache/musicbrainz_artist.jsonl  artist id -> genres and tags voted by MusicBrainz users

MusicBrainz is an open music encyclopedia. Looking up by ISRC gives the exact recording and
the exact artist id. The old pipeline searched artists by NAME, which can mix up two artists
with the same name.
`first-release-date` is the earliest release of that recording anywhere. It is the best
source for "when did this song come out" (a 2011 remaster still says the original year).

MusicBrainz allows 1 request per second, so this is the slowest step (it caches, so only once).
"""

from __future__ import annotations

from apis import Cache, get_json, progress

BASE = "https://musicbrainz.org/ws/2"
PAUSE = 1.1


def main() -> None:
    isrcs = {(v or {}).get("isrc") for v in Cache("reccobeats_track").data.values()}
    isrcs |= {(v or {}).get("isrc") for v in Cache("deezer_track").data.values()}
    isrcs = sorted(i for i in isrcs if i)

    by_isrc = Cache("musicbrainz_isrc")
    todo = [i for i in isrcs if i not in by_isrc]
    print(f"{len(isrcs):,} ISRCs, {len(todo):,} to fetch (~{len(todo) * PAUSE / 60:.0f} min)", flush=True)
    for n, isrc in enumerate(todo, 1):
        answer = get_json(f"{BASE}/isrc/{isrc}", {"fmt": "json", "inc": "artist-credits"}, pause=PAUSE)
        recordings = [
            {"id": r["id"], "title": r.get("title"), "length": r.get("length"),
             "first_release_date": r.get("first-release-date"),
             "artists": [{"id": c["artist"]["id"], "name": c["artist"]["name"]} for c in r.get("artist-credit", [])]}
            for r in (answer or {}).get("recordings", [])
        ]
        by_isrc.put(isrc, recordings or None)
        progress("isrc", n, len(todo))

    artist_ids = {rec["artists"][0]["id"] for recs in by_isrc.data.values() for rec in recs or [] if rec["artists"]}
    artists = Cache("musicbrainz_artist")
    todo = [a for a in sorted(artist_ids) if a not in artists]
    print(f"{len(artist_ids):,} artists, {len(todo):,} to fetch", flush=True)
    for n, artist_id in enumerate(todo, 1):
        a = get_json(f"{BASE}/artist/{artist_id}", {"fmt": "json", "inc": "genres+tags"}, pause=PAUSE)
        artists.put(artist_id, None if a is None else {
            "name": a.get("name"), "type": a.get("type"), "country": a.get("country"),
            "begin": (a.get("life-span") or {}).get("begin"),
            "genres": {g["name"]: g.get("count", 0) for g in a.get("genres", [])},
            "tags": {t["name"]: t.get("count", 0) for t in a.get("tags", [])},
        })
        progress("artists", n, len(todo))


if __name__ == "__main__":
    main()

"""Fetch song-level metadata ONCE per track from a keyless public API and cache it (resumable).

  python scripts/enrich_metadata.py --source deezer
  python scripts/enrich_metadata.py --source itunes
  python scripts/enrich_metadata.py --source musicbrainz

Scope = tracks that have ReccoBeats audio features OR >= MIN_PLAYS plays in the history.
Each source writes its own raw cache in data/raw/ (never re-requested for tracks already in it):
  deezer.csv       one row per track  (ISRC lookup, title+artist search fallback)
  itunes.csv       one row per track  (title+artist search)
  musicbrainz.csv  one row per ARTIST (artist search incl. genre tags)
Only track title/artist/ISRC are sent to these services.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import unicodedata
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
MIN_PLAYS = 10
UA = {"User-Agent": "SkipSight/0.2 (personal research project)"}
session = requests.Session()
session.headers.update(UA)


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def clean_title(title: str) -> str:
    title = re.sub(r"\s*[\(\[][^)\]]*(remix|remaster|version|edit|live|feat|with|from|slowed|sped|radio)[^)\]]*[\)\]]", "", title, flags=re.I)
    title = re.sub(r"\s+-\s+.*(remix|remaster|version|edit|radio|slowed|sped).*$", "", title, flags=re.I)
    return title.strip()


def get(url: str, params: dict | None = None, pause: float = 0.15):
    for attempt in range(5):
        try:
            r = session.get(url, params=params, timeout=30)
        except requests.RequestException:
            time.sleep(3 * (attempt + 1))
            continue
        if r.status_code in (429, 503):
            time.sleep(8 * (attempt + 1))
            continue
        time.sleep(pause)
        if r.status_code != 200:
            return None
        try:
            return r.json()
        except ValueError:
            return None
    return None


def scope_tracks() -> pd.DataFrame:
    h = pd.read_csv(ROOT / "spotify_history_full.csv")
    f = pd.read_csv(ROOT / "data/processed/track_features.csv").rename(columns={"spotify_track_id": "uri"})
    first = h.drop_duplicates("spotify_track_uri").set_index("spotify_track_uri")[["track_name", "artist_name", "album_name"]]
    plays = h.groupby("spotify_track_uri").size().rename("plays")
    t = f[["uri", "found", "isrc", "duration_ms"]].drop_duplicates("uri").set_index("uri").join(first).join(plays)
    t = t[(t.found == 1) | (t.plays >= MIN_PLAYS)]
    return t.reset_index().sort_values("plays", ascending=False).reset_index(drop=True)


def run(path: Path, items: list[dict], fetch, key: str, flush_every: int = 25) -> None:
    done = pd.read_csv(path, dtype=str) if path.exists() else pd.DataFrame()
    seen = set(done[key]) if len(done) else set()
    todo = [i for i in items if i[key] not in seen]
    print(f"{path.name}: {len(seen)} cached, {len(todo)} to fetch", flush=True)
    rows = []
    for n, item in enumerate(todo, 1):
        row = {key: item[key]}
        try:
            row.update(fetch(item) or {})
        except Exception as exc:  # keep the run alive; the track is retried next run if no row is written
            print("error", item[key], exc, flush=True)
            continue
        rows.append(row)
        if n % flush_every == 0 or n == len(todo):
            out = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
            out.to_csv(path, index=False)
            done, rows = out, []
            print(f"  {path.name} {len(seen) + n}/{len(seen) + len(todo)}", flush=True)
    print("finished", path.name, flush=True)


# ---------------------------------------------------------------- Deezer
_album, _artist = {}, {}


def deezer_fetch(t: dict) -> dict:
    track = None
    if isinstance(t.get("isrc"), str) and t["isrc"]:
        data = get(f"https://api.deezer.com/track/isrc:{t['isrc']}", pause=0.12)
        if data and "id" in data:
            track, how = data, "isrc"
    if track is None:
        title, artist = clean_title(t["track_name"]), t["artist_name"]
        res = get("https://api.deezer.com/search", {"q": f'artist:"{artist}" track:"{title}"', "limit": 10}, pause=0.12)
        for cand in (res or {}).get("data", []):
            if norm(cand["artist"]["name"]) == norm(artist) and norm(clean_title(cand["title"])) == norm(title):
                track, how = get(f"https://api.deezer.com/track/{cand['id']}", pause=0.12), "search"
                break
    if not track or "id" not in track:
        return {"dz_found": 0}
    alb_id, art_id = track["album"]["id"], track["artist"]["id"]
    if alb_id not in _album:
        _album[alb_id] = get(f"https://api.deezer.com/album/{alb_id}", pause=0.12) or {}
    if art_id not in _artist:
        _artist[art_id] = get(f"https://api.deezer.com/artist/{art_id}", pause=0.12) or {}
    alb, art = _album[alb_id], _artist[art_id]
    genres = [g["name"] for g in alb.get("genres", {}).get("data", [])]
    return {
        "dz_found": 1, "dz_match": how, "dz_track_id": track["id"], "dz_artist_id": art_id, "dz_album_id": alb_id,
        "dz_title": track.get("title"), "dz_artist": track["artist"]["name"],
        "dz_rank": track.get("rank"), "dz_release_date": track.get("release_date"),
        "dz_bpm": track.get("bpm"), "dz_gain": track.get("gain"),
        "dz_explicit": int(bool(track.get("explicit_lyrics"))),
        "dz_n_contributors": len(track.get("contributors", [])),
        "dz_track_position": track.get("track_position"), "dz_disk_number": track.get("disk_number"),
        "dz_duration_s": track.get("duration"),
        "dz_album_title": alb.get("title"), "dz_album_type": alb.get("record_type"),
        "dz_album_nb_tracks": alb.get("nb_tracks"), "dz_album_release_date": alb.get("release_date"),
        "dz_album_label": alb.get("label"), "dz_album_genres": "|".join(genres),
        "dz_artist_fans": art.get("nb_fan"), "dz_artist_nb_album": art.get("nb_album"),
    }


# ---------------------------------------------------------------- iTunes
def itunes_fetch(t: dict) -> dict:
    title, artist = clean_title(t["track_name"]), t["artist_name"]
    res = get("https://itunes.apple.com/search", {"term": f"{title} {artist}", "entity": "song", "limit": 10}, pause=1.2)
    for cand in (res or {}).get("results", []):
        if norm(artist.split(",")[0]) in norm(cand.get("artistName", "")) and norm(clean_title(cand.get("trackName", ""))) == norm(title):
            return {
                "it_found": 1, "it_genre": cand.get("primaryGenreName"), "it_release_date": cand.get("releaseDate", "")[:10],
                "it_explicit": int(cand.get("trackExplicitness") == "explicit"), "it_track_number": cand.get("trackNumber"),
                "it_track_count": cand.get("trackCount"), "it_disc_number": cand.get("discNumber"),
                "it_collection": cand.get("collectionName"), "it_duration_ms": cand.get("trackTimeMillis"),
                "it_price": cand.get("trackPrice"),
            }
    return {"it_found": 0}


# ---------------------------------------------------------------- MusicBrainz (per artist)
def musicbrainz_fetch(a: dict) -> dict:
    name = a["artist_name"]
    res = get("https://musicbrainz.org/ws/2/artist", {"query": f'artist:"{name}"', "limit": 5, "fmt": "json"}, pause=1.1)
    for cand in (res or {}).get("artists", []):
        if norm(cand.get("name", "")) == norm(name) and int(cand.get("score", 0)) >= 90:
            tags = sorted(cand.get("tags", []), key=lambda x: -x.get("count", 0))[:8]
            genres = sorted(cand.get("genres", []), key=lambda x: -x.get("count", 0))[:8]
            return {
                "mb_found": 1, "mb_id": cand["id"], "mb_type": cand.get("type"), "mb_gender": cand.get("gender"),
                "mb_country": cand.get("country"), "mb_area": (cand.get("area") or {}).get("name"),
                "mb_begin": (cand.get("life-span") or {}).get("begin"),
                "mb_genres": "|".join(f"{g['name']}:{g.get('count', 0)}" for g in genres),
                "mb_tags": "|".join(f"{g['name']}:{g.get('count', 0)}" for g in tags),
            }
    return {"mb_found": 0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["deezer", "itunes", "musicbrainz"], required=True)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    tracks = scope_tracks()
    if args.source == "musicbrainz":
        artists = tracks.groupby("artist_name").plays.sum().sort_values(ascending=False).reset_index()
        items = artists.to_dict("records")[: args.limit]
        run(RAW / "musicbrainz.csv", items, musicbrainz_fetch, "artist_name")
    else:
        items = tracks.to_dict("records")[: args.limit]
        fetch = deezer_fetch if args.source == "deezer" else itunes_fetch
        run(RAW / f"{args.source}.csv", items, fetch, "uri")


if __name__ == "__main__":
    sys.exit(main())

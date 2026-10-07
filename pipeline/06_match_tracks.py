"""Step 06 - Decide, for every song, which API results really are that song. No internet needed.

Input : data/interim/tracks.csv + the raw API answers in data/cache/
Output: data/interim/tracks_enriched.csv  (one row per song: chosen API data + how it was matched)

THE REFERENCE LENGTH
  1st choice: Spotify's own length for this exact track id (ReccoBeats lookup by id).
  2nd choice: the length measured from my own finished plays (step 02).
  Why not the history first: it matches Spotify to the second for 91 % of songs, but when it
  disagrees it is almost always too SHORT (scrubbing forward still ends as `trackdone`).
  Without any reference length, search results can't be verified, so they are rejected.

ACCEPT RULES (rules in matching.py)
  Lookup by exact id           (ReccoBeats by Spotify id)  -> accepted; the length is only reported
  Lookup by ISRC               (Deezer, MusicBrainz)       -> artist must match, length may not disagree
  Search by title              (ReccoBeats, Deezer)        -> title + artist must match AND length within 5 s
  When several candidates pass, the one closest in length wins.
  The `*_match` columns record how each source was matched, or why it was rejected.

RELEASE DATE (Phase 1.4)
  The earliest date among: MusicBrainz first-release-date, Deezer track date, Deezer album date.
  Remasters, deluxe editions and compilations carry later dates, so the earliest is the original.
"""

from __future__ import annotations

import pandas as pd

from apis import Cache
from common import data_path
from matching import artist_ok, clean_title, length_ok, search_text, title_ok

FEATURES = ["tempo", "energy", "danceability", "valence", "acousticness", "instrumentalness",
            "liveness", "speechiness", "loudness", "key", "mode"]

RB_TRACK, RB_FEATURES, RB_SEARCH = Cache("reccobeats_track"), Cache("reccobeats_features"), Cache("reccobeats_search")
DZ_ISRC, DZ_SEARCH, DZ_TRACK = Cache("deezer_isrc"), Cache("deezer_search"), Cache("deezer_track")
DZ_ALBUM, DZ_ARTIST = Cache("deezer_album"), Cache("deezer_artist")
MB_ISRC, MB_ARTIST = Cache("musicbrainz_isrc"), Cache("musicbrainz_artist")


def spotify_id(href) -> str:
    return str(href or "").rstrip("/").rsplit("/", 1)[-1]


def pick(song, candidates: list[dict], ref_ms, length_key, scale=1) -> tuple[dict | None, str]:
    """Search results: keep title + artist + length matches; return (best, reason)."""
    same = [c for c in candidates if title_ok(song.track_name, c["title"]) and artist_ok(song.artist_name, c["artists"])]
    if not same:
        return None, "not_found"
    if ref_ms != ref_ms:  # NaN: no reference length
        return None, "rejected:no_length_to_check"
    fits = [c for c in same if length_ok(ref_ms, (c[length_key] or 0) * scale)]
    if not fits:
        return None, "rejected:length"
    return min(fits, key=lambda c: abs(c[length_key] * scale - ref_ms)), "search"


def match_reccobeats(song, row: dict) -> None:
    exact = RB_TRACK.get(song.track_id)
    if exact:
        row.update(rb_match="id", rb_id=song.track_id, rb_isrc=exact.get("isrc"), rb_length_ms=exact.get("durationMs"))
        row["rb_length_agrees"] = length_ok(song.length_ms_history, exact.get("durationMs"))
    else:
        items = [{"title": i.get("trackTitle"), "artists": [a.get("name") for a in i.get("artists", [])],
                  "length": i.get("durationMs"), "id": spotify_id(i.get("href")), "isrc": i.get("isrc")}
                 for i in RB_SEARCH.get(clean_title(song.track_name)) or []]
        items = [i for i in items if RB_FEATURES.get(i["id"])]  # only useful if it has features
        best, reason = pick(song, items, song.length_ms_history, "length")
        row["rb_match"] = reason
        if best:
            row.update(rb_id=best["id"], rb_isrc=best["isrc"], rb_length_ms=best["length"])
    features = RB_FEATURES.get(row.get("rb_id")) or {}
    row["has_features"] = bool(features)
    row.update({f: features.get(f) for f in FEATURES})


def match_deezer(song, row: dict, ref_ms) -> None:
    track, how = None, "not_found"
    hit = DZ_ISRC.get(row.get("rb_isrc")) if row.get("rb_isrc") else None
    if hit:
        names = [hit["artist"]["name"]] + [c["name"] for c in hit.get("contributors", [])]
        same_length = length_ok(ref_ms, hit["duration"] * 1000)
        if same_length is False:
            how = "rejected:isrc_length"
        elif artist_ok(song.artist_name, names):
            track, how = hit, "isrc"
        elif same_length and title_ok(song.track_name, hit["title"]):
            # Same recording code + same title + same length: the same recording, credited
            # under another artist name (alias, label upload). Counted separately to keep an eye on it.
            track, how = hit, "isrc_alias"
        else:
            how = "rejected:isrc_artist"
    if track is None:
        query = search_text(song.artist_name, song.track_name)
        found = [DZ_TRACK.get(c["id"]) for c in (DZ_SEARCH.get(query) or {}).get("data", [])]
        items = [{"title": t["title"], "artists": [t["artist"]["name"]] + [c["name"] for c in t.get("contributors", [])],
                  "duration": t["duration"], "track": t}
                 for t in found if t]
        best, reason = pick(song, items, ref_ms, "duration", scale=1000)
        if best:
            track, how = best["track"], "search"
        elif how == "not_found":
            how = reason
    row["dz_match"] = how
    if track is None:
        return
    album = DZ_ALBUM.get(track["album"]["id"]) or {}
    artist = DZ_ARTIST.get(track["artist"]["id"]) or {}
    row.update(
        dz_id=track["id"], dz_isrc=track.get("isrc"), dz_length_ms=track["duration"] * 1000,
        dz_bpm=track.get("bpm") or None, dz_gain=track.get("gain"), dz_rank=track.get("rank"),
        dz_explicit=track.get("explicit_lyrics"), dz_release_date=track.get("release_date"),
        dz_artist_id=track["artist"]["id"], dz_artist_fans=artist.get("nb_fan"),
        dz_album_type=album.get("record_type"), dz_album_release_date=album.get("release_date"),
        dz_album_genres="|".join(g["name"] for g in (album.get("genres") or {}).get("data", [])),
    )


def match_musicbrainz(song, row: dict, ref_ms) -> None:
    recordings, rejected = [], False
    for isrc in {row.get("rb_isrc"), row.get("dz_isrc")} - {None}:
        for rec in MB_ISRC.get(isrc) or []:
            if artist_ok(song.artist_name, [a["name"] for a in rec["artists"]]) and length_ok(ref_ms, rec["length"]) is not False:
                recordings.append(rec)
            else:
                rejected = True
    if not recordings:
        row["mb_match"] = "rejected:artist_or_length" if rejected else "not_found"
        return
    dates = [r["first_release_date"] for r in recordings if r["first_release_date"]]
    main_artist = recordings[0]["artists"][0]
    artist = MB_ARTIST.get(main_artist["id"]) or {}
    top = lambda votes: "|".join(sorted(votes, key=votes.get, reverse=True)[:8])
    row.update(
        mb_match="isrc", mb_first_release_date=min(dates) if dates else None, mb_artist_id=main_artist["id"],
        mb_artist_type=artist.get("type"), mb_artist_country=artist.get("country"),
        mb_genres=top(artist.get("genres") or {}), mb_tags=top(artist.get("tags") or {}),
    )


def earliest_date(row: dict) -> tuple[str | None, str | None]:
    options = {"musicbrainz": row.get("mb_first_release_date"), "deezer_track": row.get("dz_release_date"),
               "deezer_album": row.get("dz_album_release_date")}
    valid = {k: str(v) for k, v in options.items() if v and str(v)[:4].isdigit() and int(str(v)[:4]) >= 1900}
    if not valid:
        return None, None
    source = min(valid, key=valid.get)
    return valid[source], source


def report(df: pd.DataFrame) -> None:
    n = len(df)
    print(f"{n:,} songs\n")
    for col in ["rb_match", "dz_match", "mb_match", "length_source", "release_date_source"]:
        print(df[col].fillna("-").value_counts().to_string(), "\n")
    print(f"audio features: {df['has_features'].mean():.1%} of songs")
    agree = df["rb_length_agrees"].dropna()
    print(f"ReccoBeats length (exact id) agrees with my measured length (±5 s): {agree.mean():.1%} of {len(agree):,}")
    print(f"release year known: {df['release_year'].notna().mean():.1%}")


def main() -> None:
    tracks = pd.read_csv(data_path("interim", "tracks.csv"))
    rows = []
    for song in tracks.itertuples():
        row = song._asdict()
        row.pop("Index")
        match_reccobeats(song, row)
        exact_length = row.get("rb_length_ms") if row.get("rb_match") == "id" else None
        ref_ms = exact_length if exact_length is not None else song.length_ms_history
        match_deezer(song, row, ref_ms)
        match_musicbrainz(song, row, ref_ms)

        lengths = [("spotify_id", exact_length), ("history", song.length_ms_history),
                   ("reccobeats_search", row.get("rb_length_ms")), ("deezer", row.get("dz_length_ms"))]
        row["length_source"], row["length_ms"] = next(((s, v) for s, v in lengths if v == v and v is not None), (None, None))
        row["release_date"], row["release_date_source"] = earliest_date(row)
        rows.append(row)

    df = pd.DataFrame(rows)
    df["release_year"] = pd.to_numeric(df["release_date"].str[:4], errors="coerce").astype("Int64")
    df.to_csv(data_path("interim", "tracks_enriched.csv"), index=False)
    report(df)


if __name__ == "__main__":
    main()

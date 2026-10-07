"""Build the final ML-ready tables from history + all cached song metadata.

Outputs in data/processed/:
  tracks.csv            one row per song: static features + aggregated play counters (use for training / lookup)
  artists.csv           per-artist aggregates
  genres.csv            per-genre aggregates
  skipsight_dataset.csv one row per play: static song columns + ms_played + targets (+ ctx_ columns that
                        are NOT available when scoring a never-played song)

Leakage rule: no artist/genre skip rate is stored on play rows (they include the play itself). Training
rebuilds them out-of-fold from the counters in tracks/artists/genres.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
MIN_PLAYS = 10
SKIP_BLOCKS = [(0, 3), (3, 10), (10, 30), (30, 10**6)]  # seconds into the song
GENRE_BUCKETS = {
    "pop": ["pop"], "rap_hiphop": ["rap", "hip hop", "hip-hop", "trap", "grime"], "rnb_soul": ["r&b", "rnb", "soul", "funk"],
    "rock": ["rock", "grunge", "punk", "metal", "emo"],
    "electronic": ["electro", "house", "techno", "edm", "trance", "dance", "dubstep", "hardstyle", "bass", "drum"],
    "country": ["country", "bluegrass", "americana"], "latin": ["latin", "reggaeton", "salsa", "bachata"],
    "reggae_dancehall": ["reggae", "dancehall", "ska"], "jazz_blues": ["jazz", "blues"],
    "classical": ["classical", "orchestra", "score", "soundtrack"], "indie_alt": ["indie", "alternative"],
    "folk_acoustic": ["folk", "acoustic", "singer-songwriter"], "kpop_asian": ["k-pop", "kpop", "j-pop", "asian"],
    "dutch_local": ["nederland", "dutch", "vlaams", "flemish", "belgian"], "afro": ["afro", "african"],
}


def load_history() -> pd.DataFrame:
    h = pd.read_csv(ROOT / "spotify_history_full.csv").rename(columns={"spotify_track_uri": "uri"})
    h["ts"] = pd.to_datetime(h["ts"], utc=True)
    h["start_ts"] = h["ts"] - pd.to_timedelta(h["ms_played"], unit="ms")  # Spotify ts is the END of the play
    h["skip"] = (h["reason_end"] == "fwdbtn").astype(int)
    h["finished"] = (h["reason_end"] == "trackdone").astype(int)
    h["end_kind"] = np.where(h["skip"] == 1, "skip", np.where(h["finished"] == 1, "finished", "other"))
    return h.sort_values("start_ts").reset_index(drop=True)


def read_optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def bucket_genres(text: str) -> dict:
    text = text.lower()
    return {f"g_{k}": int(any(w in text for w in words)) for k, words in GENRE_BUCKETS.items()}


def title_flags(title: str) -> dict:
    t = title.lower()
    return {
        "t_remix": int("remix" in t), "t_slowed": int("slowed" in t), "t_sped": int("sped up" in t or "speed up" in t),
        "t_feat": int(bool(re.search(r"feat\.|ft\.|\(with ", t))), "t_live": int("live" in t), "t_acoustic": int("acoustic" in t),
        "t_version": int("version" in t or "edit" in t or "remaster" in t), "t_len_chars": len(title), "t_n_words": len(title.split()),
        "t_has_paren": int("(" in title or " - " in title),
    }


def main() -> None:
    h = load_history()
    feats = pd.read_csv(PROC / "track_features.csv").rename(
        columns={"spotify_track_id": "uri", "found": "rb_found", "duration_ms": "rb_duration_ms"})
    feats = feats.drop(columns=["source"], errors="ignore").drop_duplicates("uri")
    dz, it = read_optional(RAW / "deezer.csv"), read_optional(RAW / "itunes.csv")
    mb, pv = read_optional(RAW / "musicbrainz.csv"), read_optional(PROC / "track_audio_preview.csv")

    first = h.drop_duplicates("uri").set_index("uri")[["track_name", "artist_name", "album_name"]]
    agg = h.groupby("uri").agg(
        plays=("skip", "size"), skips=("skip", "sum"), finishes=("finished", "sum"), ms_played_total=("ms_played", "sum"),
        first_play=("start_ts", "min"), last_play=("start_ts", "max"))
    # real track length: longest fully-finished play, else API durations
    full_len = h[h.finished == 1].groupby("uri").ms_played.max().rename("len_from_play")
    t = first.join(agg).join(full_len).join(feats.set_index("uri"))
    for d in (dz, it):
        if len(d):
            t = t.join(d.set_index("uri"))
    t = t.reset_index()
    for c in ["dz_found", "it_found", "rb_found"]:
        if c not in t:
            t[c] = 0
        t[c] = t[c].fillna(0).astype(int)

    # scope: has audio features, or enough plays to be worth keeping
    t = t[(t.rb_found == 1) | (t.plays >= MIN_PLAYS)].copy()

    nan = pd.Series(np.nan, index=t.index)
    dz_len = t["dz_duration_s"] * 1000 if "dz_duration_s" in t else nan
    it_len = t["it_duration_ms"] if "it_duration_ms" in t else nan
    t["duration_ms"] = t["len_from_play"].fillna(t["rb_duration_ms"]).fillna(dz_len).fillna(it_len)
    t["duration_source"] = np.select(
        [t.len_from_play.notna(), t.rb_duration_ms.notna(), dz_len.notna()], ["full_play", "reccobeats", "deezer"], "itunes")
    t = t[t.duration_ms.notna()].copy()

    rel = pd.to_datetime(t.get("dz_release_date"), errors="coerce").fillna(pd.to_datetime(t.get("it_release_date"), errors="coerce"))
    t["release_year"] = rel.dt.year
    t["release_decade"] = rel.dt.year // 10 * 10
    t["track_age_years"] = 2025 - t["release_year"]

    if len(mb):
        t = t.merge(mb.drop_duplicates("artist_name"), on="artist_name", how="left")
        t["mb_found"] = t["mb_found"].fillna(0).astype(int)
        t["artist_begin_year"] = pd.to_datetime(t["mb_begin"], errors="coerce").dt.year
    for col in ["dz_album_genres", "it_genre", "mb_tags", "mb_genres"]:
        if col not in t:
            t[col] = ""
        t[col] = t[col].fillna("")
    mb_names = t["mb_tags"].str.replace(r":\d+", "", regex=True) + "|" + t["mb_genres"].str.replace(r":\d+", "", regex=True)
    t["genre_text"] = (t["dz_album_genres"] + "|" + t["it_genre"] + "|" + mb_names).str.strip("|")
    t["genre_primary"] = t["it_genre"].where(t["it_genre"] != "", t["dz_album_genres"].str.split("|").str[0])
    t["genre_primary"] = t["genre_primary"].replace("", "unknown").fillna("unknown")
    t = pd.concat([t, pd.DataFrame([bucket_genres(x) for x in t.genre_text], index=t.index)], axis=1)
    t = pd.concat([t, pd.DataFrame([title_flags(x) for x in t.track_name], index=t.index)], axis=1)

    if "dz_album_type" in t:
        for k in ["single", "album", "ep", "compile"]:
            t[f"album_is_{k}"] = (t["dz_album_type"].fillna("") == k).astype(int)
    if "dz_bpm" in t:
        t["dz_bpm"] = t["dz_bpm"].replace(0, np.nan)  # Deezer uses 0 for "unknown"
    if "dz_artist_fans" in t:
        t["log_artist_fans"] = np.log1p(t["dz_artist_fans"])
    if "dz_rank" in t:
        t["log_track_rank"] = np.log1p(t["dz_rank"])
    t["explicit"] = t["dz_explicit"].fillna(t["it_explicit"]) if "dz_explicit" in t and "it_explicit" in t else nan
    if len(pv):
        t = t.merge(pv, on="uri", how="left")
        t["pv_ok"] = t["pv_ok"].fillna(0).astype(int)

    # per-track skip-time counters (how the skips are spread through the song)
    h = h[h.uri.isin(t.uri)].merge(t[["uri", "duration_ms"]], on="uri")
    h["pct_played"] = (h.ms_played / h.duration_ms).clip(0, 1)
    sk = h[h.skip == 1]
    for lo, hi in SKIP_BLOCKS:
        c = sk[(sk.ms_played >= lo * 1000) & (sk.ms_played < hi * 1000)].groupby("uri").size()
        t[f"skips_{lo}_{hi if hi < 10**6 else 'end'}s"] = t.uri.map(c).fillna(0).astype(int)
    for b in range(10):
        upper = (b + 1) / 10 if b < 9 else 1.01
        c = sk[(sk.pct_played >= b / 10) & (sk.pct_played < upper)].groupby("uri").size()
        t[f"skips_pct_{b * 10}"] = t.uri.map(c).fillna(0).astype(int)
    t["skip_rate"] = t.skips / t.plays
    t["mean_pct_played"] = t.uri.map(h.groupby("uri").pct_played.mean())
    t["median_skip_ms"] = t.uri.map(sk.groupby("uri").ms_played.median())

    def summarise(df: pd.DataFrame, key: str) -> pd.DataFrame:
        g = df.groupby(key).agg(tracks=("uri", "nunique"), plays=("plays", "sum"), skips=("skips", "sum"), finishes=("finishes", "sum"))
        g["skip_rate"] = g.skips / g.plays
        return g.sort_values("plays", ascending=False).reset_index()

    summarise(t, "artist_name").to_csv(PROC / "artists.csv", index=False)
    summarise(t, "genre_primary").to_csv(PROC / "genres.csv", index=False)

    # per-play table: static song columns only (identical for every play of a track)
    h = h.sort_values("start_ts")
    local = h.start_ts.dt.tz_convert("Europe/Brussels")
    h["ctx_prev_skipped"] = h.skip.shift(1).fillna(0).astype(int)
    h["ctx_hour"], h["ctx_weekday"] = local.dt.hour, local.dt.weekday
    h["ctx_gap_s"] = h.start_ts.diff().dt.total_seconds().clip(upper=3600)
    h["ctx_shuffle"] = h["shuffle"].astype(int)
    keep = ["uri", "start_ts", "ms_played", "pct_played", "skip", "finished", "end_kind", "reason_start", "reason_end",
            "ctx_prev_skipped", "ctx_hour", "ctx_weekday", "ctx_gap_s", "ctx_shuffle"]
    # whole-history counters would include the play itself -> keep them out of the per-play table
    counters = {"plays", "skips", "finishes", "ms_played_total", "skip_rate", "mean_pct_played", "median_skip_ms", "len_from_play",
                "first_play", "last_play"}
    static = t[[c for c in t if c not in counters and not c.startswith("skips_")]]
    plays = h[keep].merge(static, on="uri", how="left").rename(columns={"skip": "skipped"})

    t.to_csv(PROC / "tracks.csv", index=False)
    plays.to_csv(PROC / "skipsight_dataset.csv", index=False)

    # one big file: every play + all song features + whole-history counters (hist_*; these include the play itself)
    hist = t[["uri"] + [c for c in t if c in counters or c.startswith("skips_")]].add_prefix("hist_").rename(columns={"hist_uri": "uri"})
    full = plays.merge(hist, on="uri", how="left")
    full.to_csv(PROC / "skipsight_full.csv", index=False)
    print(f"skipsight_full.csv {full.shape}")
    print(f"tracks.csv {t.shape}  skipsight_dataset.csv {plays.shape}  skip rate {plays.skipped.mean():.3f}")


if __name__ == "__main__":
    main()

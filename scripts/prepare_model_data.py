"""Prepare the two model-ready tables (no model training).

data/processed/model_vibe.csv    one row per track: vibe_score target + content-only features
data/processed/model_timing.csv  one row per play: time-to-skip target + content-only features

Only features obtainable for a never-played song are used. No history counters, no context
(prev track, hour, position). `fold` groups by artist so CV mimics unseen songs.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
K = 5  # shrinkage strength for per-track rates

tr = pd.read_csv(P / "tracks.csv")
sel = pd.read_csv(ROOT / "reports" / "feature_selection.csv")
pl = pd.read_csv(P / "skipsight_dataset.csv", usecols=["spotify_track_uri", "reason_end", "ms_played", "duration_ms"])

feats = [f for f in sel[sel.shortlist].feature if f in tr]
ids = ["uri", "track_name", "artist_name"]

# ---- fold: artist-grouped, same assignment for both tables ----
rng = np.random.default_rng(42)
artists = tr.artist_name.unique()
fold_of = dict(zip(artists, rng.integers(0, 5, len(artists))))
tr["fold"] = tr.artist_name.map(fold_of)

# ---- model 1: vibe score (0-100), built from how the track was treated in the history ----
mu_keep = 1 - tr.skips.sum() / tr.plays.sum()
mu_pct = np.average(tr.mean_pct_played, weights=tr.plays)
KA = 5  # artist prior strength


def pooled(own_sum, own_n, num, den, global_mean):
    """shrink a track's rate toward its artist's rate (excluding itself), which is shrunk toward the global rate"""
    a_num = own_sum.groupby(tr.artist_name).transform("sum") - own_sum
    a_den = own_n.groupby(tr.artist_name).transform("sum") - own_n
    prior = (a_num + KA * global_mean) / (a_den + KA)
    return (num + K * prior) / (den + K)


keep = pooled(tr.plays - tr.skips, tr.plays, tr.plays - tr.skips, tr.plays, mu_keep)  # share of plays not skipped
pct = pooled(tr.mean_pct_played * tr.plays, tr.plays, tr.mean_pct_played * tr.plays, tr.plays, mu_pct)  # share listened
artist_plays = tr.groupby("artist_name").plays.transform("sum")
replay = 0.5 * np.log1p(tr.plays) + 0.5 * np.log1p(artist_plays)  # replays of the track and of the artist signal liking
rank = lambda s: s.rank(pct=True)
artist_fin = tr.groupby("artist_name").finishes.transform("sum")
finished = 0.5 * np.log1p(tr.finishes) + 0.5 * np.log1p(artist_fin)  # full listens of the track and of the artist
tr["vibe_score"] = (100 * (0.2 * rank(keep) + 0.15 * rank(pct) + 0.35 * rank(replay) + 0.3 * rank(finished))).round(1)
tr["vibe_keep_rate"], tr["vibe_pct_listened"], tr["vibe_replays"], tr["vibe_finishes"] = keep.round(4), pct.round(4), tr.plays, tr.finishes
vibe = tr[ids + ["fold", "vibe_score", "vibe_keep_rate", "vibe_pct_listened", "vibe_replays", "vibe_finishes", "plays"] + feats]
vibe.to_csv(P / "model_vibe.csv", index=False)

# ---- model 2: skip timing (time-to-event; non-skips are censored at the time played) ----
d = pl.merge(tr[ids + ["fold", "vibe_score", "dz_duration_s"] + [f for f in feats if f != "dz_duration_s"]],
             left_on="spotify_track_uri", right_on="uri")
d["end_kind"] = np.select([d.reason_end == "fwdbtn", d.reason_end == "trackdone"], ["skip", "finished"], "other")
d["skipped"] = (d.end_kind == "skip").astype(int)
d["sec_played"] = d.ms_played / 1000
d["track_len_s"] = d.dz_duration_s.fillna(d.duration_ms / 1000)
d["pct_played"] = (d.sec_played / d.track_len_s).clip(0, 1).round(4)
d = d[d.sec_played > 0]
cols = ids + ["fold", "skipped", "end_kind", "sec_played", "track_len_s", "pct_played", "vibe_score"] + feats
timing = d[cols].rename(columns={"vibe_score": "vibe_score_in_sample"})
timing.to_csv(P / "model_timing.csv", index=False)

print("model_vibe.csv  ", vibe.shape, "| vibe_score mean %.1f, std %.1f" % (vibe.vibe_score.mean(), vibe.vibe_score.std()))
print("model_timing.csv", timing.shape, "| skips", int(timing.skipped.sum()), "| folds", sorted(timing.fold.unique()))
print("features:", len(feats), "| missing share >10%:", [f for f in feats if vibe[f].isna().mean() > .1])
print(vibe[["vibe_score", "vibe_keep_rate", "vibe_pct_listened", "vibe_replays", "vibe_finishes"]].corr(method="spearman").round(2))




"""Vibe score v1 vs v2, tested on a time split. Writes reports/vibe_score_comparison.md.
A = first 50% of plays (time), B = next 25%, C = last 25%.
v2 weights are fit on B using features from A; both scores are tested on C using features from A+B.
Test = does the score (from earlier plays) rank which later plays get skipped? (tracks seen earlier only)"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
pl = pd.read_csv(ROOT / "data/processed/skipsight_dataset.csv",
                 usecols=["ts", "spotify_track_uri", "artist_name", "reason_start", "ms_played", "pct_played", "skipped"])
pl["ts"] = pd.to_datetime(pl.ts, format="ISO8601", utc=True)
pl = pl.sort_values("ts").reset_index(drop=True)
pl["pct"] = pl.pct_played.clip(0, 1).fillna(0)
pl["chosen"] = pl.reason_start.isin(["clickrow", "playbtn"]).astype(int)
n = len(pl)
A, B, C = pl.iloc[: n // 2], pl.iloc[n // 2: int(n * .75)], pl.iloc[int(n * .75):]
K, KA = 3, 5


def feats(h):
    """per-track features from history h, using only that history"""
    g = h.groupby("spotify_track_uri")
    t = pd.DataFrame({"plays": g.size(), "skips": g.skipped.sum(), "pct": g.pct.mean(), "chosen": g.chosen.mean(),
                      "artist": g.artist_name.first()})
    span = (h.ts.max() - h.ts.min()).days / 30 + 1
    t["rate"] = t.plays / span
    # v1: ranks of shrunk keep, pct, replays
    mk, mp = 1 - t.skips.sum() / t.plays.sum(), np.average(t.pct, weights=t.plays)
    keep = ((t.plays - t.skips) + K * mk) / (t.plays + K)
    pct = (t.pct * t.plays + K * mp) / (t.plays + K)
    r = lambda s: s.rank(pct=True)
    t["v1"] = .4 * r(keep) + .3 * r(pct) + .3 * r(np.log1p(t.plays))
    # v2 components: value = share of the song played (early skip ~0, finish ~1), pooled to the artist then global
    gv = np.average(t.pct, weights=t.plays)
    a = t.assign(s=t.pct * t.plays).groupby("artist")[["s", "plays"]].transform("sum")
    art_prior = ((a.s - t.pct * t.plays) + KA * gv) / ((a.plays - t.plays) + KA)  # excludes the track itself
    t["value"] = (t.pct * t.plays + K * art_prior) / (t.plays + K)
    t["log_rate"] = np.log1p(t.rate * 10)
    t["log_plays"] = np.log1p(t.plays)
    return t


def target(h, t):
    d = h[h.spotify_track_uri.isin(t.index)]
    return d.spotify_track_uri, d.skipped


FEATS = ["value", "chosen", "log_plays"]
tA = feats(A)
u, y = target(B, tA)
X = tA.loc[u, FEATS]
lr = LogisticRegression(C=1.0, max_iter=1000).fit((X - X.mean()) / X.std(), y)
w = pd.Series(lr.coef_[0], index=FEATS)

tAB = feats(pd.concat([A, B]))
u, y = target(C, tAB)
mu, sd = X.mean(), X.std()
z = (tAB[FEATS] - mu) / sd
tAB["v2"] = -(z @ w)  # skip-risk weights negated: higher = more liked
res = []
for name, s in [("v1 (ranked blend, guessed weights)", tAB.v1), ("v2 (artist-pooled value + chosen + replays, fitted)", tAB.v2),
                ("replay count only", tAB.log_plays), ("share listened only", tAB.value)]:
    res.append((name, roc_auc_score(1 - y, s.loc[u].values)))  # high score = liked -> predicts NOT skipped
res = pd.DataFrame(res, columns=["score", "AUC for later skips (higher = better)"]).round(3)
md = ["# Vibe score: v1 vs v2", "",
      f"Plays: A={len(A)}, B={len(B)}, C={len(C)}. Test on {len(u)} later plays of {u.nunique()} tracks seen earlier.", "",
      res.to_markdown(index=False), "", "Fitted v2 weights (standardised, negative = fewer skips):", "",
      w.round(3).to_frame("coef").to_markdown(), ""]
(ROOT / "reports" / "vibe_score_comparison.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))



rows = []
for nm, hist, test in [("A -> B", A, B), ("A -> C (older history only)", A, C), ("A+B -> C (incl. recent)", pd.concat([A, B]), C)]:
    t = feats(hist); uu, yy = target(test, t)
    rows.append([nm, len(uu)] + [roc_auc_score(1 - yy, t.loc[uu, f].values) for f in ["log_plays", "value", "chosen", "v1"]])
win = pd.DataFrame(rows, columns=["history -> test", "plays", "replays", "share listened", "chosen", "v1"]).round(3)
md += ["## Stability across time windows (AUC; 0.5 = no signal, <0.5 = inverted)", "", win.to_markdown(index=False), "",
       "Scores built only from older history do NOT predict later skips (some invert: songs you used to play a lot get skipped more later, "
       "a repeat-fatigue effect). Adding the most recent plays restores the signal. The fitted v2 weights are therefore unstable and v2 is not adopted."]
(ROOT / "reports" / "vibe_score_comparison.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md[-8:]))

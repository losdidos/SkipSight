"""Does the data support the goal? Taste-similarity priors for never-played songs
(leave-one-artist-out) and skip-timing curves by genre. No model fitting.
Writes reports/taste_analysis.md and reports/figures/taste/*.png"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures" / "taste"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})

tr = pd.read_csv(ROOT / "data/processed/tracks.csv")
pl = pd.read_csv(ROOT / "data/processed/skipsight_dataset.csv",
                 usecols=["spotify_track_uri", "ms_played", "duration_ms", "pct_played", "skipped"])
sel = pd.read_csv(ROOT / "reports/feature_selection.csv")
tr = tr[tr.plays > 0].reset_index(drop=True)
G = [c for c in tr if c.startswith("g_") and c != "g_dutch_local"]

mu = tr.skips.sum() / tr.plays.sum()
K = 20
md = []

# ---------- taste priors for a never-played song (leave-one-artist-out) ----------
audio = [f for f in sel[sel.shortlist].feature if not f.startswith("g_") and f in tr
         and tr[f].notna().mean() > .9 and tr[f].nunique() > 5]
X = tr[audio].copy()
X = X.fillna(X.median())
X = (X - X.mean()) / X.std()
X = pd.concat([X, tr[G] * 1.5], axis=1).to_numpy()  # genre flags count as taste dimensions
art = tr.artist_name.to_numpy()
nn = NearestNeighbors(n_neighbors=60).fit(X)
dist, idx = nn.kneighbors(X)
knn = np.zeros(len(tr))
for i in range(len(tr)):
    j = [k for k in idx[i] if art[k] != art[i]][:25]  # never the same artist
    knn[i] = (tr.skips.to_numpy()[j].sum() + K * mu) / (tr.plays.to_numpy()[j].sum() + K * mu / mu * 1) if False else \
        (tr.skips.to_numpy()[j].sum() + K * mu) / (tr.plays.to_numpy()[j].sum() + K)

gen_sum = np.zeros(len(tr)); gen_cnt = np.zeros(len(tr))
for g in G:
    m = tr[g] == 1
    if m.sum() == 0:
        continue
    s_all, p_all = tr.skips[m].sum(), tr.plays[m].sum()
    a_s = tr[m].groupby("artist_name").skips.transform("sum")
    a_p = tr[m].groupby("artist_name").plays.transform("sum")
    gen_sum[m.to_numpy()] += ((s_all - a_s) + K * mu) / ((p_all - a_p) + K)
    gen_cnt[m.to_numpy()] += 1
gen = np.where(gen_cnt > 0, gen_sum / np.maximum(gen_cnt, 1), mu)
ap = tr.groupby("artist_name")[["skips", "plays"]].transform("sum")
loo = ((ap.skips - tr.skips) + K * mu) / ((ap.plays - tr.plays) + K)  # artist known, track new
has_art = (ap.plays - tr.plays) > 0
tr["knn"], tr["gen"], tr["artist_loo"] = knn, gen, loo

pm = tr.set_index("uri")[["knn", "gen", "artist_loo"]]
p = pl.join(pm, on="spotify_track_uri").dropna(subset=["knn"])
p["has_art"] = p.spotify_track_uri.map(tr.set_index("uri").pipe(lambda t: has_art.set_axis(t.index)))
p["combo"] = p.knn.rank(pct=True) + p.gen.rank(pct=True)
rows = []
for name, m in [("genre prior (artist held out)", "gen"), ("audio+genre neighbours (artist held out)", "knn"),
                ("both averaged", "combo"), ("artist prior (other tracks of artist)", "artist_loo")]:
    s = p if m != "artist_loo" else p[p.has_art]
    rows.append((name, roc_auc_score(s.skipped, s[m]), len(s)))
auc = pd.DataFrame(rows, columns=["score", "AUC", "plays"])
md += ["# Taste analysis", "", "## 1. Can a never-played song be scored from similar songs?", "",
       "Each track is scored using only *other* artists' tracks (leave-one-artist-out), so it mimics a new song. "
       "AUC = how well the score ranks skipped vs not-skipped plays (0.5 = chance).", "",
       auc.round(3).to_markdown(index=False), ""]

# score deciles -> real skip rate
p["dec"] = pd.qcut(p.knn.rank(method="first"), 5, labels=False)
d = p.groupby("dec").agg(pred=("knn", "mean"), real=("skipped", "mean"), n=("skipped", "size"))
plt.figure(figsize=(5.5, 4))
plt.plot(d.pred * 100, d.real * 100, "o-", color="#2a9d8f")
lim = [0, max(d.real.max(), d.pred.max()) * 110]
plt.plot(lim, lim, "k--", lw=1)
plt.xlabel("neighbour-based predicted skip rate %")
plt.ylabel("actual skip rate %")
plt.title("Similar-song score vs reality (5 groups)")
plt.tight_layout(); plt.savefig(OUT / "1_neighbour_calibration.png"); plt.close()
md += ["Actual skip rate by neighbour-score quintile (lowest -> highest): " +
       ", ".join(f"{x*100:.1f}%" for x in d.real), ""]

# ---------- skip timing by genre ----------
p2 = pl.join(tr.set_index("uri")[["knn"] + G], on="spotify_track_uri").dropna(subset=["knn"])
p2["sec"] = p2.ms_played / 1000
p2["pct"] = (p2.pct_played.clip(0, 1) * 100)
sk = p2[p2.skipped == 1]
top = [g for g in G if p2[g].sum() >= 250]
t = np.arange(0, 31)
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for g in top:
    s = p2[p2[g] == 1]
    ax.plot(t, [((s.skipped == 1) & (s.sec <= k)).mean() * 100 for k in t], label=f"{g[2:]} ({len(s)})", lw=1.8)
ax.set_xlabel("seconds into the song"); ax.set_ylabel("% of plays already skipped")
ax.legend(fontsize=7, ncol=2); ax.set_title("Cumulative skips over the first 30s, by genre")
plt.tight_layout(); plt.savefig(OUT / "2_genre_skip_timing_seconds.png"); plt.close()

bins = np.arange(0, 101, 10)
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for g in top:
    s = sk[sk[g] == 1]
    h = np.histogram(s.pct, bins)[0] / max(len(s), 1) * 100
    ax.plot(bins[:-1] + 5, h, marker="o", ms=3, label=g[2:])
ax.set_xlabel("% of song played when skipped"); ax.set_ylabel("% of that genre's skips")
ax.legend(fontsize=7, ncol=2); ax.set_title("Where in the song each genre gets skipped")
plt.tight_layout(); plt.savefig(OUT / "3_genre_skip_timing_pct.png"); plt.close()

# overall skip share per % block (the shape the per-% model must learn)
h_all = np.histogram(sk.pct, bins)[0] / len(sk) * 100
plt.figure(figsize=(6, 3.8))
plt.bar(bins[:-1] + 5, h_all, width=8, color="#e76f51")
plt.xlabel("% of song played when skipped"); plt.ylabel("% of all skips")
plt.title("Skip timing is front-loaded")
plt.tight_layout(); plt.savefig(OUT / "4_skip_pct_blocks.png"); plt.close()

# does a high whole-song skip score also mean earlier skips? (link between the two models)
sk = sk.copy()
sk["grp"] = pd.qcut(sk.knn.rank(method="first"), 3, labels=["low-risk songs", "mid", "high-risk songs"])
plt.figure(figsize=(7, 4))
for g, c in zip(sk.grp.cat.categories, ["#457b9d", "#999", "#e76f51"]):
    s = sk[sk.grp == g]
    plt.plot(np.sort(s.sec.clip(upper=60)), np.arange(1, len(s) + 1) / len(s) * 100, label=f"{g} (n={len(s)})", color=c, lw=2)
plt.xlabel("seconds played before skip (capped at 60)"); plt.ylabel("% of skips done")
plt.legend(); plt.title("Given a skip, when does it happen? by song risk")
plt.tight_layout(); plt.savefig(OUT / "5_timing_by_risk.png"); plt.close()
med = sk.groupby("grp", observed=True).sec.median()
md += ["## 2. Skip timing", "",
       "Median seconds before a skip, by neighbour-risk tertile: " + ", ".join(f"{k} {v:.1f}s" for k, v in med.items()), ""]
gt = []
for g in top:
    s = p2[p2[g] == 1]; k = s[s.skipped == 1]
    gt.append((g[2:], len(s), s.skipped.mean() * 100, (k.sec < 3).mean() * 100 if len(k) else np.nan,
               k.sec.median() if len(k) else np.nan))
md += [pd.DataFrame(gt, columns=["genre", "plays", "skip %", "% of skips <3s", "median skip s"]).round(1)
       .to_markdown(index=False), ""]

# ---------- coverage: how much of the history can teach about a new song ----------
pc = tr.plays.value_counts(normalize=True).sort_index()
md += ["## 3. Data support", "",
       f"- tracks: {len(tr)}, artists: {tr.artist_name.nunique()}, plays: {int(tr.plays.sum())}, skips: {int(tr.skips.sum())}",
       f"- tracks with a single play: {(tr.plays == 1).mean()*100:.0f}%; artists with a single track: "
       f"{(tr.groupby('artist_name').size() == 1).mean()*100:.0f}%",
       f"- tracks never skipped: {(tr.skips == 0).mean()*100:.0f}%; tracks with 3+ plays: {(tr.plays >= 3).mean()*100:.0f}%",
       f"- genres with >=250 plays: {len(top)} of {len(G)}", ""]
(ROOT / "reports" / "taste_analysis.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))


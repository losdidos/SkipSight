"""Trainability check (not the final model): grouped CV with leakage-safe features.
Writes reports/trainability.md and reports/figures/trainability/*.png"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures" / "trainability"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})

tr = pd.read_csv(ROOT / "data/processed/tracks.csv")
pl = pd.read_csv(ROOT / "data/processed/skipsight_dataset.csv",
                 usecols=["spotify_track_uri", "artist_name", "ms_played", "skipped"])
sel = pd.read_csv(ROOT / "reports/feature_selection.csv")
feats = [f for f in sel[sel.shortlist].feature if f in tr]
df = pl.merge(tr[["uri"] + feats], left_on="spotify_track_uri", right_on="uri")
df["sec"] = df.ms_played / 1000
groups = df.artist_name  # held-out artists = a genuinely new song
X, y = df[feats], df.skipped
md = ["# Trainability check", "",
      f"{len(df)} plays, {df.skipped.sum()} skips, {len(feats)} shortlisted features, "
      "5-fold CV **grouped by artist** (every test artist is unseen, like a new import).", ""]


def clf():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=60,
                                          l2_regularization=5, random_state=0)


def reg():
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=120, min_samples_leaf=40,
                                         l2_regularization=5, random_state=0)


# ---- model 1: whole-song skip, unseen artists ----
oof = np.zeros(len(df)); imp = pd.DataFrame(0.0, index=feats, columns=["m"]); aucs = []
for a, b in GroupKFold(5).split(X, y, groups):
    m = clf().fit(X.iloc[a], y.iloc[a]); oof[b] = m.predict_proba(X.iloc[b])[:, 1]
    aucs.append(roc_auc_score(y.iloc[b], oof[b]))
    pi = permutation_importance(m, X.iloc[b], y.iloc[b], scoring="roc_auc", n_repeats=4, random_state=0)
    imp["m"] += pi.importances_mean / 5
auc = roc_auc_score(y, oof)
rng = np.random.default_rng(0)
null = [roc_auc_score(rng.permutation(y.values), oof) for _ in range(200)]
md += ["## Model 1 - will I skip this song? (unseen artists)", "",
       f"- out-of-fold AUC **{auc:.3f}** (folds: {', '.join(f'{x:.2f}' for x in aucs)}); shuffled-label AUC "
       f"{np.mean(null):.3f} +/- {np.std(null):.3f}, so the signal is real but modest.", ""]
dec = pd.qcut(pd.Series(oof).rank(method="first"), 5, labels=False)
cal = df.groupby(dec).skipped.mean() * 100
md += ["- actual skip rate by predicted-risk quintile: " + ", ".join(f"{v:.1f}%" for v in cal), ""]
top = imp.m.sort_values(ascending=False).head(15)[::-1]
plt.figure(figsize=(7, 5)); plt.barh(top.index, top.values, color="#2a9d8f")
plt.xlabel("AUC lost when feature is shuffled"); plt.title("Most impactful features: will I skip it?")
plt.tight_layout(); plt.savefig(OUT / "1_importance_whole_song.png"); plt.close()
md += ["Top features (AUC drop when shuffled):", "", top[::-1].round(4).to_frame("drop").to_markdown(), ""]
plt.figure(figsize=(5, 3.6)); plt.bar(range(5), cal.values, color="#e76f51")
plt.xticks(range(5), ["safest", "2", "3", "4", "riskiest"]); plt.ylabel("actual skip rate %")
plt.title("Predicted risk vs reality (unseen artists)")
plt.tight_layout(); plt.savefig(OUT / "2_risk_quintiles.png"); plt.close()

# ---- model 2: when in the song, given a skip ----
sk = df[df.skipped == 1].reset_index(drop=True)
ys = np.log1p(sk.sec.clip(upper=120)); Xs = sk[feats]
oof2 = np.zeros(len(sk)); imp2 = pd.Series(0.0, index=feats)
for a, b in GroupKFold(5).split(Xs, ys, sk.artist_name):
    m = reg().fit(Xs.iloc[a], ys.iloc[a]); oof2[b] = m.predict(Xs.iloc[b])
    pi = permutation_importance(m, Xs.iloc[b], ys.iloc[b], scoring="r2", n_repeats=4, random_state=0)
    imp2 += pi.importances_mean / 5
r2 = r2_score(ys, oof2); rho = pd.Series(oof2).corr(ys, method="spearman")
md += ["## Model 2 - how early do I skip it? (given a skip, unseen artists)", "",
       f"- out-of-fold R2 on log(seconds) **{r2:.3f}**, rank correlation **{rho:.2f}** ({len(sk)} skips)", ""]
top2 = imp2.sort_values(ascending=False).head(12)[::-1]
plt.figure(figsize=(7, 4.5)); plt.barh(top2.index, top2.values, color="#e9c46a")
plt.xlabel("R2 lost when feature is shuffled"); plt.title("Most impactful features: how early do I skip?")
plt.tight_layout(); plt.savefig(OUT / "3_importance_timing.png"); plt.close()
md += ["Top features (R2 drop when shuffled):", "", top2[::-1].round(4).to_frame("drop").to_markdown(), ""]
(ROOT / "reports" / "trainability.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))

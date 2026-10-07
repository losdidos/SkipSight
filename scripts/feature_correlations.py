"""Model-free correlation of each shortlisted feature with the skipped label (play level)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures" / "features"
tr = pd.read_csv(ROOT / "data/processed/tracks.csv")
pl = pd.read_csv(ROOT / "data/processed/skipsight_dataset.csv", usecols=["spotify_track_uri", "artist_name", "ms_played", "skipped"])
sel = pd.read_csv(ROOT / "reports/feature_selection.csv")
feats = [f for f in sel[sel.shortlist].feature if f in tr]
df = pl.merge(tr[["uri"] + feats], left_on="spotify_track_uri", right_on="uri")
df["sec"] = df.ms_played / 1000
sk = df[df.skipped == 1]
rows = []
for f in feats:
    d = df[[f, "skipped"]].dropna()
    s = sk[[f, "sec"]].dropna()
    rows.append((f, d[f].corr(d.skipped, method="spearman"), s[f].corr(np.log1p(s.sec), method="spearman"), len(d)))
r = pd.DataFrame(rows, columns=["feature", "corr_with_skipped", "corr_with_skip_time", "plays"])
r["abs"] = r.corr_with_skipped.abs()
r = r.sort_values("abs", ascending=False).drop(columns="abs")
r.round(3).to_csv(ROOT / "reports" / "feature_correlations.csv", index=False)
print(r.head(15).round(3).to_string(index=False))
top = r.head(15)[::-1]
plt.figure(figsize=(7, 5))
plt.barh(top.feature, top.corr_with_skipped, color=np.where(top.corr_with_skipped > 0, "#e76f51", "#457b9d"))
plt.axvline(0, color="k", lw=.8)
plt.xlabel("Spearman correlation with skipped (1 = skipped)")
plt.title("Raw data correlation with skipping (no model)")
plt.gca().spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(OUT / "corr_with_skipped.png", dpi=130)

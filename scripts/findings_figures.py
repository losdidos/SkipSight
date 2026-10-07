"""Simple figures that back the main findings. Writes reports/figures/findings/*.png"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures" / "findings"
OUT.mkdir(parents=True, exist_ok=True)

plays = pd.read_csv(ROOT / "data/processed/skipsight_dataset.csv", usecols=["spotify_track_uri", "ms_played", "skipped"])
tracks = pd.read_csv(ROOT / "data/processed/tracks.csv")
sel = pd.read_csv(ROOT / "reports/feature_selection.csv")
cols = ["uri", "tempo", "pv_harmonic_ratio", "release_year", "artist_name"] + [c for c in tracks if c.startswith("g_")]
df = plays.merge(tracks[cols], left_on="spotify_track_uri", right_on="uri")
df["sec"] = df.ms_played / 1000
df["early"] = ((df.skipped == 1) & (df.sec < 3)).astype(int)
overall = df.skipped.mean()
GENRES = {c: c[2:].replace("_", "/") for c in tracks if c.startswith("g_") and c != "g_dutch_local"}
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})


def save(name):
    plt.tight_layout()
    plt.savefig(OUT / name)
    plt.close()


# 1. feature funnel
v = sel.verdict.str.replace(r"KEEP.*", "keep", regex=True).str.replace(r"drop: (.*)", r"drop: \1", regex=True)
v = v.replace({"weak (no clear signal)": "weak signal"}).value_counts()
order = ["keep", "weak signal", "drop: redundant", "drop: constant", "drop: <30% coverage"]
v = v.reindex([o for o in order if o in v.index])
plt.figure(figsize=(7, 3.5))
plt.barh(v.index[::-1], v.values[::-1], color=["#bbb"] * (len(v) - 1) + ["#2a9d8f"])
for i, x in enumerate(v.values[::-1]):
    plt.text(x + 1, i, str(x), va="center")
plt.title(f"{len(sel)} candidate features: what we keep")
save("1_feature_funnel.png")

# 2. tempo bins: early-skip share vs overall skip rate
df["tq"] = pd.qcut(df.tempo, 4, labels=False)
edges = df.groupby("tq").tempo.agg(["min", "max"]).round().astype(int)
labels = [f"{a}-{b}" for a, b in zip(edges["min"], edges["max"])]
g = df.groupby("tq").agg(early=("early", "mean"), skip=("skipped", "mean")) * 100
x = np.arange(4)
plt.figure(figsize=(7, 4))
plt.bar(x - 0.2, g.skip, 0.4, label="skipped at all", color="#bbb")
plt.bar(x + 0.2, g.early, 0.4, label="skipped in first 3s", color="#e76f51")
plt.xticks(x, labels)
plt.xlabel("tempo (BPM, quartiles)")
plt.ylabel("% of plays")
plt.legend()
plt.title("Tempo changes WHEN you skip, not whether")
save("2_tempo_early_vs_overall.png")

# 3. cumulative skips by time for slow / mid / fast
df["tier"] = pd.cut(df.tq, [-1, 0, 2, 3], labels=["slow (Q1)", "mid (Q2-Q3)", "fast (Q4)"])
t = np.arange(0, 31)
plt.figure(figsize=(7, 4))
for tier, c in zip(df.tier.cat.categories, ["#457b9d", "#999", "#e76f51"]):
    s = df[df.tier == tier]
    plt.plot(t, [((s.skipped == 1) & (s.sec <= k)).mean() * 100 for k in t], label=tier, color=c, lw=2)
plt.xlabel("seconds into the song")
plt.ylabel("% of plays already skipped")
plt.legend()
plt.title("Slow and fast songs are dropped sooner than mid-tempo")
save("3_cumulative_skip_by_tempo.png")

# 4. tempo effect inside genres
rows = []
for c, name in GENRES.items():
    s = df[df[c] == 1]
    if len(s) < 300:
        continue
    lo, hi = s[s.tempo <= s.tempo.quantile(.25)], s[s.tempo >= s.tempo.quantile(.75)]
    rows.append((name, (hi.early.mean() - lo.early.mean()) * 100, len(s)))
r = pd.DataFrame(rows, columns=["g", "d", "n"]).sort_values("d")
plt.figure(figsize=(7, 4))
plt.barh(r.g, r.d, color=np.where(r.d > 0, "#e76f51", "#457b9d"))
plt.axvline(0, color="k", lw=.8)
plt.xlabel("extra % skipped in first 3s: fastest quarter vs slowest quarter")
plt.title("The tempo effect shows up within genres")
save("4_tempo_within_genre.png")

# 5. genre skip rate, shrunk toward the mean
K = 50
rows = []
for c, name in GENRES.items():
    s = df[df[c] == 1]
    if len(s) < 30:
        continue
    rows.append((name, s.skipped.mean() * 100, (s.skipped.sum() + K * overall) / (len(s) + K) * 100, len(s)))
r = pd.DataFrame(rows, columns=["g", "raw", "shr", "n"]).sort_values("shr")
plt.figure(figsize=(7, 5))
plt.barh(r.g, r.raw, color="#ddd", label="raw")
plt.barh(r.g, r.shr, height=.4, color="#2a9d8f", label=f"shrunk (K={K})")
plt.axvline(overall * 100, color="k", ls="--", lw=1, label="overall")
plt.xlabel("skip rate %")
plt.legend()
plt.title("Genre skip rates (small genres pulled to the mean)")
save("5_genre_skip_rate.png")

# 6. harmonic share vs early skip
df["hq"] = pd.qcut(df.pv_harmonic_ratio, 5, labels=False, duplicates="drop")
g = df.groupby("hq").early.mean() * 100
plt.figure(figsize=(6, 3.8))
plt.bar(["low", "", "mid", "", "high"], g.values, color="#e9c46a")
plt.xlabel("harmonic share (quintiles)")
plt.ylabel("% skipped in first 3s")
plt.title("More melodic / less percussive = earlier skips")
save("6_harmonic_early_skip.png")

# 7. artist fingerprint: overall vs within-artist correlation of shortlisted features
s = sel[sel.shortlist & sel.within_artist_r.notna()].copy()
s["a"] = s.whole_r.abs()
s = s.sort_values("a", ascending=False).head(12)[::-1]
y = np.arange(len(s))
plt.figure(figsize=(7, 5))
plt.barh(y + .2, s.whole_r.abs(), .4, label="across all tracks", color="#2a9d8f")
plt.barh(y - .2, s.within_artist_r.abs(), .4, label="within one artist", color="#bbb")
plt.yticks(y, s.feature)
plt.xlabel("|correlation with skip rate|")
plt.legend()
plt.title("Many features mostly identify the artist")
save("7_artist_fingerprint.png")

# 8. plays vs skip rate: why priors need shrinking
tr = tracks[tracks.plays >= 1]
b = pd.cut(tr.plays, [0, 1, 2, 4, 9, 1000], labels=["1", "2", "3-4", "5-9", "10+"])
g = tr.groupby(b, observed=True).agg(sr=("skip_rate", "mean"), n=("uri", "size"))
plt.figure(figsize=(6, 3.8))
plt.bar(g.index.astype(str), g.sr * 100, color="#8d99ae")
for i, (sr, n) in enumerate(zip(g.sr, g.n)):
    plt.text(i, sr * 100 + .3, f"n={n}", ha="center", fontsize=8)
plt.xlabel("plays of the track")
plt.ylabel("mean track skip rate %")
plt.title("Songs you replay are rarely skipped")
save("8_plays_vs_skip.png")
print("saved to", OUT)

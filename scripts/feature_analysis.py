"""Model-free feature analysis -> reports/feature_analysis.md, reports/feature_selection.csv, reports/figures/features/

Questions answered (no model is trained):
  1. Which features are redundant with each other?              (rank-correlation clusters, track level)
  2. Which features relate to WHETHER a song gets skipped?        (weighted rank correlation, 99% track-bootstrap CI)
  3. Which features relate to WHEN in the song the skip happens?  (rank correlation among skips, 99% bootstrap CI)
  4. Do those relations survive controlling for the artist?       (within-artist correlation)
  5. Tempo x genre and genre priors (shrunk rates for rarely played genres)
Everything uses static, song-level features only (what we can know for a never-played song).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "reports" / "figures" / "features"
FIG.mkdir(parents=True, exist_ok=True)
GREEN, RNG, B = "#1db954", np.random.default_rng(0), 500
MIN_COVERAGE, CLUSTER_RHO, RELATED_RHO = 0.3, 0.85, 0.6
WHOLE_MIN, TIMING_MIN = 0.04, 0.08  # minimum |rank correlation| to count as useful (besides CI excluding 0)
PERMS, P_MAX = 1000, 0.002  # permutation test for shape-agnostic relevance (5 quantile bins); ~137 features tested
out: list[str] = []

NOT_FEATURES = {"ms_played", "pct_played", "skipped", "finished", "dz_track_id", "dz_artist_id", "dz_album_id",
                "rb_found", "dz_found", "it_found", "mb_found", "pv_ok", "pv_seconds",
                "duration_ms"}  # duration_ms is partly measured from the history -> use dz_duration_s / it_duration_ms instead


def md(df: pd.DataFrame, fmt: str = ".3f") -> str:
    return df.to_markdown(index=False, floatfmt=fmt)


def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def wcorr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    sx, sy = np.sqrt(np.average((x - mx) ** 2, weights=w)), np.sqrt(np.average((y - my) ** 2, weights=w))
    return float(cov / (sx * sy)) if sx > 0 and sy > 0 else np.nan


def boot_ci(stat, n_groups: int, idx_by_group: list[np.ndarray]) -> tuple[float, float]:
    vals = []
    for _ in range(B):
        pick = RNG.integers(0, n_groups, n_groups)
        idx = np.concatenate([idx_by_group[i] for i in pick])
        v = stat(idx)
        if not np.isnan(v):
            vals.append(v)
    return (np.percentile(vals, 0.5), np.percentile(vals, 99.5)) if len(vals) > 50 else (np.nan, np.nan)


def eta2(codes: np.ndarray, y: np.ndarray, w: np.ndarray, k: int = 5) -> float:
    sw = np.bincount(codes, weights=w, minlength=k)
    sy = np.bincount(codes, weights=w * y, minlength=k)
    m = np.average(y, weights=w)
    ok = sw > 0
    between = np.sum(sw[ok] * (sy[ok] / sw[ok] - m) ** 2)
    total = np.sum(w * (y - m) ** 2)
    return float(between / total) if total > 0 else np.nan


def perm_test(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Shape-agnostic effect: variance of y explained by 5 quantile bins of x, p-value from permuting y."""
    codes = np.minimum(((rankdata(x) - 1) / len(x) * 5).astype(int), 4)
    obs = eta2(codes, y, w)
    null = np.array([eta2(codes, y[p], w[p]) for p in (RNG.permutation(len(y)) for _ in range(PERMS))])
    return obs, float((np.sum(null >= obs) + 1) / (PERMS + 1))


def nonlinear(x, ok, t, skips, sx, sok) -> dict:
    ew, pw = perm_test(x[ok], t.rate.values[ok], t.n.values[ok].astype(float))
    et, pt = (np.nan, np.nan)
    if sok.sum() > 150:
        et, pt = perm_test(sx[sok], skips.log_s.values[sok], np.ones(sok.sum()))
    return {"nl_whole_eta2": ew, "nl_whole_p": pw, "nl_timing_eta2": et, "nl_timing_p": pt}


def main() -> None:
    global out
    d = pd.read_csv(ROOT / "data/processed/skipsight_full.csv")
    base = d.skipped.mean()
    t = d.drop_duplicates("uri").set_index("uri")
    agg = d.groupby("uri").agg(n=("skipped", "size"), s=("skipped", "sum"))
    t = t.join(agg)
    t["rate"] = t.s / t.n

    cand = [c for c in t.select_dtypes("number") if not c.startswith(("hist_", "ctx_")) and c not in NOT_FEATURES and c not in ("n", "s", "rate")]
    cov = t[cand].notna().mean()
    const = [c for c in cand if t[c].nunique() <= 1]
    out.append(f"# Feature analysis (no model)\n\n{len(t)} tracks, {len(d)} plays, skip rate {base:.2%}. {len(cand)} candidate static features. "
               f"`duration_ms` is excluded (partly measured from the history); `dz_duration_s` stands in for track length.")

    # ---- 2. whole-song relevance: weighted rank-correlation of feature with track skip rate (weights = plays)
    skips = d[d.skipped == 1].copy()
    skips["log_s"] = np.log1p(skips.ms_played / 1000)
    rows = []
    track_idx = {u: i for i, u in enumerate(t.index)}
    skips["ti"] = skips.uri.map(track_idx)
    by_track_skips = [np.where(skips.ti.values == i)[0] for i in range(len(t))]  # skip rows per track
    for c in cand:
        x = t[c].values.astype(float)
        ok = ~np.isnan(x)
        if ok.sum() < 200 or c in const:
            rows.append({"feature": c})
            continue
        # whole-song
        xr = np.full(len(t), np.nan)
        xr[ok] = rankdata(x[ok])
        yy, ww = t.rate.values, t.n.values.astype(float)
        r_whole = wcorr(xr[ok], yy[ok], ww[ok])
        ok_idx = np.where(ok)[0]
        groups = [np.array([i]) for i in ok_idx]
        lo, hi = boot_ci(lambda idx: wcorr(xr[idx], yy[idx], ww[idx]), len(groups), groups)
        # timing: among skips
        sx = x[skips.ti.values]
        sok = ~np.isnan(sx)
        r_time, tlo, thi = np.nan, np.nan, np.nan
        if sok.sum() > 150:
            sxr = np.full(len(skips), np.nan)
            sxr[sok] = rankdata(sx[sok])
            syr = rankdata(skips.log_s.values)
            r_time = float(np.corrcoef(sxr[sok], syr[sok])[0, 1])
            tr = np.array([i for i in ok_idx if len(by_track_skips[i])])
            tgroups = [by_track_skips[i] for i in tr]
            tlo, thi = boot_ci(lambda idx: float(np.corrcoef(sxr[idx], syr[idx])[0, 1]), len(tgroups), tgroups)
        # within artist (artists with >=2 tracks): demean rank and rate by artist
        sub = t[ok].assign(xr=xr[ok])
        sub = sub[sub.groupby("artist_name").xr.transform("size") >= 2]
        r_within = np.nan
        if len(sub) > 100:
            xd = sub.xr - sub.groupby("artist_name").xr.transform("mean")
            yd = sub.rate - sub.groupby("artist_name").rate.transform("mean")
            r_within = wcorr(xd.values, yd.values, sub.n.values.astype(float))
        rows.append({"feature": c, "whole_r": r_whole, "whole_lo": lo, "whole_hi": hi, "timing_r": r_time, "timing_lo": tlo, "timing_hi": thi,
                     "within_artist_r": r_within, **nonlinear(x, ok, t, skips, sx, sok)})
    f = pd.DataFrame(rows).set_index("feature")
    f["coverage"] = cov
    f["group"] = [("preview" if c.startswith("pv_") else "genre" if c.startswith("g_") else "deezer" if c.startswith("dz_") else
                   "itunes" if c.startswith("it_") else "title" if c.startswith("t_") else "reccobeats" if c in
                   ("tempo", "energy", "danceability", "valence", "acousticness", "instrumentalness", "liveness", "speechiness", "loudness", "key", "mode", "rb_duration_ms")
                   else "era/other") for c in f.index]
    f["useful_whole"] = ((f.whole_lo.gt(0) | f.whole_hi.lt(0)) & f.whole_r.abs().ge(WHOLE_MIN)) | f.nl_whole_p.le(P_MAX)
    f["useful_timing"] = ((f.timing_lo.gt(0) | f.timing_hi.lt(0)) & f.timing_r.abs().ge(TIMING_MIN)) | f.nl_timing_p.le(P_MAX)
    f["shape_whole"] = np.where(f.nl_whole_p.le(P_MAX) & ~((f.whole_lo.gt(0) | f.whole_hi.lt(0)) & f.whole_r.abs().ge(WHOLE_MIN)), "non-monotonic", "")
    floor = 1 / (PERMS + 1)
    f["strength"] = (-np.log10(f.nl_whole_p.fillna(1).clip(lower=floor)) + -np.log10(f.nl_timing_p.fillna(1).clip(lower=floor))
                     + 10 * (f.nl_whole_eta2.fillna(0) + f.nl_timing_eta2.fillna(0)))  # eta2 term only breaks ties at the p-value floor

    # ---- 1. redundancy clusters among features with enough coverage
    usable = [c for c in cand if cov[c] >= MIN_COVERAGE and c not in const]
    corr = t[usable].corr(method="spearman", min_periods=200).fillna(0).values
    np.fill_diagonal(corr, 1)
    dist = 1 - np.abs(corr)
    dist = (dist + dist.T) / 2
    labels = fcluster(linkage(squareform(dist, checks=False), "average"), 1 - CLUSTER_RHO, "distance")
    clus = pd.Series(labels, index=usable)
    f["cluster"] = clus
    sizes = clus.value_counts()
    f["cluster_size"] = f.cluster.map(sizes)
    f["is_rep"] = False
    for cl, members in clus.groupby(clus):
        m = list(members.index)
        best = max(m, key=lambda c: (round(f.loc[c, "strength"], 2), f.loc[c, "coverage"]))
        f.loc[best, "is_rep"] = True

    def verdict(r: pd.Series) -> str:
        if r.name in const:
            return "drop: constant"
        if r.coverage < MIN_COVERAGE:
            return "drop: <30% coverage"
        if r.cluster_size > 1 and not r.is_rep:
            return "drop: redundant"
        if r.useful_whole and r.useful_timing:
            return "KEEP: whole + timing"
        if r.useful_whole:
            return "KEEP: whole-song"
        if r.useful_timing:
            return "KEEP: timing"
        return "weak (no clear signal)"

    f["verdict"] = f.apply(verdict, axis=1)

    # looser tier: features that are related (|rho| >= RELATED_RHO) carry mostly the same information -> keep the strongest per group
    lab2 = pd.Series(fcluster(linkage(squareform(dist, checks=False), "average"), 1 - RELATED_RHO, "distance"), index=usable)
    f["related_cluster"] = lab2
    f["shortlist"] = False
    kept = f[f.verdict.str.startswith("KEEP")]
    for cl, members in kept.groupby("related_cluster"):
        f.loc[members.strength.idxmax(), "shortlist"] = True
    f.reset_index().to_csv(ROOT / "reports" / "feature_selection.csv", index=False)

    out.append("\n## 1. Summary of verdicts\n")
    vs = f.verdict.value_counts().rename_axis("verdict").reset_index(name="features")
    out += [md(vs, ".0f"), "\nBy source group:\n", md(pd.crosstab(f.group, f.verdict).reset_index(), ".0f"),
            f"\nRules: useful = |rank-corr| >= {WHOLE_MIN} (whole-song) or >= {TIMING_MIN} (timing) with a 99% track-bootstrap CI that excludes 0 "
            f"({B} resamples), OR a shape-agnostic permutation test on 5 quantile bins with p <= {P_MAX} ({PERMS} permutations) - this catches "
            f"U-shaped effects like tempo. Redundant = |Spearman| >= {CLUSTER_RHO} with a stronger feature. Shortlist = the strongest KEEP feature of "
            f"each group of related features (|rho| >= {RELATED_RHO}). Effects are weak overall, so 'KEEP' means 'has a detectable signal', not 'strong'."]

    keep = f[f.verdict.str.startswith("KEEP")].sort_values("strength", ascending=False)
    cols = ["group", "shortlist", "shape_whole", "whole_r", "nl_whole_eta2", "nl_whole_p", "timing_r", "nl_timing_eta2", "nl_timing_p", "within_artist_r"]
    short = keep[keep.shortlist]
    out += [f"\n## 2. Shortlist: {len(short)} features (one per group of related features)\n", md(short[cols].reset_index(), ".4f"),
            f"\n### All {len(keep)} KEEP features (shortlist flag shows the representative)\n", md(keep[cols].reset_index(), ".4f")]

    # ---- redundancy detail
    big = [(cl, list(m.index)) for cl, m in clus.groupby(clus) if len(m) > 1]
    lines = []
    for cl, m in sorted(big, key=lambda x: -len(x[1])):
        rep = next(c for c in m if f.loc[c, "is_rep"])
        others = [c for c in m if c != rep]
        lines.append(f"- **{rep}** (kept) stands in for {len(others)}: " + ", ".join(others[:12]) + (" ..." if len(others) > 12 else ""))
    out += [f"\n## 3. Redundancy: {len(big)} clusters of near-duplicate features (|rho| >= {CLUSTER_RHO})\n", "\n".join(lines)]

    both = f[(f.useful_whole | f.useful_timing)]
    out += ["\n## 4. Does the signal survive controlling for the artist?\n",
            "`within_artist_r` is the correlation after removing each artist's average (artists with >= 2 tracks). If it is much smaller than "
            "`whole_r`, the feature mostly acts as an artist identifier (cold-start songs by new artists get less from it).\n",
            md(both[["group", "whole_r", "within_artist_r"]].assign(ratio=lambda x: x.within_artist_r / x.whole_r).sort_values("whole_r", key=abs, ascending=False).head(20).reset_index())]

    # ---- figure: whole vs timing relevance
    plt.figure(figsize=(7, 6))
    for g_, sub in f.groupby("group"):
        plt.scatter(sub.whole_r, sub.timing_r, label=g_, s=18)
    plt.axhline(0, color="grey", lw=.5)
    plt.axvline(0, color="grey", lw=.5)
    plt.xlabel("rank-corr with WHETHER skipped (track skip rate)")
    plt.ylabel("rank-corr with WHEN skipped (among skips)")
    plt.legend(fontsize=7)
    plt.title("Each dot = one feature")
    plt.tight_layout()
    plt.savefig(FIG / "whole_vs_timing.png", dpi=110)
    plt.close()

    # ---- 5. tempo x genre
    secs = np.arange(0, 61)

    def curve(sub: pd.DataFrame) -> np.ndarray:
        r = np.array([(sub.ms_played >= s * 1000).sum() for s in secs])
        e = np.array([((sub.skipped == 1) & (sub.ms_played >= s * 1000) & (sub.ms_played < (s + 1) * 1000)).sum() for s in secs])
        return 1 - np.cumprod(1 - e / np.maximum(r, 1))

    def windows(sub: pd.DataFrame) -> dict:
        n = len(sub)
        e3 = ((sub.skipped == 1) & (sub.ms_played < 3000)).sum()
        e10 = ((sub.skipped == 1) & (sub.ms_played >= 3000) & (sub.ms_played < 10000)).sum()
        l10 = ((sub.skipped == 1) & (sub.ms_played >= 10000)).sum()
        lo, hi = wilson(e3, n)
        return {"plays": n, "skip_rate": sub.skipped.mean(), "skip<3s": e3 / n, "skip<3s_lo": lo, "skip<3s_hi": hi, "skip3-10s": e10 / n, "skip10s+": l10 / n}

    d["tempo_q"] = pd.qcut(d.tempo, 4, labels=["Q1 slow", "Q2", "Q3", "Q4 fast"])
    bounds = d.tempo.quantile([0, .25, .5, .75, 1]).round(0).tolist()
    rows = [{"tempo_quartile": q, **windows(s)} for q, s in d.groupby("tempo_q", observed=True)]
    out += [f"\n## 5. Tempo vs skip timing (ReccoBeats tempo, quartile edges {bounds} BPM)\n",
            "Share of ALL plays skipped inside each window (so it separates 'skips more' from 'skips earlier'):\n", md(pd.DataFrame(rows))]
    plt.figure(figsize=(7, 4.5))
    for q, s in d.groupby("tempo_q", observed=True):
        plt.plot(secs, curve(s) * 100, label=f"{q} ({len(s)})")
    plt.xlabel("seconds into song")
    plt.ylabel("cumulative skip probability %")
    plt.legend()
    plt.title("Skip curve by tempo quartile")
    plt.tight_layout()
    plt.savefig(FIG / "tempo_curves.png", dpi=110)
    plt.close()

    # is the tempo effect just genre? compare fast (Q4) vs rest within genre buckets
    d["fast"] = d.tempo >= d.tempo.quantile(.75)
    rows = []
    for gname in [c for c in d if c.startswith("g_")]:
        sub = d[d[gname] == 1]
        if len(sub) < 400 or sub.fast.sum() < 60:
            continue
        a, b = windows(sub[sub.fast]), windows(sub[~sub.fast])
        rows.append({"genre": gname[2:], "plays_fast": a["plays"], "plays_rest": b["plays"], "skip<3s fast": a["skip<3s"], "skip<3s rest": b["skip<3s"],
                     "diff": a["skip<3s"] - b["skip<3s"], "skip_rate fast": a["skip_rate"], "skip_rate rest": b["skip_rate"]})
    out += ["\n### Tempo effect inside each genre (fast = top quartile BPM)\n", md(pd.DataFrame(rows))]

    # same check for other candidate timing drivers
    for col, label in [("pv_harmonic_ratio", "harmonic share (melodic vs percussive)"), ("release_year", "release year"), ("valence", "valence")]:
        d["q"] = pd.qcut(d[col], 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"], duplicates="drop")
        out += [f"\n### {label} quartiles\n", md(pd.DataFrame([{"quartile": q, **windows(s)} for q, s in d.groupby("q", observed=True)]))]

    # ---- 6. genre priors with shrinkage
    K = 50
    gm = base
    rows = []
    for gname in [c for c in d if c.startswith("g_")]:
        sub = d[d[gname] == 1]
        w = windows(sub)
        k, n = sub.skipped.sum(), len(sub)
        lo, hi = wilson(k, n)
        rows.append({"genre": gname[2:], "tracks": sub.uri.nunique(), "plays": n, "raw_skip_rate": k / n, "ci_lo": lo, "ci_hi": hi,
                     "shrunk_rate(K=50)": (k + K * gm) / (n + K), "skip<3s": w["skip<3s"], "skip10s+": w["skip10s+"]})
    gdf = pd.DataFrame(rows).sort_values("shrunk_rate(K=50)", ascending=False)
    out += ["\n## 6. Genre priors (multi-label buckets)\n",
            "Raw rate, 95% Wilson interval and the shrunk estimate `(skips + 50*global) / (plays + 50)` that a model should use for rare genres. "
            "Country, K-pop and jazz look high but their intervals are wide - that is exactly the case shrinkage handles.\n", md(gdf)]
    plt.figure(figsize=(7, 5))
    gs = gdf.sort_values("raw_skip_rate")
    y = np.arange(len(gs))
    plt.errorbar(gs.raw_skip_rate, y, xerr=[gs.raw_skip_rate - gs.ci_lo, gs.ci_hi - gs.raw_skip_rate], fmt="o", color=GREEN, label="raw + 95% CI")
    plt.scatter(gs["shrunk_rate(K=50)"], y, marker="x", color="k", label="shrunk")
    plt.axvline(base, color="red", ls="--")
    plt.yticks(y, gs.genre)
    plt.legend()
    plt.title("Genre skip rate: raw vs shrunk")
    plt.tight_layout()
    plt.savefig(FIG / "genre_shrinkage.png", dpi=110)
    plt.close()

    (ROOT / "reports" / "feature_analysis.md").write_text("\n".join(out), encoding="utf-8")
    print(vs.to_string(index=False))


if __name__ == "__main__":
    main()

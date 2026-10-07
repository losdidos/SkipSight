"""Deep exploration of data/processed/skipsight_full.csv -> reports/deep_dive_full.md + reports/figures/full/

Sections: data quality, targets, skip timing (per-second curve), feature groups vs skip (correlation, mutual
information), artist / genre / era effects, cold-start check (artists seen once), audio-preview structure (PCA),
and a leakage-safe baseline (track-grouped CV, static features only) with permutation importance per feature group.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "reports" / "figures" / "full"
FIG.mkdir(parents=True, exist_ok=True)
GREEN = "#1db954"
out: list[str] = []


def md(df: pd.DataFrame, fmt: str = ".3f") -> str:
    return df.to_markdown(index=False, floatfmt=fmt)


def section(title: str, *parts: str) -> None:
    out.extend([f"\n## {title}\n", *parts])


def save(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG / f"{name}.png", dpi=110)
    plt.close()


def main() -> None:
    d = pd.read_csv(ROOT / "data/processed/skipsight_full.csv")
    d["start_ts"] = pd.to_datetime(d["start_ts"], utc=True, format="ISO8601")
    base = d.skipped.mean()
    out.append(f"# Deep dive: skipsight_full.csv\n\n{len(d)} plays, {d.uri.nunique()} tracks, {d.shape[1]} columns, skip rate {base:.2%}.")

    cols = {
        "audio (ReccoBeats)": ["tempo", "energy", "danceability", "valence", "acousticness", "instrumentalness", "liveness", "speechiness", "loudness", "key", "mode"],
        "preview (librosa)": [c for c in d if c.startswith("pv_") and c not in ("pv_ok", "pv_seconds")],
        "genre": [c for c in d if c.startswith("g_")],
        "deezer/popularity": ["dz_rank", "log_track_rank", "dz_artist_fans", "log_artist_fans", "dz_artist_nb_album", "dz_gain", "dz_bpm", "dz_n_contributors",
                              "dz_track_position", "dz_disk_number", "dz_album_nb_tracks"],
        "release/era": ["release_year", "track_age_years", "artist_begin_year"],
        "title/format": [c for c in d if c.startswith("t_") or c.startswith("album_is_")] + ["explicit", "duration_ms"],
    }
    cols = {k: [c for c in v if c in d] for k, v in cols.items()}

    # 1. data quality
    miss = d.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0].head(15).rename("missing").reset_index().rename(columns={"index": "column"})
    const = [c for c in d.select_dtypes("number") if d[c].nunique() <= 1]
    section("1. Data quality", "Top missing columns:\n", md(miss), f"\nConstant columns: {const or 'none'}.",
            f"\nPlays per track: median {d.groupby('uri').size().median():.0f}, mean {d.groupby('uri').size().mean():.1f}, "
            f"{(d.groupby('uri').size() == 1).mean():.0%} of tracks played once.")

    # 2. targets
    ek = d.end_kind.value_counts().rename_axis("end_kind").reset_index(name="plays").assign(share=lambda x: x.plays / len(d))
    rs = d.groupby("reason_start").agg(plays=("skipped", "size"), skip_rate=("skipped", "mean")).reset_index().sort_values("plays", ascending=False)
    section("2. Targets", md(ek), "\nSkip rate by start reason:\n", md(rs))

    # 3. timing
    sk = d[d.skipped == 1]
    qs = sk.ms_played.quantile([.1, .25, .5, .75, .9]) / 1000
    section("3. When skips happen", "Seconds into song at which skips occur (quantiles): " + ", ".join(f"p{int(k * 100)}={v:.1f}s" for k, v in qs.items()))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(sk.ms_played.clip(upper=60000) / 1000, bins=60, color=GREEN)
    ax[0].set_title("Skip time (s, capped 60)")
    ax[1].hist(sk.pct_played, bins=20, color=GREEN)
    ax[1].set_title("Skip position (% of song)")
    save("skip_timing")

    # hazard per second: P(skip in second s | still playing at s), censoring non-skip endings
    secs = np.arange(0, 60)
    at_risk = np.array([(d.ms_played >= s * 1000).sum() for s in secs])
    ev = np.array([((d.skipped == 1) & (d.ms_played >= s * 1000) & (d.ms_played < (s + 1) * 1000)).sum() for s in secs])
    haz = ev / np.maximum(at_risk, 1)
    cum = 1 - np.cumprod(1 - haz)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(secs, haz * 100, color=GREEN)
    ax[0].set_title("Skip hazard per second (%)")
    ax[1].plot(secs, cum * 100, color=GREEN)
    ax[1].set_title("Cumulative skip probability (%)")
    save("hazard_overall")
    section("3b. Per-second skip chance (overall)", md(pd.DataFrame({"second": [0, 1, 2, 3, 5, 10, 20, 30, 45, 59],
                                                                      "hazard_%": [haz[s] * 100 for s in [0, 1, 2, 3, 5, 10, 20, 30, 45, 59]],
                                                                      "cum_skip_%": [cum[s] * 100 for s in [0, 1, 2, 3, 5, 10, 20, 30, 45, 59]]})))

    # 4. univariate relations to skip (static, content-only features)
    num = d[[c for v in cols.values() for c in v]].select_dtypes("number")
    num = num.loc[:, num.notna().mean() > 0.5]
    t = d.groupby("uri").agg(skip_rate=("skipped", "mean"), n=("skipped", "size"))
    tn = d.drop_duplicates("uri").set_index("uri")[num.columns].join(t)
    tn = tn[tn.n >= 5]
    corr_t = tn[num.columns].apply(lambda c: c.corr(tn.skip_rate, method="spearman")).dropna()
    corr_p = num.apply(lambda c: c.corr(d.skipped))
    mi_in = num.fillna(num.median())
    mi = pd.Series(mutual_info_classif(mi_in, d.skipped, random_state=0), index=num.columns)
    tab = pd.DataFrame({"spearman_track_skip_rate": corr_t, "pearson_play_skipped": corr_p, "mutual_info": mi}).dropna()
    group_of = {c: g for g, v in cols.items() for c in v}
    tab["group"] = tab.index.map(group_of)
    tab = tab.reindex(tab.spearman_track_skip_rate.abs().sort_values(ascending=False).index)
    section(f"4. Feature vs skip (tracks with >=5 plays: {len(tn)})", md(tab.head(30).reset_index().rename(columns={"index": "feature"})))
    top = tab.head(25).iloc[::-1]
    top.spearman_track_skip_rate.plot.barh(figsize=(7, 7), color=GREEN)
    plt.title("Top correlations with track skip rate")
    save("corr_top")
    gsum = tab.groupby("group").agg(max_abs_corr=("spearman_track_skip_rate", lambda s: s.abs().max()), mean_abs_corr=("spearman_track_skip_rate", lambda s: s.abs().mean()),
                                   mean_mi=("mutual_info", "mean")).reset_index()
    section("4b. Strength by feature group", md(gsum))

    # 5. artists and cold start
    a = d.groupby("artist_name").agg(plays=("skipped", "size"), tracks=("uri", "nunique"), skip_rate=("skipped", "mean")).reset_index()
    big = a[a.plays >= 40].sort_values("skip_rate")
    section("5. Artists (>=40 plays)", "Lowest skip rate:\n", md(big.head(10)), "\nHighest skip rate:\n", md(big.tail(10)),
            f"\nArtists with 1 track only: {(a.tracks == 1).mean():.0%} (cold start for those is the hard case); they hold {a[a.tracks == 1].plays.sum() / len(d):.0%} of plays.")
    # shrinkage effect: artist skip-rate spread vs binomial noise
    exp = a.plays * base * (1 - base)
    section("5b. Is the artist effect real?", f"Variance of artist skip rates (>=40 plays) {big.skip_rate.var():.4f} vs pure-noise expectation {(exp[a.plays >= 40] / a.plays[a.plays >= 40] ** 2).mean():.4f}.")

    # 6. time/era
    d["year"] = d.start_ts.dt.year
    yr = d.groupby("year").agg(plays=("skipped", "size"), skip_rate=("skipped", "mean")).reset_index()
    ag = d[d.release_year.notna()].assign(age=lambda x: (x.start_ts.dt.year - x.release_year).clip(0, 15)).groupby("age").agg(plays=("skipped", "size"), skip_rate=("skipped", "mean")).reset_index()
    section("6. Era", "Skip rate by listening year:\n", md(yr), "\nSkip rate by song age at play time (years, capped 15):\n", md(ag))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(yr.year, yr.skip_rate, color=GREEN)
    ax[0].set_title("Skip rate by year")
    ax[1].bar(ag.age, ag.skip_rate, color=GREEN)
    ax[1].set_title("Skip rate by song age at play")
    save("era")

    # 7. skip curve by group (per-second hazard shape for 3 contrasting groups)
    def curve(mask: pd.Series) -> np.ndarray:
        s = d[mask]
        r = np.array([(s.ms_played >= x * 1000).sum() for x in secs])
        e = np.array([((s.skipped == 1) & (s.ms_played >= x * 1000) & (s.ms_played < (x + 1) * 1000)).sum() for x in secs])
        return 1 - np.cumprod(1 - e / np.maximum(r, 1))

    groups = {"energy high": d.energy > d.energy.median(), "energy low": d.energy <= d.energy.median(),
              "popular artist": d.log_artist_fans > d.log_artist_fans.median(), "niche artist": d.log_artist_fans <= d.log_artist_fans.median(),
              "rock/indie": (d.g_rock + d.g_indie_alt) > 0, "electronic": d.g_electronic == 1, "pop": d.g_pop == 1, "rap": d.g_rap_hiphop == 1}
    plt.figure(figsize=(8, 5))
    rows = []
    for k, m in groups.items():
        c = curve(m)
        plt.plot(secs, c * 100, label=k)
        rows.append({"group": k, "plays": int(m.sum()), "cum_skip_10s_%": c[10] * 100, "cum_skip_30s_%": c[30] * 100, "cum_skip_59s_%": c[59] * 100})
    plt.legend()
    plt.title("Cumulative skip probability by group")
    save("curves_by_group")
    section("7. Skip curves by group (cumulative skip probability)", md(pd.DataFrame(rows)))

    # 8. preview structure
    pv = d.drop_duplicates("uri").set_index("uri")[cols["preview (librosa)"]].dropna()
    z = StandardScaler().fit_transform(pv)
    pca = PCA(n_components=5).fit(z)
    comp = pd.DataFrame(pca.transform(z)[:, :2], index=pv.index, columns=["pc1", "pc2"]).join(t)
    section("8. Preview audio structure (PCA)", f"Explained variance of first 5 components: {np.round(pca.explained_variance_ratio_, 3).tolist()}.",
            f"Correlation of PC1/PC2 with track skip rate (tracks>=5 plays): {comp[comp.n >= 5].pc1.corr(comp[comp.n >= 5].skip_rate, method='spearman'):.3f} / "
            f"{comp[comp.n >= 5].pc2.corr(comp[comp.n >= 5].skip_rate, method='spearman'):.3f}.")
    c5 = comp[comp.n >= 5]
    plt.figure(figsize=(6, 5))
    plt.scatter(c5.pc1, c5.pc2, c=c5.skip_rate, cmap="RdYlGn_r", s=12)
    plt.colorbar(label="skip rate")
    plt.title("Preview-audio PCA coloured by skip rate")
    save("pca_preview")

    # 9. leakage-safe baseline: static features only, grouped by track
    feats = [c for v in cols.values() for c in v if c in d and pd.api.types.is_numeric_dtype(d[c])]
    X, y, g = d[feats], d.skipped, d.uri
    oof = np.zeros(len(d))
    models = []
    for tr, te in GroupKFold(5).split(X, y, g):
        m = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=250, l2_regularization=1.0, random_state=0).fit(X.iloc[tr], y.iloc[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
        models.append((m, te))
    auc = roc_auc_score(y, oof)
    dec = pd.DataFrame({"p": oof, "y": y}).assign(decile=lambda x: pd.qcut(x.p, 10, labels=False, duplicates="drop")).groupby("decile").agg(pred=("p", "mean"), actual=("y", "mean")).reset_index()
    section("9. Content-only baseline (static features, track-grouped 5-fold)", f"Out-of-fold AUC **{auc:.3f}** (0.5 = random) using {len(feats)} features, no artist skip history, no context.",
            "\nCalibration by risk decile:\n", md(dec))
    # permutation importance by feature group on the last fold
    m, te = models[-1]
    rows = []
    for gname, gc in cols.items():
        gc = [c for c in gc if c in feats]
        if not gc:
            continue
        Xp, drops = X.iloc[te].copy(), []
        for _ in range(3):
            Xs = Xp.copy()
            Xs[gc] = Xs[gc].sample(frac=1, random_state=_).values
            drops.append(roc_auc_score(y.iloc[te], m.predict_proba(Xp)[:, 1]) - roc_auc_score(y.iloc[te], m.predict_proba(Xs)[:, 1]))
        rows.append({"group": gname, "features": len(gc), "auc_drop_when_shuffled": float(np.mean(drops))})
    imp = pd.DataFrame(rows).sort_values("auc_drop_when_shuffled", ascending=False)
    section("9b. Which feature group carries the signal (permutation, last fold)", md(imp, ".4f"))
    imp.plot.barh(x="group", y="auc_drop_when_shuffled", legend=False, color=GREEN, figsize=(7, 4))
    plt.title("AUC drop when a feature group is shuffled")
    save("group_importance")

    # 10. cold-start check: AUC on tracks whose artist has no other plays
    only = d.groupby("artist_name").uri.transform("nunique") == 1
    if only.sum() > 200 and d.skipped[only].nunique() == 2:
        section("10. Cold-start slice (artists with a single track)", f"{only.sum()} plays; out-of-fold AUC on this slice: {roc_auc_score(y[only], oof[only]):.3f}.")

    section("11. Take-aways", "- Skips are front-loaded: the per-second curve is the core of the second model.",
            "- Compare group strengths in 4b/9b to decide which feature families deserve engineering effort.",
            "- Content-only AUC in section 9 is the realistic ceiling for a never-played song before adding artist/genre skip-rate priors.")
    (ROOT / "reports" / "deep_dive_full.md").write_text("\n".join(out), encoding="utf-8")
    print("wrote reports/deep_dive_full.md")


if __name__ == "__main__":
    main()

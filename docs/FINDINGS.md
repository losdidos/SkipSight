# Findings: which features matter (no model yet)

Source: `scripts/feature_analysis.py` -> `reports/feature_analysis.md` (full tables), `reports/feature_selection.csv`
(one verdict per feature), `reports/figures/features/`. 137 static song-level features, 2,854 tracks, 11,984 plays.
Only features known for a never-played song are tested. `duration_ms` is excluded because it is partly measured from the history.

## How features were judged
- **Whole-song relevance** (does it change whether I skip): weighted rank correlation with the track skip rate, 99% bootstrap interval over tracks, plus a
  shape-agnostic permutation test (5 quantile bins) that also catches U-shaped effects.
- **Timing relevance** (does it change when I skip): same tests among skips, against log(seconds played).
- **Redundancy**: |Spearman| >= 0.85 = near duplicate (drop the weaker); |rho| >= 0.6 = related (keep only the strongest as the shortlist).
- **Artist control**: correlation after removing each artist's average, to see if a feature is just an artist fingerprint.

## Result: 137 -> 32 shortlisted
| Verdict | Features |
|---|---|
| weak, no clear signal | 67 (40 preview, 8 title, 7 genre, 6 ReccoBeats, ...) |
| KEEP (51) -> shortlist 32 | whole-song 36, timing 10, both 5 |
| redundant duplicates | 13 (e.g. `release_decade`/`track_age_years` = `release_year`; `log_artist_fans` = `dz_artist_fans`; `explicit` = `dz_explicit`; `it_duration_ms` = `dz_duration_s`) |
| constant | 4 |
| < 30% coverage | 2 (`dz_bpm`, `rb_duration_ms`) |

Shortlist by source: preview 13, genre 7, Deezer 6, ReccoBeats 3 (`tempo`, `danceability`, `instrumentalness`), iTunes 1, other 1.
Most Spotify-style features (`valence`, `acousticness`, `liveness`, `speechiness`, `key`, `mode`) show no clear signal; Deezer's own `dz_bpm` is mostly empty.

## Tempo and skip timing (the main finding)
Share of **all plays** skipped inside each window, by ReccoBeats tempo quartile (edges 53 / 111 / 125 / 136 / 220 BPM):

| Tempo | Skip rate | Skipped in first 3 s | 3-10 s | after 10 s |
|---|---|---|---|---|
| Q1 slow (<111) | 12.3% | 4.4% | 3.4% | 4.4% |
| Q2 | 8.6% | 2.9% | 2.4% | 3.4% |
| Q3 | 8.9% | 3.3% | 2.4% | 3.2% |
| Q4 fast (>136) | 11.1% | **5.9%** | 1.9% | 3.3% |

- Fast songs are not skipped much more overall, but they are skipped **earlier**: the first-3-seconds rate is about 2x that of mid tempos, then falls below it.
- The effect is **U-shaped** (slow songs are also skipped more), which is why plain correlation reports about 0 for whole-song (-0.01) while the permutation test flags it.
- It holds **inside every genre** tested (fast minus rest, first-3-s rate): pop +3.7 pts, rap +2.0, rock +2.6, electronic +2.2, reggae/dancehall +1.6, indie +1.5, folk +3.0, R&B +0.6. So it is not just a genre effect.
- Other timing drivers: a high harmonic share (melodic rather than percussive) skips earlier (first 3 s: 5.5% vs 3.3% for the lowest quartile); newer releases are skipped later, mid-aged (2nd/3rd release-year quartile) songs earlier; high valence and longer tracks push skips later.
- Implication: the per-second model needs tempo, harmonic share, genre, duration and release year; the whole-song model needs genre, artist popularity and instrumentalness.

## Genre as a prior
Rarely played genres look extreme but have wide intervals (country 18%, 61 plays, 95% CI 10-30%; K-pop 18%, 55 plays; jazz/blues 25%, 60 plays).
Use shrunk rates `(skips + 50 x global) / (plays + 50)`: country 14.5%, K-pop 14.4%, jazz 18.3% vs rock/indie 14.6%, electronic 8.8%, reggae/dancehall 7.7%.
For a never-played genre the model should start from the global rate (10.3%) plus whatever the other features say.
K-pop and country skips are late (first-3-s rate 1.8% / 3.3% vs 4.9% pop) but mostly based on very few events.

## Cautions
- Effects are small (rank correlations mostly 0.05-0.15). Expect a modest ceiling (AUC about 0.6) from content features alone.
- Many shortlisted features act partly as artist fingerprints (e.g. `instrumentalness`, `g_rock`, `dz_duration_s` lose most of their effect within an artist); they help for known artists but less for cold-start ones.
- Preview features describe a 30 s excerpt from the middle of the song, not the intro.
- ReccoBeats tempo can be half/double-time for some songs; `pv_tempo` is a second opinion (it also carries timing signal).

## Figures

`python scripts/findings_figures.py` -> `reports/figures/findings/`:
1_feature_funnel (137 -> keep/drop), 2_tempo_early_vs_overall, 3_cumulative_skip_by_tempo (U-shape), 4_tempo_within_genre, 5_genre_skip_rate (raw vs shrunk), 6_harmonic_early_skip, 7_artist_fingerprint, 8_plays_vs_skip.

## Goal check: taste similarity and timing by genre

Script `scripts/taste_analysis.py` -> `reports/taste_analysis.md`, `reports/figures/taste/`.

- **Never-played song scored from similar songs (artist held out):** genre prior alone AUC 0.53, audio+genre neighbours 0.59 (actual skip rate climbs 6.4% -> 13.7% over score quintiles), known-artist prior 0.66. So the whole-song model should combine an artist prior (when the artist is known) with a neighbour/genre prior (when not). Ceiling for brand-new artists is roughly 0.6.
- **Skip timing by genre:** ~75-80% of skips of every genre happen in the first 10% of the song, so 10% bins hide the shape; the per-% model should work in seconds early on (0-3s, 3-10s, 10-30s) and coarse % blocks after. Genres differ in how early: rock/pop/rap/indie skips are ~41% under 3s (median 4.4s), reggae/folk/latin/classical are later (median 7-9s, 23-32% under 3s).
- **Link between the two models:** high-risk songs are skipped slightly earlier (median 4.6s vs 5.5s for low-risk).
- **Data support:** 56% of tracks have one play, 63% of artists have one track, 73% of tracks were never skipped. Per-track skip rates are noisy; rely on pooled priors, not single-track rates.
- **Less relevant for this goal:** within-play context (previous song, hour, position), per-track exact stats, and most MFCC columns. They are kept in the full CSV but should not feed either model.

## Trainability check

`scripts/trainability_check.py` -> `reports/trainability.md`, `reports/figures/trainability/` (diagnostic only, not the final model). Artist-grouped CV on the 32 shortlisted features: whole-song AUC 0.60 (shuffled labels 0.50); actual skip rate 5.4% -> 13.2% from safest to riskiest predicted quintile. Timing given a skip: R2 ~0 but rank correlation 0.14, so timing is only weakly predictable from content alone and is better modelled as a hazard curve shifted by song risk and tempo/harmonic share. Top whole-song features: duration, artist fans, mfcc5_std, instrumentalness, album size, rock. Top timing features: duration, spectral bandwidth, harmonic share, tempo.

## Intro audio check

Goal: per-second features for the first 30s (slow starts, early vocals). Test: loudness of the first 0.5s vs the whole clip for the 25 most-played tracks' Deezer previews (22 downloaded): median 0.84, only 14% open quietly, so the previews are mid-song excerpts, not intros. iTunes previews could not be decoded here (m4a, no ffmpeg) and are also mid-song clips in general. Conclusion: the current audio cannot support intro features. Options: user-supplied audio of the first 30s, or restrict to whole-clip features. Not built.

## Plan as agreed
1. Vibe score per track (engagement from share played, finishes, replays) as the training target; train it on content features, apply to unseen songs.
2. Skip-timing model uses vibe score + tempo, genre, harmonic share, duration.
3. Intro features only if intro audio becomes available.

## Vibe score: is there a better formula? (`scripts/vibe_score_compare.py`, `reports/vibe_score_comparison.md`)

Tested on a time split (does a score built from earlier plays predict later skips of the same tracks?). A fitted v2 (artist-pooled share listened + chosen-vs-auto start + replays) did not beat v1 and its weights flipped sign between windows. Scores from older history alone are at or below chance (0.41-0.51 AUC); adding recent plays brings v1 to 0.62. Interpretation: taste for a *specific* track drifts and songs you used to replay get skipped later (repeat fatigue). Decision: keep v1 as the training target (simple, interpretable); do not treat it as a stable per-track "like" label. Content-based generalisation (what the model learns across songs) is the useful part, not the per-track score itself. A time-decayed version (recent plays weigh more) is the next thing to test.

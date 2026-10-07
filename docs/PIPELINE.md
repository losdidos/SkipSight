# SkipSight data pipeline

Goal: one ML-ready dataset to predict **skip chance** for a song (whole-song and per second played), using only
features that can be obtained for a song that was never in the listening history.

## Run order

| # | Command | Output | Notes |
|---|---------|--------|-------|
| 1 | `python scripts/merge_history.py` | `spotify_history_full.csv` | merges the raw `Streaming_History_Audio_*.json` |
| 2 | `python scripts/enrich_reccobeats.py` | `data/processed/track_features.csv` | ReccoBeats audio features + ISRC + duration (resumable) |
| 3 | `python scripts/enrich_metadata.py --source deezer\|itunes\|musicbrainz` | `data/raw/{deezer,itunes,musicbrainz}.csv` | one request set per song (MusicBrainz: per artist); resumable; the three can run in parallel |
| 4 | `python scripts/analyse_previews.py` | `data/processed/track_audio_preview.csv` | own librosa analysis of the 30 s Deezer preview; resumable; mp3 not kept |
| 5 | `python scripts/build_dataset.py` | `tracks.csv`, `artists.csv`, `genres.csv`, `skipsight_dataset.csv` | in `data/processed/` |

Fetching happens **once per song**; everything afterwards is counted from the history.
Only title / artist / ISRC are sent to the public APIs (no keys, no account data).

## Scope
A track is kept when it has ReccoBeats audio features **or** at least 10 plays, and a known duration.
Result: **2,854 tracks, 11,984 plays**, skip rate 10.3 %.

## Definitions
- `skipped` = `reason_end == 'fwdbtn'` (Spotify's own flag is unreliable). `end_kind` = skip / finished (`trackdone`) / other (censored: logout, endplay, ...).
- `duration_ms` = longest fully finished play of the track (2,063 tracks); else ReccoBeats (643); else Deezer (148). See `duration_source`.
- Spotify `ts` is the END of a play, so `start_ts = ts - ms_played`.
- `pct_played = ms_played / duration_ms` (0-1).

## Sources and columns (prefix = source)
| Prefix | Source | Columns |
|--------|--------|---------|
| none (`tempo`, `energy`, `danceability`, `valence`, `acousticness`, `instrumentalness`, `liveness`, `speechiness`, `loudness`, `key`, `mode`, `rb_*`) | ReccoBeats | Spotify-style audio features, ISRC |
| `dz_` | Deezer | rank (popularity), release date, bpm (0 = unknown -> NaN), gain, explicit, contributors, track/disc position, album type / size / label / genres, artist fans, artist album count |
| `it_` | iTunes | primary genre, release date, explicit, track number/count, collection, price |
| `mb_`, `artist_begin_year` | MusicBrainz (per artist) | type, gender, country, area, career start, genre/tag lists with vote counts |
| `pv_` | own librosa analysis of the Deezer 30 s preview (excerpt from the middle of the song) | 20 MFCC mean/std, spectral centroid/bandwidth/rolloff/flatness/contrast, ZCR, RMS + dynamic range, onset strength/rate, tempo, beat regularity, 12 chroma means, harmonic ratio |
| derived | build_dataset.py | `release_year/decade`, `track_age_years`, `genre_text`, `genre_primary`, 15 `g_*` multi-hot genre buckets (merged from Deezer + iTunes + MusicBrainz), `t_*` title flags (remix, slowed, sped, feat, live, acoustic, version, length), `album_is_*`, `log_artist_fans`, `log_track_rank`, `explicit` |

## Tables
- **tracks.csv** - one row per song: all static features **plus counters** from the history: `plays, skips, finishes, ms_played_total, skip_rate, mean_pct_played, median_skip_ms`, skips per time block (`skips_0_3s, skips_3_10s, skips_10_30s, skips_30_ends`) and per 10 % block (`skips_pct_0..90`). Use these to build per-second skip curves (hazard) per song / genre / artist.
- **artists.csv / genres.csv** - tracks, plays, skips, finishes, skip rate.
- **skipsight_dataset.csv** - one row per play: `ms_played`, `pct_played`, `skipped`, `end_kind`, `reason_start/end`, the static song columns, and `ctx_*` columns (previous skip, hour, weekday, gap, shuffle) which are **not available when scoring a new song** - do not use them in the final model.

## Leakage rules
1. Whole-history counters (plays, skips, skip rates) are **not** in the per-play table because they include the play itself.
2. When training, artist / genre / similar-track skip rates must be rebuilt **out-of-fold** from the counters (group folds by track).
3. Static features are identical for all plays of a track (verified).
4. `reason_end`, `ms_played`, `pct_played` are targets / support for the per-second model, never inputs for a whole-song model.

## Coverage (share of tracks)
ReccoBeats 99.4 %, Deezer 98.5 %, iTunes 76.1 %, MusicBrainz (artist) 89.4 %, preview audio 95.1 %,
release year 99.2 %, genre 100 % (2.4 % "unknown"), Deezer bpm only 27.7 % (Deezer returns 0 for most tracks - use `tempo` / `pv_tempo`).

## Findings so far
See `docs/FINDINGS.md` and `reports/feature_analysis.md`. Skip rate varies by genre bucket (electronic 8.8 %, reggae/dancehall 7.5 % vs
rock/indie 14.7 %, country 18 % on few plays); loudness/energy correlate negatively with skipping, indie/alt and
long-career artists positively. Skips concentrate in the first 10 s (~65-70 %).
Small genres have few plays, so genre rates must be shrunk toward the global rate when modelled.

## Next (not done yet)
Retrain both models on these tables: (a) whole-song skip probability, (b) per-second skip hazard (discrete-time survival on `ms_played`
with `end_kind` censoring), with track-grouped cross-validation.

## One big file
- **skipsight_full.csv** - one row per play with everything: play info, all static song features (Deezer, iTunes, MusicBrainz, ReccoBeats, preview analysis, derived) and the whole-history counters as `hist_*` columns. The `hist_*` columns include the play itself, so drop them (or rebuild out-of-fold) before training. The other files are unchanged.

## Analysis scripts
| Command | Output |
|---|---|
| `python scripts/deep_dive_full.py` | `reports/deep_dive_full.md` + `reports/figures/full/` (quality, timing, correlations, era) |
| `python scripts/feature_analysis.py` | `reports/feature_analysis.md`, `reports/feature_selection.csv`, `reports/figures/features/` (redundancy, relevance, tempo x genre, genre priors) |

Findings are summarised in `docs/FINDINGS.md`. The earlier exploratory scripts and the first content model were removed; no model exists yet.

## Model-ready tables (`scripts/prepare_model_data.py`)

Only features available for a never-played song (the 32 shortlisted ones). No history counters, no context (previous track, hour, position). Intro audio is out of scope.

- `data/processed/model_vibe.csv` (2,854 tracks): target `vibe_score` (0-100) = 0.4 x rank of shrunk keep-rate (share of plays not skipped) + 0.3 x rank of shrunk mean share listened + 0.3 x rank of log(plays). Components are included (`vibe_keep_rate`, `vibe_pct_listened`, `vibe_replays`) for transparency; do not use them as inputs. Model 1 learns vibe_score from content features, then scores unseen songs.
- `data/processed/model_timing.csv` (11,241 plays, 1,072 skips): time-to-event table. `sec_played` + `skipped` (event; finishes and other ends are censored), `pct_played`, `track_len_s`, content features. `vibe_score_in_sample` is computed from the same history, so for training model 2 replace it with the out-of-fold vibe prediction from model 1.
- `fold` (0-4) groups by artist and is identical in both tables. Always cross-validate by fold so test songs mimic unseen ones.
- Known limits: 56% of tracks have 1 play, so vibe_score is rank-based and noisy for them (shrunk toward the mean).

### Vibe score fix (Rihanna check)
Rihanna's "40.6" was the median of her 21 tracks: her 5 big tracks scored 93-100 but 1-play tracks (e.g. Take A Bow) scored ~25 because each track was judged alone and only shrunk toward the global average. Fix in `prepare_model_data.py`: keep-rate and share-listened are now shrunk toward the artist's rate (excluding the track itself), and the replay part blends track plays with the artist's total plays. Result: Rihanna's mean 54 -> 63, lowest track 25 -> 42, one-play tracks ~45-49 (neutral-positive) instead of ~25. Scores remain ranks (mean 50 by construction), so "50" means average for your library, not "neutral".

### Vibe score weights (v1.2)
Play count and finishes now carry most of the weight: 0.35 replays (track + artist plays), 0.30 finishes (`trackdone` plays, track + artist), 0.20 keep-rate, 0.15 share listened. Added `vibe_finishes`. Rihanna's tracks now score 68-99 (mean 80). Weights are judgement calls, not fitted (the time-split test could not support fitted weights, see FINDINGS.md). Caveat: heavily played artists score high even on a track with several skips; the per-track skip signal is now a minor part.

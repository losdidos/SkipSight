# SkipSight – project plan (restart, v2)

Goal: predict for a song I pick (1) the chance I skip it and (2) the chance I skip it at each second.
Approach: first understand the data (no models), then build a **vibe score** from musical features, then the timing model, then the web UI.

Each phase ends with a short write-up in `docs/` so every decision is explained.

---

## Phase 0 – Clean up the project
- Move the old scripts to `archive/` (git keeps the history). Rebuild one clear pipeline, numbered in run order.
- `pipeline/` = scripts that produce data. `analysis/` = notebooks that explain findings (text + charts).
- One command rebuilds everything from the raw data, so the files and the code can't drift apart.
- **Replicable for anyone** (teacher requirement): put your own `Streaming_History_Audio_*.json` in `data/input/`, run one command, get the same tables.
  Nothing personal is hard-coded: birth year, time zone and paths live in `config.yaml`. `requirements.txt` has pinned versions. API caches make re-runs fast.

## Phase 1 – Clean data (correct labels, correct matches)
| Step | Why |
|---|---|
| 1.1 New skip label: ended by `fwdbtn` / `backbtn` / `endplay`, and stopped well before the end of the song | `fwdbtn` alone misses more than half the skips |
| 1.2 Context columns: `reason_start`, shuffle, previous song skipped, position in the skip streak | skips during a streak are about mood, not the song |
| 1.3 Validate every API match: length within ±5 s of the real played length + artist name matches; otherwise empty | wrong matches spread from ReccoBeats → Deezer |
| 1.4 MusicBrainz lookup by ISRC; earliest release date across sources | name-only matching mixes up artists; remasters change the year |
| 1.5 Whole-word genre/title matching | "dance" matched "dancehall", "live" matched "Oliver" |
| 1.6 Data dictionary (`docs/DATA.md`): every column, its source and its meaning | |

## Phase 2 – Understand the data (no models)
Rule: always look at song features on **clean plays** (the song came on by itself and the previous song wasn't skipped), or split by context. Otherwise streak skips hide the real effects.

1. **Context first:** skip rate by `reason_start`, shuffle, streak, hour and platform. How much of skipping is context rather than the song?
2. **Each feature against skipping:** skip rate per bin with a confidence interval, Spearman correlation, mutual information (also catches U-shaped effects).
3. **Era and nostalgia:** skip rate by release year and by *my age at release* (born 2006). Check the confounder: in 2019 I listened to songs that were new back then.
4. **Artist similarity:** Deezer's `/artist/{id}/related` (keyless; built from millions of users' listening). Do similar artists get similar skip rates? Then an unseen artist can borrow the skip rate of artists I know.
5. **Redundancy:** which features are near-duplicates of each other.
6. Write-up → `docs/FINDINGS.md` (rewritten).

## Phase 2b – My taste: like-score and music families (no supervised model)
1. **Like-score per song**, built only from behaviour (implicit feedback):
   - picked it on purpose (`clickrow` / `playbtn`), not autoplay
   - played on many *different days/months* (not 10 repeats in one evening)
   - finished on clean plays
   - came back to it after a long break
   - skipped on clean plays = dislike
2. **Families**, found three independent ways, then compared:
   - **Sound:** clustering on audio features (scaled; k-means / Gaussian mixture; a UMAP map to look at them)
   - **Artists:** a network of similar artists from Deezer related → communities
   - **My sessions:** songs I play in the same listening sessions belong together (my personal "moods"). Needs only the history.
3. Describe each family (tempo, energy, era, genres, example songs) and its average like-score.
4. A new song → nearest family → it inherits that family's like-score. This is the basis of the vibe score.

## Phase 3 – Vibe score
- Combine the features that survived Phase 2 into one score from 0 to 100.
- Version A: hand-chosen weights based on the correlations (simple and easy to explain).
- Version B: logistic regression = a weighted sum whose weights are *learned*. It's the same idea as A, with the weights fitted from the data.
- Compare them on unseen artists (folds grouped by artist) and pick one. Context stays fixed to one scenario: the song came on by itself.

## Phase 4 – Per-second model
Discrete-time hazard model: for each time slice, the chance of a skip given I'm still listening. Inputs: position in the song + vibe score + tempo/genre. The overall skip chance follows from this, so the two models agree.

## Phase 5 – Web UI
Search the catalogue (songs with complete features) → overall skip % → a per-second skip % bar above the play bar.

---

## Decision log
| Date | Decision | Why |
|---|---|---|
| 2026-10-07 | Old vibe score dropped | Mostly measured play counts (exposure); did not predict later skips |
| 2026-10-07 | Context is kept and fixed to one scenario at prediction time, not dropped | Dropping it doesn't remove its effect |
| 2026-10-07 | Analysis before models | Understand the data before building scores on it |
| 2026-10-07 | Phase 0 done; 1.1 + 1.2 done (`pipeline/02_label_plays.py`) | New label: 2,808 skips (23.9 %) vs 1,244 with fwdbtn only; agrees with Spotify's flag on 95 % |
| 2026-10-07 | Song length is measured from my own finished plays (median), not from an API | It's the ground truth, and step 1.3 needs it to check API matches |
| 2026-10-07 | Back button + same song next = restart, not a skip | Replaying a song means I like it |
| 2026-10-07 | `clean_play` = `reason_start = trackdone` | Skip rate 5.4 % on clean plays vs 66 % on the others; after 3+ skips in a row it's 91 %: context is huge |

| 2026-10-07 | Labelling moved after enrichment (02 tracks → 03–05 fetch → 06 match → 07 label) | The label needs the song length; songs never finished get it from the APIs |
| 2026-10-07 | Reference length = Spotify's length for the exact id (via ReccoBeats), else my history | History matches Spotify to the second for 91 % of songs, but when it differs it's too short (scrubbing) |
| 2026-10-07 | Fetch raw API answers once (cache), decide matches offline (step 06) | Match rules can change without new requests |
| 2026-10-07 | No official Spotify API | Feb 2026: Premium required, bulk lookups and popularity removed, features gone since 2024. Small gain; teacher would need Premium |
| 2026-10-07 | iTunes dropped | No ISRC lookup (search only = error-prone), slow rate limit; genre/date covered by Deezer + MusicBrainz |
| 2026-10-07 | Deezer: plain-text search | Advanced syntax `artist:"" track:""` returned 0 results for 880 of 881 queries |

## Open questions (for Phase 2)
- Songs I **chose** myself (`clickrow`/`playbtn`) get skipped 63 % of the time. Is that browsing (clicking through a playlist), which isn't a real dislike?

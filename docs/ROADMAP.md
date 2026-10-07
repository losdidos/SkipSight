# SkipSight – Roadmap & results

Where the project stands, what each finished step produced, and what comes next.
The *why* behind each phase is in [PLAN.md](PLAN.md). Every column is described in [DATA.md](DATA.md).

## Status

| Phase | Step | Status |
|---|---|---|
| 0 | Clean project structure, config, one-command pipeline | ✅ done |
| 1.1 | New skip label | ✅ done |
| 1.2 | Context columns (sessions, streaks, clean plays) | ✅ done |
| 1.3 | Validate API matches (length ±5 s + artist) | ✅ done |
| 1.4 | MusicBrainz by ISRC, earliest release date | ✅ done |
| 1.5 | Exact genre / whole-word title matching | ✅ done |
| 1.6 | Data dictionary ([DATA.md](DATA.md)) | ✅ done (kept up to date) |
| 2 | Understand the data (correlations, no models) | ⏳ next |
| 2b | My taste: like-score + music families | ⬜ |
| 3 | Vibe score | ⬜ |
| 4 | Per-second skip model | ⬜ |
| 5 | Web UI | ⬜ |

---

## Phase 0 – Project structure (done 2026-10-07)

**What changed**
- The old scripts and reports moved to `archive/`. Git keeps their history.
- The data is now built by numbered scripts in `pipeline/`, run in order by `python run_pipeline.py`.
- `config.yaml` holds everything personal (birth year, time zone, folders, thresholds). Anyone can run the project on their own Spotify export without changing code.
- `requirements.txt` pins exact package versions.

**Why:** the old data files and code had drifted apart. Now every table can be rebuilt from the raw export with one command, so they always match the code.

---

## Phase 1.1 – Skip label (done 2026-10-07)

**Old label:** a skip only if the play ended with the "next" button (`fwdbtn`).
This missed skips made by clicking another song (`endplay`) or pressing back (`backbtn`).

**New label:** a play is a skip if *I* stopped it (`fwdbtn`, `backbtn` or `endplay`) **and** it stopped well before the end (< 90 % played and > 15 s left).

Song length comes from my own history: the median `ms_played` of the plays that ran to the end. This is known for **73 %** of the 3,448 songs. Enrichment fills in the rest later.

### Results (12,885 plays)

| Label | Skips | Skip rate |
|---|---|---|
| Old (`fwdbtn` only) | 1,244 | 10.6 % |
| **New** | **2,808** | **23.9 %** |
| Spotify's own `skipped` flag | 2,421 | 20.6 % |

The old label missed more than half of the real skips.

### Which rule decided each label

| Rule | Plays | Label |
|---|---|---|
| finished (`trackdone`) | 8,843 | not a skip |
| user stopped early | 1,995 | skip |
| stopped in first 30 s (length unknown) | 813 | skip |
| user stopped near the end | 99 | not a skip |
| restarted (back button, same song again) | 5 | not a skip |
| not a choice (logout, crash, error, remote) | 880 | unknown |
| needs length (length unknown, > 30 s played) | 250 | unknown → after enrichment |

The label is known for **91 %** of plays.

### Check against Spotify's flag
The new label agrees with Spotify's `skipped` flag on **95.1 %** of plays. Where they differ:
- **Near the end:** Spotify calls a "next" press in the last seconds a skip (92 plays). We don't, because the song was basically finished.
- **Early stops:** Spotify sometimes doesn't flag a clear early stop (~480 plays). We do.

**Decision:** keep our own label. Spotify doesn't document its rule; ours is written down and can be checked.

---

## Phase 1.2 – Context (done 2026-10-07)

New columns:
- `session_id`: a new session starts after 30 min of silence.
- `pos_in_session`: the song's position within its session.
- `prev_skip`: whether the song before it was skipped.
- `skip_streak`: how many skips in a row came right before it.
- `chosen`: I picked the song myself.
- `clean_play`: the song came on by itself after the previous one finished.
- `hour`, `weekday`: local time.

Detail worth knowing: Spotify's `ts` is when a play **ended**, so start time = `ts − ms_played`.

### Results: context matters a lot

| Situation | Skip rate |
|---|---|
| **Clean play** (came on by itself after a finished song) | **5.4 %** |
| Not a clean play | 66.4 % |
| Previous song was **not** skipped | 9.7 % |
| Previous song **was** skipped | 74.9 % |
| Skip streak 0 | 9.7 % |
| Skip streak 1 | 46.3 % |
| Skip streak 2 | 70.1 % |
| Skip streak 3+ | 91.0 % |
| Not chosen by me | 17.0 % |
| **Chosen by me** (`clickrow` / `playbtn`) | **62.7 %** |

**What this means:** whether I skip depends heavily on the *situation*, not only on the song. When I'm in a skip streak I'm hunting for something, and almost everything gets skipped. If we analysed all plays together, those streaks would hide what I actually think of a song.

**Decision:** Phase 2 analyses song features on **clean plays**, or splits results by context. The models keep context as input and fix it to "came on by itself" when they predict (see PLAN.md decision log).

---

## Open questions

| Question | Where it gets answered |
|---|---|
| Why are songs I **chose myself** skipped 63 % of the time? Probably browsing (clicking through a playlist), which wouldn't be a real dislike. | Phase 2.1 |
| Is 30 minutes of silence the right session gap? | Phase 2.1 (check the distribution of gaps) |
| The 250 "needs length" plays | after step 1.3 (enrichment gives the length) |

## Known data problems & research (2026-10-07)

### Should we drop songs I never finished? → No
Most were played once (754 of 923), but together they hold **29 % of all skips** (813 of 2,808).
Dropping them would delete a third of the "I don't like this" examples. Skip rates would look lower
than they are, and a model would barely see what I dislike. This is **selection bias**: removing rows
because of the outcome you want to predict. Their label still works (stopped in the first 30 s, or
Spotify's length from ReccoBeats), so they stay.

### Missing audio features (77 % of songs, 86 % of plays covered)
- Coverage grows with play count: 73 % of songs played once, 90 % of songs played 10+ times.
- **Is the gap biased?** On clean plays, songs without features are skipped 4.4 % vs 5.5 % with
  features; on all plays 22.9 % vs 24.0 %. Small difference, so analysing only songs with features
  shouldn't distort the results much. (Recheck after the final match.)
- **Possible fix:** compute features ourselves from Deezer's 30 s previews.
  - `librosa`: tempo, loudness, brightness, onset rate. Works on Windows.
  - Essentia's pretrained models (danceability, arousal/valence, style): closer to Spotify-style
    features, but only installs on Linux/macOS (WSL or Docker on Windows).
  - Decide in Phase 2, only if the missing songs turn out to matter.

### ReccoBeats tempo: octave errors
Compared with Deezer's BPM on 701 songs: **87 % agree (±3 %), 11 % are exactly ½ or 2×**.
Tempo detectors often hear the half or double beat. ReccoBeats doesn't publish how its features are
computed, so our own cross-check is the only evidence we have. Fix for Phase 2: *fold* tempo into one
octave (e.g. 70–140 BPM) so 70 and 140 count as the same tempo, and check whether Deezer's BPM agrees.

### Song length from my history can be too short
Matches Spotify's length (±1 s) for 91 % of songs. When it differs, it's almost always shorter
(*S&M* by Rihanna: 162 s measured, 243 s real). That's because scrubbing forward still ends as `trackdone`.
**Fixed:** Spotify's length for the exact id (via ReccoBeats) is used first.

### Other
- Only the album's main artist is in the export (featured artists are lost, compilations show
  "Various Artists"). Deezer and MusicBrainz give the full artist list → use them for artist similarity.
- Genres: Deezer = album genre (broad), MusicBrainz = artist genre (not per song). Usefulness is measured in Phase 2.

---

## Phase 1.3–1.5 – Enrichment, validated (done 2026-10-07)

**New pipeline order:** `01 merge → 02 tracks → 03 ReccoBeats → 04 Deezer → 05 MusicBrainz → 06 match → 07 label → 08 build`.
Steps 03–05 save the raw API answers (cache). Step 06 decides offline which answers really are my song,
so match rules can be tuned and re-run in seconds.

**Match rules:** a lookup by exact id or ISRC is trusted if the length doesn't disagree. A search result
needs the same title + artist + length within 5 s. Why all three: there are many songs called "Wolves",
and James Arthur's "Wolves" is only 2.5 s longer than Selena Gomez's.

### Results (3,448 songs)
| Source | Matched | How |
|---|---|---|
| ReccoBeats (audio features) | 2,666 (77 %), **86 % of plays** | 2,572 by exact Spotify id, 94 by validated search |
| Deezer (BPM, genres, popularity, related artists) | 3,031 (88 %) | 2,502 ISRC, 512 search, 17 ISRC alias |
| MusicBrainz (first release date, artist genres) | 1,673 (49 %) | by ISRC only (no name guessing) |
| Song length | 3,237 (94 %) | Spotify's length for the id (2,572), else my history (665) |
| Release year | 88 % | MusicBrainz 1,567, Deezer track 1,448, Deezer album 30 |

### Things we found and fixed on the way
- **Deezer's advanced search was broken:** `artist:"..." track:"..."` returned results for 1 of 881 queries
  (the old pipeline used it too). Plain-text search + our own checks: Deezer coverage 68 % → 88 %.
- **My history length can be too short** (scrubbing), so Spotify's length for the exact id is now the reference.
  This also improved the skip label: **2,960 skips (24.8 %)**, up from 2,808. Spotify's length makes more
  stops measurable, so fewer plays fall back to the "first 30 s" rule (813 → 177).
- **Artist names differ between catalogues:** `Ke$ha`/`Kesha`, `Ms. Lauryn Hill`/`Lauryn Hill`,
  `Olly Alexander (Years & Years)`. Handled; `Drake` ≠ `Drake Bell` and `Lady Gaga` ≠ `Bradley Cooper` still rejected.
- **Genres are now exact names from lists** (59 genres on ≥ 30 songs, e.g. `dance_pop`, `cloud_rap`), not
  substring matches. Title flags use whole words: 349 *slowed*, 97 *sped up*, 124 *remix*, 295 *feat* songs.

### How wrong was the old enrichment?
| Check | Result |
|---|---|
| Old Deezer matches pointing to the same song as now | 93 % of 2,810 |
| Old Deezer matches whose length is off by > 5 s (wrong version/song) | 59 (2.1 %) |
| Old Deezer matches rejected or not found now | 134 |
| Old release year later than the true first release (remaster/compilation) | 12 % (3 % by 5+ years) |
| Old ReccoBeats name-matches, never validated | 268 |

So the old data was mostly right, but about 5–10 % of songs had a wrong match or a wrong year. The old
code also had no record of *how* each match was made, so the errors couldn't be found. Now every song
has its `*_match` column.

## Next: Phase 2 – Understand the data (no models)
First notebook: context and skipping, then each feature against skipping on clean plays.
Start with the open questions above (chosen songs skipped 64 %, tempo folding, missing features).

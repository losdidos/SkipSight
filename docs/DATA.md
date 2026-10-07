# Data dictionary

Every table the pipeline makes, where its columns come from and what they mean.
Updated whenever a pipeline step changes. Run order: `python run_pipeline.py` (steps 01 → 08).

| Step | Script | Makes |
|---|---|---|
| 01 | `01_merge_history.py` | `interim/plays.csv` |
| 02 | `02_tracks.py` | `interim/tracks.csv` |
| 03 | `03_fetch_reccobeats.py` | `cache/reccobeats_*.jsonl` |
| 04 | `04_fetch_deezer.py` | `cache/deezer_*.jsonl` |
| 05 | `05_fetch_musicbrainz.py` | `cache/musicbrainz_*.jsonl` |
| 06 | `06_match_tracks.py` | `interim/tracks_enriched.csv` |
| 07 | `07_label_plays.py` | `interim/plays_labeled.csv` |
| 08 | `08_build_dataset.py` | `output/tracks.csv`, `output/plays.csv` ← **use these for analysis** |

The cache holds raw API answers (one JSON line per question). Steps 03–05 only ask what's not cached yet.

---

## `output/plays.csv` – one row per play

### From the Spotify export (step 01)
| Column | Meaning |
|---|---|
| `play_id` | row number, 0 = oldest play |
| `ts` | UTC time the play **ended** (Spotify convention) |
| `platform` | device / OS |
| `ms_played` | how long it played (ms) |
| `track_id` | Spotify track id |
| `track_name`, `artist_name`, `album_name` | as Spotify names them (`artist_name` = album's main artist) |
| `reason_start` | why it started: `trackdone` (previous song ended), `fwdbtn`, `backbtn`, `clickrow`, `playbtn`, `appload`, ... |
| `reason_end` | why it ended: `trackdone`, `fwdbtn`, `backbtn`, `endplay`, `logout`, `unexpected-exit`, ... |
| `shuffle`, `offline`, `incognito_mode` | as in the export |
| `skipped` | Spotify's own skip flag (undocumented rule; only used as a cross-check) |

### Label (step 07)
| Column | Meaning |
|---|---|
| `length_ms`, `length_source` | song length used: `spotify_id` (Spotify's length for this exact id) > `history` (median of my finished plays) |
| `fraction_played` | `ms_played / length_ms`, capped at 1 |
| `skip` | **the label**: 1 = skip, 0 = not a skip, empty = can't tell |
| `label_rule` | which rule decided `skip` (table below) |

| `label_rule` | Condition (first match wins) | `skip` |
|---|---|---|
| `finished` | `reason_end = trackdone` | 0 |
| `restarted` | ended with back button and the next play is the same song | 0 |
| `user_stopped_early` | ended by me (`fwdbtn`/`backbtn`/`endplay`), < 90 % played and > 15 s left | 1 |
| `user_stopped_near_end` | ended by me, but close to the end | 0 |
| `stopped_in_first_30s` | ended by me, length unknown, < 30 s played | 1 |
| `no_length` | ended by me, length unknown, > 30 s played | empty |
| `not_a_choice` | logout, crash, error, remote, unknown | empty |

### Context (step 07)
| Column | Meaning |
|---|---|
| `start_local`, `hour`, `weekday` | start time in the local time zone (config); weekday 0 = Monday |
| `session_id`, `pos_in_session` | listening session (new one after 30 min of silence); 1 = first song |
| `prev_skip` | previous song in this session was skipped (unknown counts as no) |
| `skip_streak` | number of skips in a row right before this play |
| `came_on_by_itself` / `clean_play` | `reason_start = trackdone`: the fairest test of "do I like this song?" |
| `chosen` | I picked it (`clickrow` / `playbtn`) |

### Age (step 08)
| Column | Meaning |
|---|---|
| `my_age_at_play` | my age in the year of the play (config birth year) |
| `song_age_at_play` | years between the song's release and this play (0 = it was new) |

---

## `output/tracks.csv` – one row per song

### From my history (step 02)
| Column | Meaning |
|---|---|
| `n_plays`, `n_finished` | plays, and plays that ran to the end |
| `length_ms_history` | median `ms_played` of the finished plays |

### Match results (step 06): how each source was matched, or why it was rejected
| Column | Values |
|---|---|
| `rb_match` | `id` (exact Spotify id), `search` (same title + artist + length ±5 s), `not_found`, `rejected:length`, `rejected:no_length_to_check` |
| `dz_match` | `isrc`, `isrc_alias` (same ISRC + title + length, other artist name), `search`, `not_found`, `rejected:*` |
| `mb_match` | `isrc`, `not_found`, `rejected:artist_or_length` |
| `rb_length_agrees` | Spotify's length vs my measured length within ±5 s (a check of my measurement) |
| `length_ms`, `length_source` | final song length and where it came from |

### ReccoBeats (`rb_*` + audio features)
`rb_id` (Spotify id the features belong to), `rb_isrc`, `rb_length_ms`, `has_features`, and
`tempo`, `energy`, `danceability`, `valence`, `acousticness`, `instrumentalness`, `liveness`,
`speechiness`, `loudness`, `key`, `mode` (Spotify-style audio features, 0–1 except tempo/loudness/key/mode).

### Deezer (`dz_*`)
`dz_id`, `dz_isrc`, `dz_length_ms`, `dz_bpm`, `dz_gain` (loudness), `dz_rank` (popularity on Deezer),
`dz_explicit`, `dz_release_date`, `dz_artist_id`, `dz_artist_fans`, `dz_album_type` (album/single/compile/ep),
`dz_album_release_date`, `dz_album_genres` (`|`-separated).

### MusicBrainz (`mb_*`)
`mb_first_release_date` (earliest release of this recording anywhere), `mb_artist_id`, `mb_artist_type`
(Person/Group), `mb_artist_country`, `mb_genres`, `mb_tags` (top 8 by user votes, `|`-separated).

### Derived (steps 06 + 08)
| Column | Meaning |
|---|---|
| `release_date`, `release_date_source`, `release_year` | earliest of MusicBrainz / Deezer track / Deezer album dates |
| `age_at_release` | my age when the song came out (negative = before I was born) |
| `tempo_folded` | tempo moved into [70, 140) BPM, so half/double-tempo errors count as the same tempo |
| `tempo_check` | ReccoBeats tempo vs Deezer BPM: `same`, `octave` (2× or ½), `different`, `unknown` |
| `genres` | all genre names of the song (Deezer album + MusicBrainz artist), `|`-separated |
| `genre_*` | 1 if the song has that exact genre (only genres on ≥ 30 songs) |
| `title_*` | version words in the title, whole-word match: `remix`, `sped_up`, `slowed`, `live`, `acoustic`, `remaster`, `feat`, `edit` |

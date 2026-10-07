"""Step 07 - Give every play a skip label and its listening context.

Input : data/interim/plays.csv, data/interim/tracks_enriched.csv (for the song length)
Output: data/interim/plays_labeled.csv   (every play + label + context columns)

Thresholds and the time zone come from config.yaml.
This step runs after the API steps, because the label needs the song's length: measured from my
own finished plays when possible (step 02), otherwise Spotify's length from the APIs (step 06).

1. SKIP LABEL  (column `skip`: 1 = skip, 0 = not a skip, empty = can't tell)
   Rule, in order (the column `label_rule` says which rule decided):
   - reason_end = trackdone                              -> 0  finished
   - user ended it, next play is the same song            -> 0  restarted (pressed back to replay it)
   - user ended it, length known, stopped well before end -> 1  user_stopped_early
   - user ended it, length known, close to the end       -> 0  user_stopped_near_end
   - user ended it, length unknown, within 30 s           -> 1  stopped_in_first_30s (no song is that short)
   - user ended it, length unknown, after 30 s            -> ?  no_length
   - anything else (logout, app crash, error, ...)        -> ?  not_a_choice
   "User ended it" = reason_end in config skip_label.user_end_reasons.

2. CONTEXT  (skips are not only about the song: the situation matters too)
   Note: Spotify's `ts` is when the play ENDED, so start = ts - ms_played.
   - session_id     : a new session starts after `gap_minutes` of silence
   - pos_in_session : 1 = first song of the session
   - prev_skip      : was the previous song in this session skipped?
   - skip_streak    : how many skips in a row came right before this play
   - came_on_by_itself (reason_start = trackdone) : the previous song finished and this one followed
   - chosen         : I picked it myself (clickrow / playbtn)
   - clean_play     : came on by itself = the fairest test of "do I like this song?"
   - hour, weekday  : local time (config listener.timezone)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import CONFIG, data_path

RULES = CONFIG["skip_label"]
USER_END = set(RULES["user_end_reasons"])


def add_skip_label(plays: pd.DataFrame) -> pd.DataFrame:
    user_ended = plays["reason_end"].isin(USER_END)
    length = plays["length_ms"]
    seconds_left = (length - plays["ms_played"]) / 1000
    fraction = plays["ms_played"] / length
    restarted = plays["track_id"].eq(plays["track_id"].shift(-1)) & plays["reason_end"].eq("backbtn")
    early = (fraction < RULES["max_fraction_played"]) & (seconds_left > RULES["min_seconds_left"])

    conditions = [
        plays["reason_end"].eq("trackdone"),
        user_ended & restarted,
        user_ended & length.notna() & early,
        user_ended & length.notna(),
        user_ended & (plays["ms_played"] < 30_000),
        user_ended,
    ]
    rules = ["finished", "restarted", "user_stopped_early", "user_stopped_near_end",
             "stopped_in_first_30s", "no_length"]
    values = [0, 0, 1, 0, 1, np.nan]

    plays["fraction_played"] = fraction.clip(upper=1).round(3)
    plays["label_rule"] = np.select(conditions, rules, default="not_a_choice")
    plays["skip"] = np.select(conditions, values, default=np.nan)
    return plays


def add_context(plays: pd.DataFrame) -> pd.DataFrame:
    end = pd.to_datetime(plays["ts"], utc=True)
    start = end - pd.to_timedelta(plays["ms_played"], unit="ms")
    local = start.dt.tz_convert(CONFIG["listener"]["timezone"])
    plays["start_local"] = local.dt.tz_localize(None)
    plays["hour"] = local.dt.hour
    plays["weekday"] = local.dt.dayofweek  # 0 = Monday

    silence = start - end.shift(1)
    new_session = silence.isna() | (silence > pd.Timedelta(minutes=CONFIG["sessions"]["gap_minutes"]))
    plays["session_id"] = new_session.cumsum()
    plays["pos_in_session"] = plays.groupby("session_id").cumcount() + 1

    # Unknown labels count as "not a skip" for the context columns only
    prev = plays.groupby("session_id")["skip"].shift(1).fillna(0)
    plays["prev_skip"] = prev.astype(int)
    run_block = (prev == 0).groupby(plays["session_id"]).cumsum()
    plays["skip_streak"] = prev.groupby([plays["session_id"], run_block]).cumsum().astype(int)

    plays["came_on_by_itself"] = plays["reason_start"].eq("trackdone")
    plays["chosen"] = plays["reason_start"].isin(["clickrow", "playbtn"])
    plays["clean_play"] = plays["came_on_by_itself"]
    return plays


def report(plays: pd.DataFrame) -> None:
    known = plays["skip"].notna()
    print(f"{len(plays):,} plays, label known for {known.sum():,} ({known.mean():.0%})")
    print("\nWhich rule decided the label:")
    print(plays["label_rule"].value_counts().to_string())

    print("\nOld label (fwdbtn only) vs new label vs Spotify's own 'skipped' flag, on plays with a known label:")
    k = plays[known]
    print(f"  old  fwdbtn only : {(k['reason_end'] == 'fwdbtn').sum():>7,} skips ({(k['reason_end'] == 'fwdbtn').mean():.1%})")
    print(f"  new  label       : {int(k['skip'].sum()):>7,} skips ({k['skip'].mean():.1%})")
    flag = k["skipped"].astype("boolean")
    print(f"  Spotify skipped  : {int(flag.sum()):>7,} skips ({flag.mean():.1%})  [missing in {flag.isna().mean():.0%} of rows]")
    agree = (k["skip"] == flag.astype(float))[flag.notna()].mean()
    print(f"  new label agrees with Spotify's flag on {agree:.1%} of the rows where the flag exists")

    print("\nSkip rate by context (why we analyse clean plays separately):")
    for column in ["clean_play", "chosen", "prev_skip"]:
        print(f"  {column:<12}", k.groupby(column)["skip"].mean().round(3).to_dict())
    streak = k["skip_streak"].clip(upper=3)
    print("  skip_streak (3 = 3+)", k.groupby(streak)["skip"].mean().round(3).to_dict())


def main() -> None:
    plays = pd.read_csv(data_path("interim", "plays.csv"))
    tracks = pd.read_csv(data_path("interim", "tracks_enriched.csv"), usecols=["track_id", "length_ms", "length_source"])
    plays = plays.merge(tracks, on="track_id", how="left")
    plays = add_context(add_skip_label(plays))

    plays.to_csv(data_path("interim", "plays_labeled.csv"), index=False)
    print("song length used for the label:", plays["length_source"].fillna("unknown").value_counts().to_dict(), "\n")
    report(plays)


if __name__ == "__main__":
    main()

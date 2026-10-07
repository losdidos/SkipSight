# SkipSight

Predicts, from your own Spotify history, (1) the chance you skip a song and (2) the chance you skip it at each second.
The plan and every decision behind it: [docs/PLAN.md](docs/PLAN.md). Progress and results so far: [docs/ROADMAP.md](docs/ROADMAP.md). What every column means: [docs/DATA.md](docs/DATA.md).

## Run it on your own data
1. Request **Extended streaming history** at spotify.com/account/privacy (takes a few days).
2. Put the `Streaming_History_Audio_*.json` files in `data/input/`.
3. Set your birth year and time zone in `config.yaml`.
4. Install and run (Python 3.13):
   ```
   python -m venv .venv
   .venv\Scripts\activate        # macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   python run_pipeline.py
   ```

## Layout
| Folder | What's in it |
|---|---|
| `pipeline/` | numbered scripts that build the data, run in order by `run_pipeline.py` |
| `analysis/` | Jupyter notebooks: the data analysis, with explanations and charts |
| `data/` | everything generated (private, not in git): `input/`, `interim/`, `cache/`, `output/` |
| `docs/` | plan, data dictionary, findings |
| `archive/` | the first version of the project, kept for reference only |

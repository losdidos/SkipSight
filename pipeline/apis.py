"""Shared helpers for the API steps: polite HTTP requests and a raw-answer cache.

The cache stores the API's raw answer (JSON) for every question we asked, one line per question,
in data/cache/<name>.jsonl. Why:
- Re-running the pipeline never asks an API the same question twice (fast, and polite to free APIs).
- The match rules (step 06) work on these raw answers, so changing a rule needs no new requests.
- "Asked, but nothing found" is stored too (as null), so misses are not retried every run.
  Network errors are NOT stored, so they are retried on the next run.
"""

from __future__ import annotations

import json
import time

import requests

from common import data_path

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "SkipSight/0.3 (school research project)"


class Cache:
    def __init__(self, name: str):
        self.path = data_path("cache", f"{name}.jsonl")
        self.data: dict = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # half-written line from an interrupted run: it will be fetched again
                    self.data[entry["key"]] = entry["value"]

    def __contains__(self, key) -> bool:
        return str(key) in self.data

    def get(self, key):
        return self.data.get(str(key))

    def put(self, key, value) -> None:
        self.data[str(key)] = value
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": str(key), "value": value}, ensure_ascii=False) + "\n")


def get_json(url: str, params: dict | None = None, pause: float = 0.2):
    """GET a JSON answer. Waits and retries on rate limits and network errors.

    Returns the parsed JSON, or None when the API says "not found".
    Raises after 5 failed attempts, so a broken API stops the run instead of silently caching misses.
    """
    for attempt in range(5):
        try:
            response = SESSION.get(url, params=params, timeout=30)
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
            continue
        time.sleep(pause)
        if response.status_code in (429, 500, 502, 503, 504):
            time.sleep(10 * (attempt + 1))
            continue
        if response.status_code in (400, 404):
            return None
        response.raise_for_status()
        data = response.json()
        # Deezer answers 200 with an "error" object; code 4 = "quota exceeded, slow down"
        if isinstance(data, dict) and isinstance(data.get("error"), dict):
            if data["error"].get("code") == 4:
                time.sleep(5 * (attempt + 1))
                continue
            return None
        return data
    raise RuntimeError(f"giving up on {url} {params}")


def progress(name: str, done: int, total: int, every: int = 200) -> None:
    if done % every == 0 or done == total:
        print(f"  {name}: {done:,}/{total:,}", flush=True)

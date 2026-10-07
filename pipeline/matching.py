"""Rules that decide whether an API result is really the song from my history.

Used by the fetch steps (a loose first filter) and by step 06 (the final decision).

Why three checks:
- Title alone is not enough: there are many songs called "Wolves".
- Length alone is not enough: James Arthur's "Wolves" is only 2.5 s longer than Selena Gomez's.
- Title + artist + length (within 5 s) together are very unlikely to match by accident.
"""

from __future__ import annotations

import re
import unicodedata

LENGTH_TOLERANCE_MS = 5_000

# Bracketed or " - " parts that name the SAME recording in a different package.
# Remix / acoustic / live / sped up are NOT removed: those are different recordings.
# \b = whole words only, so "mono" doesn't match "Monologue".
_PACKAGING = (r"\b(?:feat|ft|featuring|with|remaster(?:ed)?|\d{4} remaster|single version|album version"
              r"|radio edit|mono|stereo|explicit|bonus track|deluxe|from)\b")


def norm(text) -> str:
    """Lowercase, no accents, no punctuation: 'Beyoncé – Halo!' -> 'beyonce halo'.

    Works for any alphabet (Korean, Japanese, ... titles stay non-empty).
    """
    text = "".join(c for c in unicodedata.normalize("NFKD", str(text)) if not unicodedata.combining(c))
    text = text.casefold().replace("&", " and ").replace("$", "s")  # Ke$ha -> kesha
    return " ".join(re.findall(r"\w+", text))


def clean_title(title) -> str:
    title = str(title)
    title = re.sub(rf"\s*[\(\[][^\)\]]*({_PACKAGING})[^\)\]]*[\)\]]", "", title, flags=re.I)
    title = re.sub(rf"\s+-\s+[^-]*({_PACKAGING}).*$", "", title, flags=re.I)
    return norm(title)


def search_text(artist, title) -> str:
    """Plain 'artist title' search text. (Deezer's advanced syntax artist:"..." track:"..."
    returned no results at all in 2026, so plain text + our own strict checks are used.)"""
    return f"{artist} {clean_title(title)}"


def artist_names(name) -> set[str]:
    """All the names hidden in one artist string: 'Selena Gomez & Marshmello' ->
    {'selena gomez and marshmello', 'selena gomez', 'marshmello'}."""
    name = str(name)
    without_brackets = re.sub(r"\s*[\(\[].*?[\)\]]", "", name)  # 'Olly Alexander (Years & Years)'
    parts = re.split(r"\s*(?:,|&|;|/|\bfeat\.?|\bft\.?|\bfeaturing\b|\bwith\b|\band\b|\bx\b)\s*", name, flags=re.I)
    return {n for n in [norm(name), norm(without_brackets)] + [norm(p) for p in parts] if n}


def _same_name(a: str, b: str) -> bool:
    """Equal, or the shorter name (2+ words) is fully inside the longer: 'lauryn hill' ~ 'ms lauryn hill'."""
    if a == b:
        return True
    short, long = sorted([a.split(), b.split()], key=len)
    return len(short) >= 2 and set(short) <= set(long)


def artist_ok(spotify_artist, candidate_artists: list) -> bool:
    if norm(spotify_artist) == "various artists":
        return True  # compilation: Spotify gives no real artist, so we can't check it
    wanted = artist_names(spotify_artist)
    found = set().union(*(artist_names(a) for a in candidate_artists)) if candidate_artists else set()
    return any(_same_name(w, f) for w in wanted for f in found)


def title_ok(spotify_title, candidate_title) -> bool:
    return clean_title(spotify_title) == clean_title(candidate_title)


def length_ok(reference_ms, candidate_ms) -> bool | None:
    """True/False, or None when one of the two lengths is unknown (can't check)."""
    if reference_ms is None or candidate_ms is None or reference_ms != reference_ms or candidate_ms != candidate_ms:
        return None
    return abs(float(reference_ms) - float(candidate_ms)) <= LENGTH_TOLERANCE_MS

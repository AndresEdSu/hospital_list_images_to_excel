from __future__ import annotations

import re
import unicodedata


SPACE_RE = re.compile(r"\s+")


def strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def normalize_text(value: str) -> str:
    value = strip_accents(value).lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return SPACE_RE.sub(" ", value).strip()


def clean_display_text(value: str) -> str:
    value = value.replace("|", " ").replace("_", " ")
    value = re.sub(r"^[\s•·.,;:()\-–—]+", "", value)
    value = re.sub(r"[\s•·.,;:()\-–—]+$", "", value)
    return SPACE_RE.sub(" ", value).strip()

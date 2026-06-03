from __future__ import annotations

import hashlib
import html
import re
from typing import Set, Tuple


TEXT_MIN_CHARS = 20


def normalize_text(text: str, *, strip_html: bool = False) -> str:
    text = html.unescape(str(text))
    text = text.replace("\u200b", " ")
    text = text.replace("\ufeff", " ")
    if strip_html:
        text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def clean_text(text: str, *, strip_html: bool = True) -> str:
    text = html.unescape(str(text))
    text = text.replace("\u200b", " ")
    text = text.replace("\ufeff", " ")
    if strip_html:
        text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def word_ngrams(text: str, n: int) -> Set[Tuple[str, ...]]:
    words = re.findall(r"\w+", text.lower())
    if len(words) < n:
        return set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def safe_id(text: str) -> str:
    text = text.strip().replace("/", "__")
    text = re.sub(r"[^A-Za-z0-9_.=-]+", "_", text)
    return text.strip("_") or "model"


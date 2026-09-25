"""
Multi-view text normalization for business names and addresses.
Deliberately simple, rule-based, and language-agnostic-ish so it does not
overfit to US/India patterns and generalizes reasonably to France.
"""
from __future__ import annotations
import re

LEGAL_SUFFIXES = [
    "private limited", "pvt ltd", "pvt. ltd.", "pvt ltd.", "private ltd",
    "limited", "ltd", "ltd.", "llp", "llc", "l.l.c", "inc", "inc.",
    "incorporated", "corporation", "corp", "corp.", "co", "co.", "company",
    "sarl", "sas", "sa", "gmbh", "plc",
]
LEGAL_SUFFIX_RE = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in sorted(LEGAL_SUFFIXES, key=len, reverse=True)) + r")\b\.?",
    flags=re.IGNORECASE,
)

ADDRESS_ABBR = {
    r"\brd\b": "road", r"\bst\b": "street", r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard", r"\bapt\b": "apartment", r"\bfl\b": "floor",
    r"\bste\b": "suite", r"\bdr\b": "drive", r"\bln\b": "lane",
    r"\bhwy\b": "highway", r"\bpvt\b": "private",
}

PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
WS_RE = re.compile(r"\s+")


def _base_clean(text: str) -> str:
    if not text:
        return ""
    t = text.lower().strip()
    t = t.replace("&", " and ")
    t = PUNCT_RE.sub(" ", t)
    t = WS_RE.sub(" ", t).strip()
    return t


def normalize_name(text: str) -> str:
    t = _base_clean(text)
    stripped = LEGAL_SUFFIX_RE.sub(" ", t)
    stripped = WS_RE.sub(" ", stripped).strip()
    return stripped if stripped else t  # never fully empty out a real name


def normalize_address(text: str) -> str:
    t = _base_clean(text)
    for pat, repl in ADDRESS_ABBR.items():
        t = re.sub(pat, repl, t)
    t = WS_RE.sub(" ", t).strip()
    return t


def tokens(text: str) -> set:
    return set(text.split()) if text else set()


def rare_tokens(text: str, min_len: int = 4) -> set:
    """Tokens likely to be discriminative (longer, non-numeric-generic)."""
    return {tok for tok in tokens(text) if len(tok) >= min_len}

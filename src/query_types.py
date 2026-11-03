"""Rule-based query-type tags, computed from the query text only (never from retrieval results).

Rules are documented and frozen in docs/query_tagging.md. Order is fixed: visual > exact > general.

- visual: the query refers to a visual element or appearance of the page (chart, figure, photo, logo,
  stamp, axis, ...).
- exact:  the query contains a lexical anchor that can be matched literally on the page: a digit, an
  acronym/code (>= 2 capitals), a quoted span, or a capitalised word that is not the first word
  (a name or a field label such as "Log-in No.").
- general: neither.
"""

from __future__ import annotations

import re

VISUAL_TERMS = [
    r"chart", r"graph", r"plot", r"plotted", r"figure", r"fig", r"diagram", r"map", r"image", r"picture",
    r"photo", r"photograph", r"logo", r"axis", r"x-axis", r"y-axis", r"legend", r"bar", r"pie",
    r"illustration", r"drawing", r"drawn", r"monogram", r"stamp", r"handwritten", r"color", r"colour",
    r"bold", r"symbol", r"icon",
]
VISUAL_RE = re.compile(r"\b(" + "|".join(VISUAL_TERMS) + r")s?\b", re.IGNORECASE)

DIGIT_RE = re.compile(r"\d")
ACRONYM_RE = re.compile(r"\b[A-Z][A-Z&]{1,}\b")  # SCGP, FDA, B&W, CC
QUOTE_RE = re.compile(r"[\"“”‘’']([^\"“”‘’']{2,})[\"“”‘’']")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-\.&']*")


def _capitalised_non_initial(query: str) -> list[str]:
    words = WORD_RE.findall(query)
    return [w for w in words[1:] if w[0].isupper() and w not in {"I"}]


def tag_query(query: str) -> tuple[str, str]:
    """Return (tag, rule that fired)."""
    m = VISUAL_RE.search(query)
    if m:
        return "visual", f"visual term '{m.group(0)}'"
    if DIGIT_RE.search(query):
        return "exact", "digit"
    m = ACRONYM_RE.search(query)
    if m:
        return "exact", f"acronym '{m.group(0)}'"
    m = QUOTE_RE.search(query)
    if m:
        return "exact", "quoted span"
    caps = _capitalised_non_initial(query)
    if caps:
        return "exact", f"capitalised '{caps[0]}'"
    return "general", "no rule"

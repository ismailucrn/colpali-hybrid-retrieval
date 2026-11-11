"""System A: BM25 over page text, identical to vidore-benchmark's `BM25Retriever` (the paper's BM25 baseline).

Preprocessing (copied from `BM25Retriever.preprocess_text`): NLTK `word_tokenize`, keep tokens that are
alphanumeric, lowercase, drop NLTK English stopwords. Scoring: `rank_bm25.BM25Okapi` with its default
parameters (k1=1.5, b=0.75, epsilon=0.25) — no tuning.

Post-hoc variant (added 2026-11-11, pre-registered before any run, `mode="codes"`): keeps lexical anchors that
the official preprocessing drops. Text is lowercased and split into alphanumeric runs joined by "-", ".", "/"
or "&" (e.g. "2-26", "no.73-170", "b&w"); each joined token is kept whole *and* as its parts; NLTK English
stopwords are removed except "no" (used as "number" in forms). Used only for the post-hoc analysis A_code.
"""

from __future__ import annotations

import re
from pathlib import Path

import nltk
import numpy as np
from rank_bm25 import BM25Okapi

NLTK_DIR = Path(__file__).resolve().parents[1] / "data" / "nltk_data"
NLTK_RESOURCES = {"punkt_tab": "tokenizers/punkt_tab", "stopwords": "corpora/stopwords"}
CODE_TOKEN = re.compile(r"[a-z0-9]+(?:[-./&][a-z0-9]+)*")
CODE_SEPARATORS = re.compile(r"[-./&]")
KEPT_STOPWORDS_CODES = {"no"}


def ensure_nltk_data() -> None:
    """Use a repo-local NLTK data dir (gitignored) and download the two resources if missing."""
    if str(NLTK_DIR) not in nltk.data.path:
        nltk.data.path.insert(0, str(NLTK_DIR))
    for name, path in NLTK_RESOURCES.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, download_dir=str(NLTK_DIR), quiet=True)


class Tokenizer:
    def __init__(self, mode: str = "official"):
        if mode not in ("official", "codes"):
            raise ValueError(f"unknown tokenizer mode {mode!r}")
        ensure_nltk_data()
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize

        self.mode = mode
        self.stop_words = set(stopwords.words("english"))
        if mode == "codes":
            self.stop_words -= KEPT_STOPWORDS_CODES
        self.word_tokenize = word_tokenize

    def __call__(self, text: str) -> list[str]:
        if self.mode == "official":
            return [w.lower() for w in self.word_tokenize(text) if w.isalnum() and w.lower() not in self.stop_words]
        tokens = []
        for tok in CODE_TOKEN.findall(text.lower()):
            parts = CODE_SEPARATORS.split(tok)
            for t in ([tok] + parts if len(parts) > 1 else [tok]):
                if t and t not in self.stop_words:
                    tokens.append(t)
        return tokens


class BM25Index:
    def __init__(self, texts: list[str], tokenizer: Tokenizer | None = None):
        self.tokenizer = tokenizer or Tokenizer()
        self.tokenized_corpus = [self.tokenizer(t) for t in texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def score(self, query: str) -> np.ndarray:
        return np.asarray(self.bm25.get_scores(self.tokenizer(query)), dtype=np.float64)

    def score_matrix(self, queries: list[str]) -> np.ndarray:
        return np.stack([self.score(q) for q in queries])

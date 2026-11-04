"""System A: BM25 over page text, identical to vidore-benchmark's `BM25Retriever` (the paper's BM25 baseline).

Preprocessing (copied from `BM25Retriever.preprocess_text`): NLTK `word_tokenize`, keep tokens that are
alphanumeric, lowercase, drop NLTK English stopwords. Scoring: `rank_bm25.BM25Okapi` with its default
parameters (k1=1.5, b=0.75, epsilon=0.25) — no tuning.
"""

from __future__ import annotations

from pathlib import Path

import nltk
import numpy as np
from rank_bm25 import BM25Okapi

NLTK_DIR = Path(__file__).resolve().parents[1] / "data" / "nltk_data"
NLTK_RESOURCES = {"punkt_tab": "tokenizers/punkt_tab", "stopwords": "corpora/stopwords"}


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
    def __init__(self):
        ensure_nltk_data()
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize

        self.stop_words = set(stopwords.words("english"))
        self.word_tokenize = word_tokenize

    def __call__(self, text: str) -> list[str]:
        return [w.lower() for w in self.word_tokenize(text) if w.isalnum() and w.lower() not in self.stop_words]


class BM25Index:
    def __init__(self, texts: list[str], tokenizer: Tokenizer | None = None):
        self.tokenizer = tokenizer or Tokenizer()
        self.tokenized_corpus = [self.tokenizer(t) for t in texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def score(self, query: str) -> np.ndarray:
        return np.asarray(self.bm25.get_scores(self.tokenizer(query)), dtype=np.float64)

    def score_matrix(self, queries: list[str]) -> np.ndarray:
        return np.stack([self.score(q) for q in queries])

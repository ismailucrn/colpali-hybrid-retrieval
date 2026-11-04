"""Tests for System A's preprocessing and scoring (CPU, no dataset download)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.bm25 import BM25Index, Tokenizer  # noqa: E402


def test_tokenizer_matches_official_preprocessing():
    tok = Tokenizer()
    # lowercase; stopwords ("what", "is", "the", "in") dropped; punctuation and non-alphanumeric tokens dropped
    assert tok("What is the dividend payout in 2012?") == ["dividend", "payout", "2012"]
    # "No" is an NLTK stopword and "-"-joined tokens are not alphanumeric: both dropped, as in the official code
    assert tok("What is the Log-in No. ?") == []


def test_bm25_ranks_the_matching_page_first():
    pages = ["annual report dividend payout 2012", "invoice number total amount", "meeting agenda minutes"]
    scores = BM25Index(pages).score("What is the dividend payout in 2012?")
    assert scores.argmax() == 0
    assert scores.shape == (3,)

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


def test_codes_tokenizer_keeps_codes_and_no():
    tok = Tokenizer(mode="codes")
    # "No" survives, the code is kept whole and as parts; other stopwords still dropped
    assert tok("What is No. 2-26 of subheading H ?") == ["no", "2-26", "2", "26", "subheading", "h"]
    assert tok("Log-in No.73-170") == ["log-in", "log", "no.73-170", "no", "73", "170"]
    # the official tokenizer is unchanged by the new mode
    assert Tokenizer()("What is No. 2-26 of subheading H ?") == ["subheading", "h"]

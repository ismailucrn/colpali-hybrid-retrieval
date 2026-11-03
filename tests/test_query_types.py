"""Tests for the frozen query-tagging rules (examples are DocVQA query texts)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.query_types import tag_query  # noqa: E402


@pytest.mark.parametrize("query,expected", [
    ("In fig 3, what is plotted along x axis?", "visual"),          # visual wins over the digit
    ("What is the name of the company in the logo?", "visual"),
    ("What is the title of the chart on the right ?", "visual"),
    ("What is the dividend payout in 2012?", "exact"),               # digit
    ("What is the full form of FERS?", "exact"),                     # acronym
    ("At what rate is Methoprene currently added to B&W's tobaccos?", "exact"),
    ("What time is the ‘formal presentation’?", "exact"),            # quoted span
    ("What is the Log-in No. ?", "exact"),                           # capitalised field label
    ("Who is the letter from?", "general"),
    ("What is the heading?", "general"),
    ("How many bullet points are there?", "general"),
])
def test_tag_query(query, expected):
    assert tag_query(query)[0] == expected


def test_first_word_capital_is_not_an_anchor():
    assert tag_query("What is the date of the letter ?")[0] == "general"

# Speaker notes

These notes go with `slides/slides.pdf` (17 main slides + 3 backup slides). Planned length is about 12–15 minutes, i.e. about 1 minute per main slide. Every number on the slides comes from `report/generated/numbers.tex`, which is written from `results/`.

## Slide by slide

**1 · Title.** Name the question in one sentence. Point to the repository URL on the slide.

**2 · Motivation.**
- Two ways to search scanned PDFs: the text pipeline (OCR, chunks, BM25 or dense embeddings) and ColPali (page images, no OCR).
- ColPali wins on the benchmark.
- In text retrieval, mixing lexical and dense rankings is a standard cheap improvement.
- So the natural question is whether that improvement still applies when one side reads images.

**3 · Question and predictions.**
- Stress that P1–P3 were written *before* any fusion run, and that the git commits on the slide prove it.
- Control D answers the obvious objection: "maybe any second retriever would help".

**4 · How ColPali scores a page.**
- Walk through the MaxSim formula: each query token takes its best page patch, then the scores are summed.
- The intuition: a query word can match the patch where that word is printed. This is why ColPali can do some "lexical" matching without OCR, which foreshadows the result.

**5 · Literature.**
- Two messages:
  1. Fusion in text IR is not unconditional. Bruch et al. find that default RRF falls below the better input when the inputs are unequal.
  2. Text + visual fusion has been tried, with mixed and uncontrolled results.
- Be honest that the gap was narrowed after a targeted search. That is a strength, not a weakness: we corrected our framing instead of over-claiming.

**6 · Reproduction.**
- We used the same checkpoint, data revision and protocol, and the authors' code.
- The measured value is 0.94 points below the model card, inside the ±2 tolerance fixed in advance.
- The paper's own number (54.4) is for v1.0. v1.0 does not run on the current library, so we reproduce v1.3 and report 54.4 only as context.

**7 · Design.**
- Same text for BM25 and BGE-M3. Same fusion formula and same tuning budget for C and D.
- The split and the tags were frozen before any fusion run.
- The test split was used once, and the script refuses to run with changed settings.

**8 · Dev.**
- On dev, fusion is already below ColPali at every k. The best k sits at the edge of the grid (k = 1), which means: give each system's top ranks the most weight.
- After seeing this we added weighted RRF. We labelled it dev-informed and committed its grid and selection rule before running it.

**9 · Test.**
- The headline: −5.4 points, significant after Holm correction. D behaves the same, so C ≈ D.
- The weighted version ties with ColPali. Its small dev gain disappeared on test, which is selection noise from searching 90 configurations, as anticipated.

**10 · By query type.**
- The surprise: ColPali is *best* on exact-match queries, the very type where we expected BM25 to help.
- Visual queries are few (17), so do not over-read that column.

**11 · Headroom.**
- The key slide. BM25 finds the gold page that ColPali misses on only 3 of 225 queries, and never on an exact-match query.
- Even an oracle that picks the better system per query gains only +1.8 nDCG@5 points.
- Fusion can mostly only import BM25's mistakes: 42 of C's 49 losses are queries where BM25 had missed.

**12 · Mechanism.**
- BM25 depends entirely on the query's words surviving OCR and tokenisation. With zero overlap BM25 never succeeds, while ColPali still succeeds 23% of the time.

**13 · Examples.**
- Show one unfavourable example concretely ("No. 2-26": "No" is a stopword and "2-26" is removed by the tokenizer).
- Show one favourable example ("name of the laboratory").
- Mention that the "bullet points" rescue is probably luck. This is honest reporting of favourable cases.

**14 · Cost.**
- Fusion is almost free: a fraction of a millisecond and a sub-MB index. So the obstacle is quality, not cost.
- Caveat: the components were measured on different devices.

**15 · Limitations.** Read the list. The most important items are: one task, ColPali was trained on DocVQA data, and the small test set.

**16 · Conclusions.**
- The reproduction holds. Fusion hurts, and the control shows it is not specific to BM25.
- The literature is consistent with this: fusion fails with unequal inputs and helped where the text was clean and equally strong.
- Open question: would better text extraction change this?

**17 · Reproducibility and AI use.**
- Everything is re-runnable on a laptop from the committed score matrices, and a fresh clone reproduces all outputs exactly.
- State plainly what the AI did and what you did and verified.

## Likely questions

- **Why v1.3 and not the paper's v1.0?**
  v1.0 is marked unsupported in the current colpali-engine. Making it run would mean old library versions and an unknown time cost. We reproduce the authors' own reported v1.3 number with the same protocol. The paper's 54.4 is shown as context only.
- **Is −0.94 points a failed reproduction?**
  No. It is within the ±2 tolerance fixed before the run. Likely sources are library versions and bf16 ties, but we did not isolate the cause (stated as a limitation).
- **Why RRF?**
  It is the standard, training-free fusion and needs no score calibration between very different scorers. Bruch et al. show that score-level convex combination can be better; we list that as untested. The per-query oracle (+1.8 points) and the weighted-RRF confidence interval suggest the upside is small.
- **k = 1 is at the edge of the grid. Isn't the grid too narrow?**
  Smaller k only moves further toward "each system's top ranks dominate". The weighted ablation explores the same direction in a better way, and it still only ties with ColPali.
- **Did you tune on the test set?**
  No. All choices were made on dev and committed before the single test run. The test script checks this. Weighted RRF was added after seeing dev, not test, and is labelled dev-informed.
- **Why BGE-M3 as the control?**
  It is the paper's own dense text baseline, chosen and committed before any result. It reads the same OCR text, so C vs D isolates "lexical vs dense", not "different input".
- **Are rule-based query tags reliable?**
  They are coarse, but they cannot be biased by the results: they use the query text only and were frozen before any fusion run. Weaknesses are documented in `docs/query_tagging.md`.
- **Is ColPali advantaged on DocVQA?**
  Yes, possibly. DocVQA is in its training data (the split is not stated by the authors), while BM25 and BGE-M3 are zero-shot. This is listed as a limitation. It is a reason the result may not transfer to other corpora.
- **Would better OCR change the result?**
  Possibly. That is our main open hypothesis. IRPAPERS saw gains with clean LLM transcriptions where text and image retrieval were about equally strong.
- **What did the AI do?**
  See slide 17 and `docs/ai_usage.md`: code, Colab driving, literature extraction and drafting. You made the design decisions, approved every run and verified *(fill in what you checked)*.

# Speaker notes

These notes go with `slides/slides.pdf` (18 main slides + 3 backup slides). Planned length is about 12–15 minutes, i.e. about 1 minute per main slide. Every number on the slides comes from `report/generated/numbers.tex`, which is written from `results/`.

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

**14 · Post-hoc checks.**
- We asked: what if the problem was the fusion function, or the tokenizer? We pre-registered two checks after the main test run, and they are labelled exploratory.
- Score-level fusion: dev picks ColPali alone, for both text retrievers.
- A better tokenizer makes BM25 better on its own (+2.3 points), but only on queries ColPali already solves. BM25-only hits stay at 3.
- Both alternative explanations are therefore ruled out on this task.

**15 · Post-hoc set 2: a full text RAG stack, and reranking ColPali.**
- Added on 15 November, pre-registered before running, labelled exploratory.
- A standard text RAG retrieval stack: BM25 + BGE-M3 hybrid, then a cross-encoder reranks the top 50. The cross-encoder helps the text pipeline by about 5 points, but the stack is still about 13 points below ColPali.
- The same cross-encoder applied to ColPali's own candidates makes dev worse at every depth, so dev picks "no reranking".
- So the negative result is not just "BM25 is too weak": even a strong reader of the OCR text adds nothing to ColPali here.
- OCR quality: a dictionary-share proxy does not explain who wins. Say that the proxy is crude.
- If asked about BGE-M3: the 4 empty-OCR pages share one embedding that ranks high for most queries. BGE-M3 alone is understated, but D and the hybrid barely change (checked on dev only).

**16 · What are ColPali's remaining misses?**
- Before trying more methods, we asked which errors any second signal could fix. The flags were fixed in a commit before counting.
- The clearest pattern: in 58 of the 78 test misses, no query word appears on the gold page's OCR. For hits this is 19 of 147. Dev shows the same contrast (52 of 80 vs 26 of 146).
- Read two examples aloud: "What is the name of the company on the letter head?" and "What is the date of the letter?". Many letters fit these questions. They were written by annotators looking at one page, so as search queries they are under-specified.
- Key point: even a much cleaner transcription (Qwen3.5) leaves 53 of the 78. So this is about the queries, not the OCR.

**17 · Post-hoc set 3: four attempts to beat ColPali.**
- Pre-registered on 16 November before running; newest Qwen models; settings chosen on dev; test used once (its fourth use, so exploratory).
- Cleaner OCR: BM25 improves by about 4.6 points, but fusing it with ColPali still hurts (−3.5 points, significant after Holm). Selective fusion and a Qwen3-VL embedding ensemble change nothing significant.
- Visual reranker: the only thing that reads the page image against the question. Dev +4.9 points, but on test only +1.1 and not significant. It fixed 10 misses but pushed 8 correct pages out of the top 5. It also costs about 7 seconds per query.
- Say it plainly: the dev gain did not replicate. That is why we choose settings on dev and look at test only once.

**18 · Post-hoc set 4: stronger visual models, and a fresh-task check.**
- We asked: if text cannot help, can a better *image* model? Again pre-registered, dev selection, one test run (the fifth on DocVQA, so exploratory).
- Two newer ColPali-style retrievers beat ColPali on DocVQA test: Tomoro ColQwen3-4B by about 6.6 points, ColQwen2.5 by about 4.8. Fusing either one with ColPali also helps (about +4.3), but less than the stronger model alone. The 8B Qwen reranker gives +3.2, just significant.
- Key point: because the DocVQA test set had been used so often, a rule fixed beforehand picked one candidate (Tomoro), and we tested it once on InfoVQA, a task we had never used. It gains 8.5 points there (p = 0.0001). As a bonus, ColPali reproduces its published InfoVQA score exactly (0.857).
- How to read it: fusion helps when the second ranking is visual and strong, and hurts when it is text. That is the same unequal-inputs story from the other side.
- Caveat to say aloud: these models were trained on more data, and we did not check training-data overlap.

**19 · Post-hoc set 5: is DocVQA special?**
- The obvious objection to everything so far: "that is one task". We took the fusion settings exactly as frozen on DocVQA dev and ran them on all nine other ViDoRe v1 tasks, pre-registered, with nothing tuned there. Each task is a fresh test.
- Result: BM25 fusion is significantly worse than ColPali on three tasks (InfoVQA, ArxivQA, TabFQuAD) and significantly better on none. On the text-dense tasks (TAT-DQA, ShiftProject, the synthetic reports) it is roughly neutral.
- Our pre-registered guess was that the headroom (queries BM25 finds and ColPali misses) would predict where fusion helps. It does not: the headroom is at most 7% anywhere. What tracks the loss, looked at afterwards and labelled post-hoc, is how much weaker BM25 is than ColPali. That is the unequal-inputs story again, now across tasks.
- The learned router, trained on nine tasks and tested on the tenth, almost always keeps ColPali. It does not help. An oracle shows that some queries do benefit, but score-margin features cannot find them.
- Caveats: four tasks have only 100 queries. ColPali is near the ceiling on the synthetic tasks. ColPali misses its card by just over 2 points on two tasks, which we did not investigate.

**20 · Set 5: what it shows.** On the left, every task's fusion difference, with only the visual tasks and TabFQuAD significantly negative. On the right, the post-hoc pattern: the further BM25 is below ColPali, the more fusion costs. Say that the right panel is descriptive and was not pre-registered.

**21 · Cost.**
- Fusion is almost free: a fraction of a millisecond and a sub-MB index. So the obstacle is quality, not cost.
- Caveat: the components were measured on different devices.

**22 · Limitations.** Read the list. The most important items are: one task, ColPali was trained on DocVQA data, and the small test set.

**23 · Conclusions.**
- The reproduction holds. Fusion hurts, and the control shows it is not specific to BM25.
- The literature is consistent with this: fusion fails with unequal inputs and helped where the text was clean and equally strong.
- Open question: would better text extraction change this?

**24 · Open questions.** Pick two: corpora where text retrieval is close to visual retrieval, and checking training-data overlap. Mention ViDoRe V2 as a harder benchmark where fusion should help even less.

**25 · Reproducibility and AI use.**
- Everything is re-runnable on a laptop from the committed score matrices, and a fresh clone reproduces all outputs exactly.
- State plainly what the AI did and what you did and verified.

## Likely questions

- **Maybe DocVQA is just a bad task for BM25?**
  We checked (slides 19–20). With the DocVQA settings frozen, fusion never helps significantly on any of the nine other ViDoRe v1 tasks, including the text-dense ones. The negative result is not DocVQA-specific.
- **Why v1.3 and not the paper's v1.0?**
  v1.0 is marked unsupported in the current colpali-engine. Making it run would mean old library versions and an unknown time cost. We reproduce the authors' own reported v1.3 number with the same protocol. The paper's 54.4 is shown as context only.
- **Is −0.94 points a failed reproduction?**
  No. It is within the ±2 tolerance fixed before the run. Likely sources are library versions and bf16 ties, but we did not isolate the cause (stated as a limitation).
- **Why RRF? Bruch et al. say convex combination is better.**
  RRF is the standard, training-free fusion and needs no score calibration between very different scorers. We also tested convex combination post-hoc (slide 14): on dev, every α < 1 is below ColPali, so it selects ColPali alone.
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
  We tested it (slide 17). A Qwen3.5-4B transcription makes BM25 about 4.6 points better, but fusing it with ColPali still hurts. The gold page still shares no word with the query for 53 of ColPali's 78 test misses. So on this task the obstacle is the queries, not the OCR. IRPAPERS saw gains where text and image retrieval were about equally strong, which is not the case here.
- **The visual reranker gained 4.9 points on dev. Why not report that?**
  Because dev chose its settings. The test run, done once, gives +1.1 (not significant). We report both and explain the gap: it fixed 10 misses but also broke 8 hits on test, and N was the top of a rising grid.
- **Did you use the test set too often?**
  Four times, each with settings frozen beforehand and labelled exploratory after the first. That is a limitation: one of the four results could be noise. The main conclusion rests on the first, pre-registered run.
- **What did the AI do?**
  See slide 25 and the report's AI-use appendix (App. B): code, Colab driving, literature extraction and drafting. You made the design decisions, approved every pre-registration and run, reviewed each result and read all papers in the literature matrix.
- **So did you find an improvement or not?**
  For the project's question (does a *text* signal help ColPali?): no, across four sets. For "can retrieval here be improved at all?": yes. A newer visual retriever is better, and it holds on a fresh task where its settings were frozen in advance. Fusion with ColPali helps only when the second model is visual and strong.

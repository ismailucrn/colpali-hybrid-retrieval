# Literature matrix

Eight papers are analysed in full (§2–§3). Three more appear as related work only (§4).

**Labels.**
- **[P] Published:** the authors' own claim, with its location in the paper.
- **[I] Interpretation:** our own reading.
- **[M] Measured:** our own result, with the file it comes from.
- **(unverified):** not yet checked against the paper text. Do not cite it before checking.

**How the details were checked.** Paper details were extracted on 2026-11-06 from the arXiv HTML or PDF versions listed with each paper. The extraction was AI-assisted (see `docs/ai_usage.md`). **The author must read every paper and confirm every cell before it goes into the report.** Section and table numbers are given so each cell can be checked quickly.

---

## 1. Search and selection approach

**Date of search:** the candidate set was drafted between 2026-10-23 and 11-03. Every paper was verified, and a targeted search for prior hybrid work was run, on 2026-11-06.

**Sources and search strategy**
1. **Seed set:** the main paper, ColPali (arXiv 2407.01449), plus its direct lineage:
   - the paper that introduced late interaction (ColBERT);
   - the benchmark it introduced and its successor (ViDoRe V1, inside the ColPali paper; ViDoRe V2);
   - the closest competing visual retriever (DSE).
2. **Method used in our improvement:** the original RRF paper (Cormack et al., 2009). The text-IR evidence on hybrid lexical + dense fusion comes from Bruch et al. (an analysis of fusion functions) and from BEIR (zero-shot BM25 vs dense retrieval).
3. **Targeted search for prior work closest to our research question** (web search, arXiv), with two queries:
   - "hybrid BM25 ColPali fusion visual document retrieval OCR text lexical"
   - "combining OCR text retrieval with visual document retrieval ColPali reciprocal rank fusion arXiv 2025"

   This search found IRPAPERS (2602.17687), the slide-retrieval study (2509.15211) and "Lost in OCR Translation?" (2505.05666). The DSE paper's own appendix (A.2) also contains a hybrid experiment.

**Inclusion criteria.** A paper was kept if it does at least one of the following:
- (a) introduces the method we reproduce or one of its components;
- (b) defines the evaluation protocol we use;
- (c) is a competing approach to visual document retrieval;
- (d) supplies the fusion method or the evidence behind our hypothesis;
- (e) directly tests text + visual fusion, which is our research question.

**Kept vs dropped**

| Paper | Decision | Reason |
|---|---|---|
| ColPali, ColBERT, ViDoRe V2, DSE | matrix | Criteria (a)–(c) |
| RRF (Cormack et al.), Bruch et al. | matrix | Criterion (d): our fusion method, and the strongest analysis of when it works |
| IRPAPERS, slide-retrieval study | matrix | Criterion (e): the closest prior work. Both are newer than our project plan and change how the gap must be stated (§5) |
| BEIR | related work (§4) | Strong evidence on BM25 vs dense retrieval, but it does **not** evaluate fusion [P, §2.1 mentions hybrids only as related work] |
| VisRAG | related work (§4) | Focuses on generation; our scope has no generation |
| Lost in OCR Translation? | related work (§4) | Compares OCR and visual pipelines but evaluates no fusion |
| RAG (Lewis et al., 2020) | dropped | Background only; no evidence on retrieval fusion |
| arXiv 2505.01457 (GME + ColQwen RRF), arXiv 2511.21121 (pyramid indexing) | found, **not read** | Search hits that fuse several visual paths. Read them if time allows; until then they are not cited |

**Known bias in the search.** It used two English queries on one web search engine, on one date. It did not use Google Scholar or Semantic Scholar citation tracking of ColPali. Newer or non-arXiv work may be missing. *[I]* Given the deadline, the next step to reduce this bias is a forward-citation check of ColPali on Semantic Scholar.

---

## 2. Overview

| # | Paper | Role for us | Modalities | One-line evidence relevant to our question |
|---|---|---|---|---|
| 1 | ColPali (Faysse et al.) | Reproduced method (B) | page image ↔ text query | Beats every text pipeline on ViDoRe V1 (avg nDCG@5 81.3 vs 67.0) [P, Table 2] |
| 2 | ColBERT (Khattab & Zaharia) | Origin of late interaction | text ↔ text | Late interaction is close to BERT re-ranking quality at a fraction of its cost [P, Table 1] |
| 3 | DSE (Ma et al.) | Competing visual retriever | screenshot ↔ text query | Beats OCR text on slides, but is below a same-size text model on Wikipedia [P, Table 1]; hybrid gives "minimal improvement" [P, App. A.2] |
| 4 | ViDoRe V2 (Macé et al.) | Newer protocol | page image ↔ text query | V1 is near saturation; V2 evaluates visual retrievers only, with no text or BM25 baselines [P, §5, Table 2] |
| 5 | RRF (Cormack et al.) | Fusion method (C, D) | rank lists (any) | RRF beats the best single run and Condorcet with k = 60 [P, Tables 1–2] |
| 6 | Bruch et al. | Fusion analysis | text (lexical + dense) | Default RRF falls below the better single retriever on 4/9 datasets; tuned convex combination is better [P, Table 2] |
| 7 | IRPAPERS (Shorten et al.) | Closest prior work | page image + LLM transcription | Text + image fusion beats image alone (R@1 49 vs 43) when text ≈ image [P, Abstract] |
| 8 | Slide retrieval (Giouroukis et al.) | Close prior work | slide image + captions/OCR | RRF[ColPali visual, ColPali on captions] ≈ ColPali alone (81.0 vs 82.7 nDCG@10) [P, Table 1] |

---

## 3. Matrix (seven fields per paper)

### 3.1 ColPali — Faysse, Sibille, Wu, Omrani, Viaud, Hudelot, Colombo

*Citation:* "ColPali: Efficient Document Retrieval with Vision Language Models", arXiv 2407.01449v6 (28 Feb 2025). Venue: not stated on the pages read. ICLR 2025 is (unverified); check the published version before citing a venue.
*Read:* https://arxiv.org/html/2407.01449v6, §1–7 verbatim. Appendices A.3, B.2–B.4 and C–F were not read verbatim.

- **Question and contribution.** [P, §1–2] Practical document retrieval is bottlenecked by the ingestion pipeline (PDF parsing, OCR, layout detection, captioning), not by the embedding model. The paper makes two contributions:
  - the **ViDoRe** benchmark for page-level retrieval;
  - **ColPali**, which indexes pages from their image alone using late interaction.

  It sets three requirements: R1 retrieval quality, R2 fast online querying, R3 fast indexing.
- **Modalities and assumptions.** [P, §4.1] Pages are images and queries are text. Text in the page is read from the pixels, with no OCR.
  - [P, §4.2] Training data are English only "by design".
  - [I] Assumption: everything a query needs is visible in a 448×448 rendering (1024 patches). Small print may be lost at this resolution; this is relevant to our DocVQA failures (`results/failure_analysis/examples.md`).
- **Mechanism and objective.**
  - [P, §2.2, §4.1] Backbone: PaliGemma-3B (SigLIP-So400m vision encoder + Gemma-2B). Every output token is projected to D = 128.
  - [P, Eq. 1] Score: ColBERT-style **MaxSim**, the sum over query tokens of the maximum dot product over page vectors.
  - [P, Eq. 2] Loss: in-batch contrastive loss, softmax cross-entropy of the positive score against the hardest in-batch negative.
  - [P, §4.2] Query augmentation with 5 `<unused0>` tokens. LoRA (r = 32, α = 32), 1 epoch, batch size 32, bf16.
- **Data and protocol.**
  - [P, §4.2] Training: 118,695 query–page pairs, 63 % from academic datasets and 37 % synthetic (Claude-3 Sonnet questions on web-crawled PDFs). The authors checked that no multi-page PDF appears in both ViDoRe and the training set.
  - [P, §3.1, Table 1] ViDoRe V1 has five academic tasks (DocVQA 500/500, InfoVQA, TAT-DQA, arXivQA, TabFQuAD) and five practical tasks (100 queries / 1000 pages each). Main metric: nDCG@5.
  - [I] The overlap check covers *multi-page PDFs*. It does not obviously cover the single-page academic sets such as DocVQA, whose *train* split is part of the training data. This is train/test from the same distribution, not a leak, but it favours ColPali on DocVQA relative to zero-shot text baselines.
- **Baselines and evidence.**
  - [P, §3.2] Text pipelines are built with "Unstructured" (text-only, + OCR, + captioning) and embedded with BM25 or BGE-M3. A page's score is the max over its chunks.
  - [P, Table 2] DocVQA column (nDCG@5):

    | System | DocVQA nDCG@5 |
    |---|---|
    | ColPali | **54.4** |
    | BM25 + OCR | 36.8 |
    | BM25 + captioning | 38.4 |
    | BGE-M3 + OCR | 25.7 |
    | BiPali | 30.0 |

    ViDoRe average: ColPali 81.3; best text pipeline (BGE-M3 + captioning) 67.0.
  - [P, §3.2] "Little difference" between BM25 and BGE-M3 on the same text: the authors read this as a "visual information bottleneck".
  - [M, `results/colpali_v1.3_docvqa/metrics.json`] We measured nDCG@5 0.5752 for checkpoint v1.3. Its model card reports 0.5845.
- **Resources and reproducibility.**
  - [P, §5.2, Fig. 2] Indexing on an L4: about 0.39 s/page for ColPali vs 7.22 s/page for the Unstructured pipeline (unverified: the per-page totals come from App. B.4 / Table 5, which was not read verbatim).
  - [P, §5.2] 257.5 KB per page in fp16. Token pooling with factor 3 keeps 97.8 % of performance with 66.7 % fewer vectors.
  - Query encoding: about 30 ms for ColPali vs about 22 ms for BGE (unverified: the extracted HTML text is garbled; check the PDF).
  - [P, §4.2] Training on 8 GPUs (GPU type not stated).
  - [P, Reproducibility statement] Code, adapters, training set and benchmark are released.
  - [I] Licence wording is inconsistent ("MIT" vs "open licenses") and should be checked on the repositories.
- **Limitations and connections.**
  - [P, §5.2, §6–7] Pooling degrades most on text-dense documents (Shift). Conclusions about unfreezing the vision encoder "may change" at larger training scale. Future work: end-to-end RAG and confidence estimation.
  - [I] The paper compares ColPali **against** text pipelines but never **combines** them, which is the starting point of our question. Its BM25 baseline (Unstructured + Tesseract) differs from the `_tesseract` OCR we use (see `docs/deviations.md`).

### 3.2 ColBERT — Khattab, Zaharia

*Citation:* "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT", SIGIR 2020, DOI 10.1145/3397271.3401075; arXiv 2004.12832v2.
*Read:* https://arxiv.org/html/2004.12832v2, in full.

- **Question and contribution.** [P, §1, §4] BERT re-rankers cost orders of magnitude more than earlier rankers. Can late interaction keep their quality at much lower cost (RQ1–RQ4)? Contributions:
  - the late-interaction paradigm;
  - BERT query and document encoders;
  - re-ranking and end-to-end retrieval with FAISS.
- **Modalities and assumptions.** [P] Text only: passages and queries. [I] Assumption: relevance is a sum of per-query-token soft matches. This favours queries whose tokens each have a counterpart in the document.
- **Mechanism and objective.**
  - [P, §3.2] One shared BERT with `[Q]` / `[D]` markers. Queries are padded with `[mask]` tokens to N_q = 32 (query augmentation). Linear projection to m = 128, then L2 normalisation.
  - [P, §3.3, Eq. 3] Score S = Σ_i max_j E_qi · E_dj (MaxSim).
  - [P, §3.3] Loss: pairwise softmax cross-entropy over ⟨q, d+, d−⟩ triples.
- **Data and protocol.** [P, §4.1]
  - MS MARCO passage ranking: 8.8 M passages; MRR@10 on dev / local eval.
  - TREC CAR: about 29 M passages; MAP.
  - Re-ranking uses BM25's top 1000 candidates.
- **Baselines and evidence.**
  - [P, Table 1] MS MARCO dev MRR@10, re-ranking: ColBERT 34.9 vs BERT-base 36.0 (own training) and BM25 16.7.
  - [P, Table 2] End-to-end ColBERT: 36.0, Recall@1000 96.8.
  - [P, §4.4, Fig. 5] Ablations: single-vector, average-sim and no-query-augmentation variants are all worse.
- **Resources and reproducibility.**
  - [P, Table 1, §4.2] Re-ranking latency 61 ms vs 10,700 ms for BERT-base ("over 170×"); about 13,900× fewer FLOPs.
  - [P, Table 4] Index size 27–286 GiB depending on dimension and precision. [P, §4.5] Indexing takes about 3 h on 4 GPUs.
  - Code release: not stated in the paper text.
- **Limitations and connections.**
  - [P, §3.2, Table 4] Storing one vector per token is expensive; dimension and precision control the footprint. Compression is left to future work [P, §4.2].
  - [I] ColPali reuses MaxSim, D = 128, query augmentation and the in-batch loss family directly. Image patches play the role of document tokens.
  - [I] BEIR's cost table shows the same storage concern for ColBERT (§4).

### 3.3 DSE — Ma, Lin, Li, Chen, Lin

*Citation:* "Unifying Multimodal Retrieval via Document Screenshot Embedding", arXiv 2406.11251v2 (2 Dec 2024). arXiv comment: "EMNLP2024 main".
*Read:* https://arxiv.org/html/2406.11251v2, in full.

- **Question and contribution.** [P, Abstract, §1] Treat document screenshots as a unified input and skip content extraction, which is "tedious, prone to errors, and has information loss". Contributions:
  - DSE;
  - the Wiki-SS dataset;
  - evidence on text-heavy (Wikipedia) and mixed-modality (slides) corpora.
- **Modalities and assumptions.** [P, §3.1] Screenshot documents and text-only queries. [I] Assumption: a single vector per page is enough. This is the main design difference from ColPali.
- **Mechanism and objective.**
  - [P, §3.2, §4.3] Phi-3-vision (about 4B parameters) bi-encoder.
  - Screenshot → (4×4 + 1) crops of 336×336 → patch tokens. The page embedding is the last hidden state of `</s>`.
  - Cosine similarity; InfoNCE loss with in-batch and hard negatives, τ = 0.02; LoRA.
- **Data and protocol.**
  - [P, §4.1] Wiki-SS: 1,267,874 Wikipedia screenshots. Evaluated on NQ (3,610 test questions) with top-k accuracy (exact answer match).
  - [P, §4.2] SlideVQA-open: 50,714 slides, 2,136 test questions; nDCG@10 and Recall@10. The OCR text comes from pytesseract.
- **Baselines and evidence.**
  - [P, Table 1] NQ Top-1:

    | System | NQ Top-1 |
    |---|---|
    | DSE | 46.2 |
    | BM25 | 29.5 |
    | E5 | 47.6 |
    | Phi-3 text retriever | **50.6** |

    [P, §5.1] The authors' reading: VLMs "still cannot fully capture the text content in a screenshot".
  - [P, Table 1] SlideVQA nDCG@10: DSE 75.3 vs BM25 55.8 and E5 59.3.
  - [P, §5.4.2] Of 50 text-retriever failures on slides, 22 were text-extraction errors and 28 needed visual context.
  - [P, App. A.2, Table 4] **Hybrid** (fused rankings): DSE + Phi-3 text reaches 71.5 nDCG@10 on SlideVQA. On Wiki-SS the authors report "minimal improvement".
  - *Unverified:* the exact fusion method used in A.2, and how 71.5 compares with DSE alone in the same setting. The fused number is below supervised DSE alone (75.3, Table 1); check whether A.2 uses a different setting.
- **Resources and reproducibility.**
  - [P, §4.3] Trained on 2× A100 80 GB.
  - [P, §5.3, Fig. 4] On an H100, going from 1×1 to 4×4 crops raises NQ Top-10 from 62.0 to 73.7 but lowers encoding speed from 12.2 to 4.3 documents/s.
  - [P, Abstract v2] Code, models and Wiki-SS are released via Tevatron.
- **Limitations and connections.**
  - [P, §7] A gap remains on PDFs and general documents; blurry or low-resolution screenshots hurt; high resolution is costly.
  - [I] DSE is the single-vector alternative to ColPali. It is the only seed paper that reports a text-on-text vs screenshot comparison where text **wins** (NQ), and it tried a hybrid and found little gain. ViDoRe V2 later ranks DSE models close to ColQwen and above colpali-v1.3 (§3.4).

### 3.4 ViDoRe Benchmark V2 — Macé, Loison, Faysse

*Citation:* "ViDoRe Benchmark V2: Raising the Bar for Visual Retrieval", arXiv 2505.17166v2 (19 Sep 2025). Venue: not stated.
*Read:* https://arxiv.org/html/2505.17166v2, in full.

- **Question and contribution.** [P, Abstract] ViDoRe V1 is "approaching saturation", with top models above 90 nDCG@5. V2 adds harder, more realistic queries:
  - blind contextual querying;
  - long and cross-document queries;
  - synthetic generation with human review [P, §2].
- **Modalities and assumptions.** [P] Page image ↔ text query, with multilingual queries (FR, EN, ES, DE) [§3]. [I] It assumes visual retrievers are the systems to compare; text pipelines are out of scope.
- **Mechanism and objective.** [P] A benchmark, not a model. Generator models see only summaries and metadata, to reduce extractive questions with high lexical overlap [§2].
- **Data and protocol.** [P, Table 1] Four domains: insurance (FR), biomedical, economics, ESG (human and synthetic subsets). 2.5–15.6 relevant pages per query on average. Main metric: nDCG@5 [§5]. Code: `vidore-benchmark`; MTEB planned [§4].
- **Baselines and evidence.**
  - [P, Table 2] 15 visual retrievers. Average nDCG@5: best 0.599 (ColQwen2.5 variant); colpali-v1.3 0.546; dse-qwen2 0.580.
  - [P, §5] Rankings are consistent between V1 and V2.
  - **No BM25, OCR-text or hybrid baseline** is evaluated [P, §5, Table 2].
- **Resources and reproducibility.**
  - [P, §4] Datasets are on HF; evaluation CLI given; leaderboard referenced.
  - [P, Table 1 fn. 1] The insurance set was removed for copyright reasons.
  - Dataset licences are not stated.
- **Limitations and connections.**
  - [P, §5–6] Some small models overfit. Future work: confidence estimation and more languages.
  - [I] V2 deliberately **reduces lexical overlap** between query and page. Our failure analysis shows lexical overlap is exactly what BM25 needs (`results/failure_analysis/analysis.json`). On V2 we would therefore expect BM25 to be even weaker and fusion even less useful (Hypothesis, untested).
  - [I] V2 is the newer protocol, but it has multiple relevant pages per query and multilingual splits. We kept V1 DocVQA because its published numbers make reproduction checkable (`docs/deviations.md`).

### 3.5 Reciprocal Rank Fusion — Cormack, Clarke, Büttcher

*Citation:* "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods", SIGIR 2009 (short paper, 2 pages), DOI 10.1145/1571941.1572114.
*Read:* the PDF from cormack.uwaterloo.ca, in full (text extracted locally).

- **Question and contribution.** [P, §1] RRF was meant as an unsupervised baseline for metaranking. It turned out to beat Condorcet Fuse, CombMNZ, and individual learning-to-rank methods.
- **Modalities and assumptions.** [P] Any set of rankings over the same documents. RRF combines ranks "without regard to the arbitrary scores" [§2]. [I] Implicit assumption: the input rankings are of comparable quality. RRF has no way to down-weight a weak system.
- **Mechanism and objective.**
  - [P, §1] RRFscore(d) = Σ_r 1 / (k + r(d)), with **k = 60** "fixed during a pilot investigation". No training is needed.
  - [P, §2] The authors' intuition: one or two systems that rank a document highly can lift it. The constant k "mitigates the impact of high rankings by outlier systems" [§1].
- **Data and protocol.** [P, §1, Tables 1–3]
  - Pilot: 30 Wumpus configurations on four TREC collections.
  - TREC 3, 5, 9 and Robust 2004: fusion of submitted runs.
  - LETOR 3: fusion of learning-to-rank methods.
  - Metric: MAP; sign test, plus 95 % CIs on LETOR.
- **Baselines and evidence.**
  - [P, Table 1] k sweep 0–500: MAP .2072–.2147, nearly flat. "Not critical" around k = 60. Best individual run .2016.
  - [P, Table 2] RRF beats Condorcet on all four collections. It beats the best single run except TREC 9, where the best run was manual.
  - [P, §1] On average 4–5 % above Condorcet, CombMNZ and the best system. Sign tests p ≈ 0.008–0.04.
  - [P, Table 3] On LETOR 3, RRF (0.6051) beats all learners; CombMNZ is 0.6107 (n.s.).
- **Resources and reproducibility.** [P, §2] Simpler than Condorcet; ranks can be summed one system at a time. [I] Trivial to reimplement; our version is `src/fusion.py`.
- **Limitations and connections.**
  - [I] The evidence comes from fusing **many** runs (e.g. 30 configurations, TREC submissions) of broadly similar quality. It says nothing about fusing **two** systems with a large quality gap. That is our setting: on test, BM25 nDCG@5 0.382 vs ColPali 0.570 [M, `results/test_evaluation/metrics.json`].
  - [I] The paper reports MAP only (P@k, R-precision and NDCG are said to be "comparable" [§2]), not shallow-rank metrics such as nDCG@5.
  - [I] Bruch et al. (§3.6) later re-test RRF's robustness to k.

### 3.6 Fusion functions for hybrid retrieval — Bruch, Gai, Ingber

*Citation:* "An Analysis of Fusion Functions for Hybrid Retrieval", arXiv 2210.11934v2 (4 May 2023). Related DOI 10.1145/3596512 (ACM TOIS is (unverified)).
*Read:* https://arxiv.org/abs/2210.11934 and the HTML v2 (§1–4.1 and §4.2–7 verbatim). Appendices B–D were not read.

- **Question and contribution.** [P, §1]
  - RQ1: is convex combination (CC) of normalised scores reasonable, and how much does normalisation matter?
  - RQ2: how sensitive is RRF to its parameter?
  - RQ3: does RRF's disregard of score distances matter?

  Finding [P, Abstract]: "Contrary to existing studies, we find RRF to be sensitive to its parameters."
- **Modalities and assumptions.** [P, §3.2] Text only. BM25 (PISA) + all-MiniLM-L6-v2 dense retriever, fused on the union of top-1000 lists. Other pairs are in the appendices.
- **Mechanism and objective.** [P, §4–5]
  - CC: α · f_sem + (1 − α) · f_lex after min-max, theoretical min-max or z-score normalisation.
  - RRF with parameter η, including a separate η per retriever and a weighted ("rrf-CC") variant, plus a smoothed RRF.
  - α is tuned on in-domain validation queries (α = 0.8) [P, §5].
- **Data and protocol.** [P, §3.2, Table 1] MS MARCO plus 8 BEIR datasets. In-domain = MS MARCO, NQ, Quora (the dense model was trained on them); the rest are zero-shot. Metric: NDCG@1000 and Recall@1000 (@100 for SciFact and NFCorpus).
- **Baselines and evidence.**
  - [P, Table 2] Tuned CC beats both single retrievers on all 9 datasets.
  - **RRF (η = 60) is below the better single retriever on 4 of 9**:

    | Dataset | RRF (η = 60) | Better single retriever |
    |---|---|---|
    | MS MARCO | 0.425 | 0.441 (dense) |
    | Quora | 0.877 | 0.889 (dense) |
    | HotpotQA | 0.675 | 0.682 (BM25) |
    | FiQA | 0.464 | 0.467 (dense) |

  - [P, §5.1, Fig. 7] "NDCG swings wildly as a function of rrf parameters". Performance improves when the two η values differ, i.e. when the weaker retriever's ranks are discounted.
  - [P, Table 3] Tuned RRF "does not generalize well" out of domain.
  - [P, App. A, Table 5] With a strong learned sparse model (SPLADE) + BM25, even tuned CC is not above SPLADE alone on MS MARCO (0.507 vs 0.508) or NQ (0.587 vs 0.591).
- **Resources and reproducibility.**
  - [P, §6, Fig. 12] α converges with "less than 5%" of the training data. RRF parameters stay sensitive to the sample.
  - No latency or compute figures are reported (unverified: based on the sections read).
- **Limitations and connections.**
  - [P, §7] Only collection-level analysis (a per-query analysis "may change" conclusions); two retrievers only; linear fusion only.
  - [I] This is the strongest prior evidence for our outcome. Equal-weight RRF can fall below the better input, and the gain disappears when one input dominates (the SPLADE case).
  - [I] Our dev selection independently arrived at a weight of 0.95 on ColPali [M, `results/fusion_docvqa/dev_selection_weighted.json`]. This matches their observation that the best fusion leans heavily toward the stronger retriever. In our case, it leans so far that the result is ColPali alone.

### 3.7 IRPAPERS — Shorten, Skaburskas, Jones, Pierse, Esposito, Trengrove, Dilocker, van Luijt (Weaviate)

*Citation:* "irpapers: A Visual Document Benchmark for Scientific Retrieval and Question Answering", arXiv 2602.17687v1 (5 Feb 2026). Preprint; venue not stated.
*Read:* https://arxiv.org/html/2602.17687, about the first third verbatim (Abstract, §1–3.6). Later sections were only summarised; nothing from them is used here.

- **Question and contribution.** [P, Abstract] How do image-based and text-based retrieval compare on scientific papers, and are their failures complementary? Benchmark: 3,230 pages from 166 IR papers, each with an image and a transcription; 180 needle-in-a-haystack questions.
- **Modalities and assumptions.**
  - [P, §2] Page images, plus **GPT-4.1 transcriptions** as the text side. These are not classical OCR.
  - [I] The text side is therefore much cleaner than Tesseract OCR. Text retrieval is roughly as strong as image retrieval here, unlike in our setting.
- **Mechanism and objective.** [P, §3.1–3.4]
  - Text: BM25 + Arctic 2.0 dense, fused by relative score fusion (RSF; min-max normalisation, then a weighted sum).
  - Image: ColModernVBERT (MaxSim); ColPali and ColQwen2 are also evaluated.
  - Multimodal hybrid: an α sweep between text-only and image-only. At α = 0.5 the weights are 0.25 / 0.25 / 0.5.
  - RRF is also defined (1 / (k + rank)); the k value is not stated in the part read.
- **Data and protocol.** [P, §2] Questions are generated by Claude Sonnet 4.5, one per non-reference page of 19 papers. Metrics: Recall@1/5/20 for retrieval; LLM-judged answer alignment for QA.
- **Baselines and evidence.** [P, Abstract] Recall@1 / 5 / 20:

  | System | R@1 | R@5 | R@20 |
  |---|---|---|---|
  | Text hybrid | 46 | 78 | 91 |
  | Image (ColModernVBERT) | 43 | 78 | 93 |
  | **Multimodal hybrid** | **49** | **81** | **95** |

  - [P, §1] Complementarity at Recall@1: 22 queries succeed with text only, 18 with images only.
  - [I] No significance test or confidence interval appears in the part read. 180 queries is small; a gain of 6 R@1 points is about 11 queries.
- **Resources and reproducibility.** [P, §3.3, §1] Also tests MUVERA compression: 41 % R@1, 2 points below the uncompressed model. Data and code are on GitHub (Weaviate), (unverified: not opened).
- **Limitations and connections.**
  - Representation-level fusion is left to future work [P, §3.4]. Other stated limitations were only seen in a summary, so they are not cited.
  - [I] This is the closest positive result to our hypothesis. Its conditions differ from ours in exactly the variables our analysis points to:
    - near-equal text and image strength vs our 0.38 vs 0.57;
    - many queries that succeed only with text (22/180) vs our 3/225 [M, `results/failure_analysis/analysis.json`];
    - clean LLM transcriptions vs noisy Tesseract OCR.

### 3.8 Slide retrieval study — Giouroukis, Dimitriadis, Papadopoulos, Shao, Tsoumakas

*Citation:* "What's the Best Way to Retrieve Slides? A Comparative Study of Multimodal, Caption-Based, and Hybrid Retrieval Techniques", arXiv 2509.15211v1 (18 Sep 2025). The HTML template names CIKM 2025, but the DOI is a placeholder, so the venue is (unverified).
*Read:* https://arxiv.org/html/2509.15211, in full.

- **Question and contribution.** [P, §1] Which text-to-slide retrieval architecture gives the best accuracy / latency / storage trade-off? Visual (ColPali, DSE), caption-based (VLM captions + BM25 / dense / "textual ColPali"), OCR + BM25, hybrids and rerankers are compared.
- **Modalities and assumptions.** [P, §3] Slide images, VLM-generated captions, and OCR text (Tesseract on SlideVQA).
- **Mechanism and objective.** [P, §3.5]
  - Hybrids either take k/2 results from each method or use RRF. The paper does not say which rows use which method, and gives no RRF k.
  - Reranking: retrieve 100 candidates, rerank to the top 10.
  - **No BM25 + ColPali (visual) fusion is tested.** The only fusion involving the visual retriever is RRF[ColPali visual, ColPali text encoder on captions].
- **Data and protocol.** [P, §3.1, §3.7] SlideVQA (open-domain, 2,215 test questions) and LPM (838 test queries from lecture transcripts). Metrics: nDCG@10 and Recall@10, plus latency and storage.
- **Baselines and evidence.** [P, Table 1] nDCG@10 on SlideVQA:

  | System | SlideVQA nDCG@10 |
  |---|---|
  | ColPali visual | **82.7** |
  | RRF[visual, textual] | 81.0 |
  | BM25 on OCR | 54.3 |
  | BM25 on Gemma3 captions | 68.8 |

  - On LPM: ColPali visual 66.7 vs RRF 66.8.
  - BM25 + dense text helps with Molmo captions (68.9 vs 63.6 / 65.3) but not with Gemma3 captions (66.5 vs BM25 68.8).
  - **No significance testing** and no per-query-type analysis.
- **Resources and reproducibility.** [P, §3.8, Table 1] Single RTX 3090.
  - Latency: ColPali 0.21 s/query; BM25 0.04 s; a VLM reranker adds about 7–14 s.
  - Storage: ColPali 12.9 GB vs BM25 0.04 GB on SlideVQA.
  - Code release: not stated (unverified).
- **Limitations and connections.**
  - [P, §5] Reranker latency is a key challenge. There is no limitations section.
  - [I] An independent result consistent with ours: adding a text-based ranking to ColPali through RRF does not beat ColPali alone. It does not test our exact question (lexical BM25 on OCR + visual ColPali) and reports no uncertainty.

---

## 4. Related work (not in the matrix)

- **BEIR** — Thakur, Reimers, Rücklé, Srivastava, Gurevych. NeurIPS 2021 Datasets & Benchmarks; arXiv 2104.08663v4.
  - [P, Abstract, §5] Across 18 zero-shot datasets (nDCG@10), BM25 is "a robust baseline". Dense retrievers often underperform it out of domain (e.g. DPR −47.7 %, TAS-B −2.8 % vs BM25 on average). Re-ranking and late interaction (ColBERT +2.5 %) do best, at high cost.
  - **Fusion is not evaluated** [P, §2.1 mentions hybrids only as related work].
  - Use: background for why BM25 is a meaningful second signal, not evidence that fusion helps.
- **VisRAG** — Yu et al., arXiv 2410.10594v2.
  - [P, §3.2, §4] VLM-based single-vector retriever (MiniCPM-V 2.0, position-weighted mean pooling, InfoNCE) plus VLM generation. Claims a 20–40 % end-to-end gain over text RAG (Abstract).
  - It evaluates MP-DocVQA, not our DocVQA. Its retrieval table was not read verbatim, so no numbers are cited.
  - Use: shows the field moving to vision-only RAG. Generation is out of our scope.
- **Lost in OCR Translation?** — Most et al., ACM DocEng 2025 (per the paper header); arXiv 2505.05666v1.
  - [P, Table 1] On 4,196 degraded scientific documents (DocDeg), a Llama 3.2 90B OCR + embedding pipeline beats ColQwen2 at every degradation level (total nDCG@5 0.523 vs 0.274).
  - Its ViDoRe DocVQA comparison (ColQwen2 ahead) was seen only in a summary (unverified).
  - No fusion is evaluated.
  - Use: text quality and domain shift can reverse the visual-vs-text ranking, so our conclusion is specific to DocVQA with Tesseract OCR.

---

## 5. Synthesis and research gap

### What the papers agree on
1. **Late interaction gives near cross-encoder quality at bi-encoder cost** (ColBERT). ColPali carries this to page images and beats every text pipeline on ViDoRe V1, including DocVQA (54.4 vs at most 38.4) [P, ColPali Table 2].
2. **Visual retrievers lose some fine text information.**
   - On text-heavy Wikipedia, a text retriever with the same backbone beats DSE [P, DSE Table 1].
   - With strong OCR on degraded documents, text pipelines beat ColQwen2 [P, Lost in OCR, Table 1].
   - *Interpretation:* this is the basis of our hypothesis that a lexical signal can still add something.
3. **Hybrid lexical + dense fusion helps in text IR, but not unconditionally.**
   - Tuned convex combination beats both inputs on all 9 datasets.
   - Default RRF falls below the better input on 4 of 9.
   - Fusion gains vanish when one input dominates (SPLADE + BM25) [P, Bruch et al., Tables 2, 5].
   - RRF's own evidence comes from fusing many runs of similar quality [P, Cormack et al., Tables 1–2].

### Where the evidence is weak or conflicting
- **Text + visual fusion has been tested only recently and with mixed results:**
  - helps on scientific papers with clean LLM transcriptions (IRPAPERS: +6 R@1);
  - "minimal improvement" for DSE on Wikipedia (DSE App. A.2);
  - no gain on slides (slide study: RRF 81.0 vs ColPali 82.7).
- None of these reports significance tests, a control for "any second retriever", or a pre-registered analysis by query type.
  - *[I]* Our statement is limited to the sections we read: IRPAPERS was read only to §3.6, and the DSE hybrid section was read without its full experimental setup.
- **No study fuses BM25 over the standard benchmark OCR with ColPali on a ViDoRe task.** The two benchmark papers (ColPali, ViDoRe V2) evaluate visual retrievers either *against* text pipelines or *without* them, never *combined* with them [P, ColPali §3.2 / Table 2; ViDoRe V2 Table 2].

### Research gap (justified)
Prior work shows three things:
- lexical + dense fusion is a cheap and usually positive step in text IR, but it can fail when the inputs differ strongly in quality (Bruch et al.);
- text + visual fusion of page retrievers has been tried a few times with mixed, uncontrolled results (IRPAPERS, DSE A.2, slide study);
- the main visual retrieval benchmarks do not test fusion at all (ColPali, ViDoRe V2).

**The open question is therefore not whether text + visual fusion *can* help. It is *under which conditions* a lexical signal adds anything to a late-interaction visual retriever.** Specifically:
- on the standard benchmark, with the standard OCR text;
- with a control that separates "lexical signal" from "any second retriever";
- broken down by query type;
- with honest uncertainty.

Our experiment fills this gap for one task (ViDoRe V1 DocVQA). It uses:
- the published checkpoint, reproduced within tolerance;
- BM25 over the benchmark's Tesseract OCR;
- a dense-text control (BGE-M3, the paper's own dense baseline);
- frozen query-type tags and a dev/test split;
- paired permutation tests with Holm correction.

### How our result connects back
[M, `results/test_evaluation/`, `results/failure_analysis/`]
- Equal-weight RRF lowers nDCG@5 by 5.4 points (C 0.516 vs B 0.570, p_Holm 0.0006).
- The dense control behaves the same (C ≈ D).
- BM25 finds the gold page where ColPali misses it on only 3 of 225 test queries.

*[I]* Read together with the literature, this is consistent with the conditions under which fusion is reported to fail rather than a contradiction of the papers where it helps:
- a large quality gap between the inputs (Bruch et al.);
- little complementarity (3/225 here vs 22/180 text-only successes in IRPAPERS);
- noisy OCR instead of clean transcriptions.

**Hypotheses for future work (not tested):**
- Fusion with BM25 should start to help when the OCR text improves, or on text-dense corpora where text retrieval is close to visual retrieval.
- On ViDoRe V2, which reduces query–page lexical overlap by design, fusion should help even less.

### Items to verify before citing
ColPali venue and App. B.4 latency numbers; Bruch et al. venue (TOIS?); DSE App. A.2 fusion method and setting; slide-study venue; IRPAPERS §4–5 (fusion table, k, significance); VisRAG Table 2.

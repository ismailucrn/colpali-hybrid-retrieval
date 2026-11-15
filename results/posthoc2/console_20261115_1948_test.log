# POST-HOC set 2 test-split results (exploratory; main test run was 2026-11-04) (n = 225 queries), nDCG@5 unless stated

| System | Role | nDCG@5 [95% CI] | Recall@5 | Recall@10 | nDCG@10 | MRR@10 | Δ vs B [95% CI] | p (perm.) | p (Holm) | W/T/L vs B |
|---|---|---|---|---|---|---|---|---|---|---|
| A: BM25 | reference | 0.3818 [0.3231, 0.4410] | 0.4444 | 0.4978 | 0.3991 | 0.3686 | – | – | – | – |
| BGE-M3 (dense text) | reference | 0.2622 [0.2087, 0.3169] | 0.3067 | 0.4178 | 0.2973 | 0.2620 | – | – | – | – |
| B: ColPali v1.3 | baseline | 0.5699 [0.5104, 0.6275] | 0.6533 | 0.6800 | 0.5789 | 0.5459 | – | – | – | – |
| T_hyb: RRF(BM25, BGE-M3) | post-hoc 2: text hybrid | 0.3851 [0.3257, 0.4442] | 0.4400 | 0.5022 | 0.4053 | 0.3751 | -0.1848 [-0.2342, -0.1366] | 0.0001 | 0.0003 | 12/140/73 |
| T_rr: T_hyb top-N reranked by cross-encoder | post-hoc 2: text RAG stack | 0.4385 [0.3770, 0.4979] | 0.4889 | 0.5289 | 0.4509 | 0.4264 | -0.1314 [-0.1804, -0.0846] | 0.0001 | 0.0003 | 20/148/57 |
| B_ce: ColPali top-N reranked by cross-encoder | post-hoc 2: text signal on ColPali | 0.5699 [0.5104, 0.6275] | 0.6533 | 0.6800 | 0.5789 | 0.5459 | +0.0000 [+0.0000, +0.0000] | 1.0000 | 1.0000 | 0/225/0 |
| Cascade: ColPali reorders BM25 top 10 | post-hoc 2: cascade (descriptive) | 0.4537 [0.3915, 0.5140] | 0.4933 | 0.4978 | 0.4551 | 0.4407 | – | – | – | – |
| Cascade: ColPali reorders BM25 top 20 | post-hoc 2: cascade (descriptive) | 0.4757 [0.4142, 0.5366] | 0.5244 | 0.5378 | 0.4802 | 0.4612 | – | – | – | – |
| Cascade: ColPali reorders BM25 top 50 | post-hoc 2: cascade (descriptive) | 0.5013 [0.4396, 0.5616] | 0.5600 | 0.5867 | 0.5098 | 0.4849 | – | – | – | – |
| Cascade: ColPali reorders BM25 top 100 | post-hoc 2: cascade (descriptive) | 0.5145 [0.4530, 0.5739] | 0.5822 | 0.6044 | 0.5215 | 0.4945 | – | – | – | – |

## Pairs (post-hoc set 2): reranker effect, hybrid vs BM25, full text stack vs BM25

| Pair | Δ [95% CI] | p (perm.) | W/T/L |
|---|---|---|---|
| T_rr vs T_hyb | +0.0533 [+0.0136, +0.0946] | 0.0116 | 36/171/18 |
| T_hyb vs A | +0.0033 [-0.0210, +0.0286] | 0.7898 | 18/185/22 |
| T_rr vs A | +0.0566 [+0.0190, +0.0956] | 0.0034 | 32/173/20 |

## By query type (exploratory, uncorrected)

| System | exact (n=93) | general (n=115) | visual (n=17) |
|---|---|---|---|
| A: BM25 | 0.617 | 0.231 | 0.118 |
| BGE-M3 (dense text) | 0.458 | 0.130 | 0.088 |
| B: ColPali v1.3 | 0.833 | 0.391 | 0.339 |
| T_hyb: RRF(BM25, BGE-M3) | 0.620 (Δ -0.213, p=0.000) | 0.238 (Δ -0.153, p=0.000) | 0.096 (Δ -0.243, p=0.030) |
| T_rr: T_hyb top-N reranked by cross-encoder | 0.694 (Δ -0.139, p=0.000) | 0.261 (Δ -0.130, p=0.000) | 0.243 (Δ -0.096, p=0.399) |
| B_ce: ColPali top-N reranked by cross-encoder | 0.833 (Δ +0.000, p=1.000) | 0.391 (Δ +0.000, p=1.000) | 0.339 (Δ +0.000, p=1.000) |
| Cascade: ColPali reorders BM25 top 10 | 0.671 | 0.322 | 0.155 |
| Cascade: ColPali reorders BM25 top 20 | 0.705 | 0.338 | 0.155 |
| Cascade: ColPali reorders BM25 top 50 | 0.748 | 0.338 | 0.251 |
| Cascade: ColPali reorders BM25 top 100 | 0.755 | 0.355 | 0.273 |

## Post-hoc: without duplicated query texts (n = 212)

| System | nDCG@5 | Δ vs B |
|---|---|---|
| A: BM25 | 0.4023 | – |
| BGE-M3 (dense text) | 0.2783 | – |
| B: ColPali v1.3 | 0.5977 | – |
| T_hyb: RRF(BM25, BGE-M3) | 0.4064 | -0.1913 |
| T_rr: T_hyb top-N reranked by cross-encoder | 0.4606 | -0.1370 |
| B_ce: ColPali top-N reranked by cross-encoder | 0.5977 | +0.0000 |
| Cascade: ColPali reorders BM25 top 10 | 0.4785 | – |
| Cascade: ColPali reorders BM25 top 20 | 0.4998 | – |
| Cascade: ColPali reorders BM25 top 50 | 0.5241 | – |
| Cascade: ColPali reorders BM25 top 100 | 0.5380 | – |

## Resources (per system; components measured on different devices, see note)

| System | Index time (s) | Index size (MB) | Query latency, sequential (ms) | parallel (ms) | Fusion overhead (ms) |
|---|---|---|---|---|---|
| A: BM25 | 0.35 | 0.63 | 0.12 | 0.12 | 0.000 |
| BGE-M3 (dense text) | 21.10 | 1.95 | 29.04 | 29.04 | 0.000 |
| B: ColPali v1.3 | 111.70 | 125.85 | 51.30 | 51.30 | 0.000 |
| T_hyb: RRF(BM25, BGE-M3) | 21.44 | 2.58 | 29.30 | 29.18 | 0.133 |
| T_rr: T_hyb top-N reranked by cross-encoder | 21.44 | 2.58 | 13624.73 | 13624.61 | 0.133 |
| B_ce: ColPali top-N reranked by cross-encoder | 111.70 | 125.85 | 51.30 | 51.30 | 0.000 |
| Cascade: ColPali reorders BM25 top 10 | 112.04 | 126.48 | 51.42 | 51.42 | 0.000 |
| Cascade: ColPali reorders BM25 top 20 | 112.04 | 126.48 | 51.42 | 51.42 | 0.000 |
| Cascade: ColPali reorders BM25 top 50 | 112.04 | 126.48 | 51.42 | 51.42 | 0.000 |
| Cascade: ColPali reorders BM25 top 100 | 112.04 | 126.48 | 51.42 | 51.42 | 0.000 |

Component latencies were measured on different devices (ColPali: L4 GPU; BGE-M3: Mac MPS; BM25 and fusion: Mac CPU). Sequential = sum of components + fusion; parallel = slowest component + fusion.

## Top-5 headroom: T_rr vs B

| Group | T_rr_hit | B_hit | T_rr_only | B_only |
|---|---|---|---|---|
| all (n=225) | 110 | 147 | 6 | 43 |
| exact (n=93) | 69 | 84 | 2 | 17 |
| general (n=115) | 36 | 56 | 3 | 23 |
| visual (n=17) | 5 | 7 | 1 | 3 |

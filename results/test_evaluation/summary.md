# Test-split results (n = 225 queries), nDCG@5 unless stated

| System | Role | nDCG@5 [95% CI] | Recall@5 | Recall@10 | nDCG@10 | MRR@10 | Δ vs B [95% CI] | p (perm.) | p (Holm) | W/T/L vs B |
|---|---|---|---|---|---|---|---|---|---|---|
| A: BM25 | text baseline | 0.3818 [0.3231, 0.4410] | 0.4444 | 0.4978 | 0.3991 | 0.3686 | -0.1881 [-0.2366, -0.1422] | 0.0001 | 0.0006 | 10/144/71 |
| BGE-M3 (dense text) | D's text component | 0.2622 [0.2087, 0.3169] | 0.3067 | 0.4178 | 0.2973 | 0.2620 | -0.3077 [-0.3637, -0.2519] | 0.0001 | 0.0006 | 5/122/98 |
| B: ColPali v1.3 | reproduced baseline | 0.5699 [0.5104, 0.6275] | 0.6533 | 0.6800 | 0.5789 | 0.5459 | – | – | – | – |
| C: RRF(BM25, ColPali) k=1 | pre-registered improvement | 0.5162 [0.4574, 0.5740] | 0.6178 | 0.6667 | 0.5324 | 0.4895 | -0.0537 [-0.0759, -0.0322] | 0.0001 | 0.0006 | 9/167/49 |
| D: RRF(BGE-M3, ColPali) k=1 | pre-registered control | 0.5187 [0.4597, 0.5778] | 0.6089 | 0.6667 | 0.5372 | 0.4959 | -0.0512 [-0.0716, -0.0298] | 0.0001 | 0.0006 | 4/171/50 |
| C_w: wRRF(BM25, ColPali) w=.95 k=20 | ablation E, dev-informed | 0.5706 [0.5109, 0.6287] | 0.6533 | 0.6800 | 0.5796 | 0.5470 | +0.0008 [-0.0021, +0.0049] | 0.9366 | 1.0000 | 2/220/3 |
| D_w: wRRF(BGE-M3, ColPali) w=.95 k=10 | ablation E, dev-informed | 0.5699 [0.5104, 0.6275] | 0.6533 | 0.6800 | 0.5789 | 0.5459 | +0.0000 [+0.0000, +0.0000] | 1.0000 | 1.0000 | 0/225/0 |

## Pairs (P3: lexical signal vs any second retriever)

| Pair | Δ [95% CI] | p (perm.) | W/T/L |
|---|---|---|---|
| C vs D | -0.0025 [-0.0268, +0.0212] | 0.8439 | 25/174/26 |
| C_w vs D_w | +0.0008 [-0.0021, +0.0049] | 0.9366 | 2/220/3 |

## By query type (exploratory, uncorrected)

| System | exact (n=93) | general (n=115) | visual (n=17) |
|---|---|---|---|
| A: BM25 | 0.617 (Δ -0.217, p=0.000) | 0.231 (Δ -0.160, p=0.000) | 0.118 (Δ -0.221, p=0.061) |
| BGE-M3 (dense text) | 0.458 (Δ -0.376, p=0.000) | 0.130 (Δ -0.261, p=0.000) | 0.088 (Δ -0.251, p=0.058) |
| B: ColPali v1.3 | 0.833 | 0.391 | 0.339 |
| C: RRF(BM25, ColPali) k=1 | 0.774 (Δ -0.059, p=0.000) | 0.345 (Δ -0.046, p=0.006) | 0.265 (Δ -0.074, p=0.128) |
| D: RRF(BGE-M3, ColPali) k=1 | 0.771 (Δ -0.063, p=0.000) | 0.343 (Δ -0.048, p=0.002) | 0.331 (Δ -0.008, p=1.000) |
| C_w: wRRF(BM25, ColPali) w=.95 k=20 | 0.837 (Δ +0.004, p=1.000) | 0.390 (Δ -0.001, p=0.743) | 0.335 (Δ -0.004, p=1.000) |
| D_w: wRRF(BGE-M3, ColPali) w=.95 k=10 | 0.833 (Δ +0.000, p=1.000) | 0.391 (Δ +0.000, p=1.000) | 0.339 (Δ +0.000, p=1.000) |

## Post-hoc: without duplicated query texts (n = 212)

| System | nDCG@5 | Δ vs B |
|---|---|---|
| A: BM25 | 0.4023 | -0.1954 |
| BGE-M3 (dense text) | 0.2783 | -0.3194 |
| B: ColPali v1.3 | 0.5977 | – |
| C: RRF(BM25, ColPali) k=1 | 0.5416 | -0.0561 |
| D: RRF(BGE-M3, ColPali) k=1 | 0.5485 | -0.0492 |
| C_w: wRRF(BM25, ColPali) w=.95 k=20 | 0.5985 | +0.0008 |
| D_w: wRRF(BGE-M3, ColPali) w=.95 k=10 | 0.5977 | +0.0000 |

## Resources (per system; components measured on different devices, see note)

| System | Index time (s) | Index size (MB) | Query latency, sequential (ms) | parallel (ms) | Fusion overhead (ms) |
|---|---|---|---|---|---|
| A: BM25 | 0.35 | 0.63 | 0.12 | 0.12 | 0.000 |
| BGE-M3 (dense text) | 21.10 | 1.95 | 29.04 | 29.04 | 0.000 |
| B: ColPali v1.3 | 111.70 | 125.85 | 51.30 | 51.30 | 0.000 |
| C: RRF(BM25, ColPali) k=1 | 112.04 | 126.48 | 51.55 | 51.43 | 0.128 |
| D: RRF(BGE-M3, ColPali) k=1 | 132.79 | 127.81 | 80.47 | 51.43 | 0.132 |
| C_w: wRRF(BM25, ColPali) w=.95 k=20 | 112.04 | 126.48 | 51.56 | 51.44 | 0.141 |
| D_w: wRRF(BGE-M3, ColPali) w=.95 k=10 | 132.79 | 127.81 | 80.47 | 51.43 | 0.133 |

Component latencies were measured on different devices (ColPali: L4 GPU; BGE-M3: Mac MPS; BM25 and fusion: Mac CPU). Sequential = sum of components + fusion; parallel = slowest component + fusion.

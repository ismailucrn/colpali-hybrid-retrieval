# POST-HOC test-split results (exploratory; main test run was 2026-11-04) (n = 225 queries), nDCG@5 unless stated

| System | Role | nDCG@5 [95% CI] | Recall@5 | Recall@10 | nDCG@10 | MRR@10 | Δ vs B [95% CI] | p (perm.) | p (Holm) | W/T/L vs B |
|---|---|---|---|---|---|---|---|---|---|---|
| A: BM25 (official tokenizer) | reference | 0.3818 [0.3231, 0.4410] | 0.4444 | 0.4978 | 0.3991 | 0.3686 | – | – | – | – |
| A_code: BM25, code-preserving tokenizer | post-hoc | 0.4052 [0.3454, 0.4647] | 0.4578 | 0.5333 | 0.4297 | 0.3978 | – | – | – | – |
| B: ColPali v1.3 | baseline | 0.5699 [0.5104, 0.6275] | 0.6533 | 0.6800 | 0.5789 | 0.5459 | – | – | – | – |
| C: RRF(BM25, ColPali) k=1 | reference (main run) | 0.5162 [0.4574, 0.5740] | 0.6178 | 0.6667 | 0.5324 | 0.4895 | – | – | – | – |
| C_code: RRF(A_code, ColPali) k=1 | post-hoc | 0.5253 [0.4666, 0.5845] | 0.6133 | 0.6756 | 0.5459 | 0.5049 | -0.0446 [-0.0658, -0.0236] | 0.0001 | 0.0003 | 9/171/45 |
| C_cc: CC(BM25, ColPali) | post-hoc | 0.5699 [0.5104, 0.6275] | 0.6533 | 0.6800 | 0.5789 | 0.5459 | +0.0000 [+0.0000, +0.0000] | 1.0000 | 1.0000 | 0/225/0 |
| D_cc: CC(BGE-M3, ColPali) | post-hoc control | 0.5699 [0.5104, 0.6275] | 0.6533 | 0.6800 | 0.5789 | 0.5459 | +0.0000 [+0.0000, +0.0000] | 1.0000 | 1.0000 | 0/225/0 |

## Pairs (post-hoc): lexical vs dense under CC; tokenizer effect alone and inside RRF

| Pair | Δ [95% CI] | p (perm.) | W/T/L |
|---|---|---|---|
| C_cc vs D_cc | +0.0000 [+0.0000, +0.0000] | 1.0000 | 0/225/0 |
| A_code vs A | +0.0234 [+0.0039, +0.0455] | 0.0274 | 14/206/5 |
| C_code vs C | +0.0091 [-0.0009, +0.0202] | 0.0907 | 12/207/6 |

## By query type (exploratory, uncorrected)

| System | exact (n=93) | general (n=115) | visual (n=17) |
|---|---|---|---|
| A: BM25 (official tokenizer) | 0.617 | 0.231 | 0.118 |
| A_code: BM25, code-preserving tokenizer | 0.667 | 0.236 | 0.118 |
| B: ColPali v1.3 | 0.833 | 0.391 | 0.339 |
| C: RRF(BM25, ColPali) k=1 | 0.774 | 0.345 | 0.265 |
| C_code: RRF(A_code, ColPali) k=1 | 0.789 (Δ -0.044, p=0.002) | 0.349 (Δ -0.041, p=0.013) | 0.272 (Δ -0.067, p=0.128) |
| C_cc: CC(BM25, ColPali) | 0.833 (Δ +0.000, p=1.000) | 0.391 (Δ +0.000, p=1.000) | 0.339 (Δ +0.000, p=1.000) |
| D_cc: CC(BGE-M3, ColPali) | 0.833 (Δ +0.000, p=1.000) | 0.391 (Δ +0.000, p=1.000) | 0.339 (Δ +0.000, p=1.000) |

## Post-hoc: without duplicated query texts (n = 212)

| System | nDCG@5 | Δ vs B |
|---|---|---|
| A: BM25 (official tokenizer) | 0.4023 | – |
| A_code: BM25, code-preserving tokenizer | 0.4271 | – |
| B: ColPali v1.3 | 0.5977 | – |
| C: RRF(BM25, ColPali) k=1 | 0.5416 | – |
| C_code: RRF(A_code, ColPali) k=1 | 0.5510 | -0.0467 |
| C_cc: CC(BM25, ColPali) | 0.5977 | +0.0000 |
| D_cc: CC(BGE-M3, ColPali) | 0.5977 | +0.0000 |

## Resources (per system; components measured on different devices, see note)

| System | Index time (s) | Index size (MB) | Query latency, sequential (ms) | parallel (ms) | Fusion overhead (ms) |
|---|---|---|---|---|---|
| A: BM25 (official tokenizer) | 0.35 | 0.63 | 0.12 | 0.12 | 0.000 |
| A_code: BM25, code-preserving tokenizer | 0.04 | 0.78 | 0.09 | 0.09 | 0.000 |
| B: ColPali v1.3 | 111.70 | 125.85 | 51.30 | 51.30 | 0.000 |
| C: RRF(BM25, ColPali) k=1 | 112.04 | 126.48 | 51.56 | 51.43 | 0.135 |
| C_code: RRF(A_code, ColPali) k=1 | 111.73 | 126.63 | 51.52 | 51.43 | 0.135 |
| C_cc: CC(BM25, ColPali) | 112.04 | 126.48 | 51.43 | 51.31 | 0.007 |
| D_cc: CC(BGE-M3, ColPali) | 132.79 | 127.81 | 80.35 | 51.31 | 0.007 |

Component latencies were measured on different devices (ColPali: L4 GPU; BGE-M3: Mac MPS; BM25 and fusion: Mac CPU). Sequential = sum of components + fusion; parallel = slowest component + fusion.

## Top-5 headroom: A vs B

| Group | A_hit | B_hit | A_only | B_only |
|---|---|---|---|---|
| all (n=225) | 100 | 147 | 3 | 50 |
| exact (n=93) | 62 | 84 | 0 | 22 |
| general (n=115) | 36 | 56 | 3 | 23 |
| visual (n=17) | 2 | 7 | 0 | 5 |

## Top-5 headroom: A_code vs B

| Group | A_code_hit | B_hit | A_code_only | B_only |
|---|---|---|---|---|
| all (n=225) | 103 | 147 | 3 | 47 |
| exact (n=93) | 67 | 84 | 0 | 17 |
| general (n=115) | 34 | 56 | 3 | 25 |
| visual (n=17) | 2 | 7 | 0 | 5 |

# POST-HOC set 4 test-split results (exploratory; main test run was 2026-11-04) (n = 225 queries), nDCG@5 unless stated

| System | Role | nDCG@5 [95% CI] | Recall@5 | Recall@10 | nDCG@10 | MRR@10 | Δ vs B [95% CI] | p (perm.) | p (Holm) | W/T/L vs B |
|---|---|---|---|---|---|---|---|---|---|---|
| B: ColPali v1.3 | baseline | 0.5699 [0.5104, 0.6275] | 0.6533 | 0.6800 | 0.5789 | 0.5459 | – | – | – | – |
| Q8: Qwen3-VL-Embedding-8B | post-hoc 4: descriptive | 0.5009 [0.4394, 0.5606] | 0.5644 | 0.6400 | 0.5269 | 0.4913 | – | – | – | – |
| CQ: ColQwen2.5 (v0.2) | post-hoc 4: stronger retriever | 0.6177 [0.5583, 0.6752] | 0.6889 | 0.7422 | 0.6349 | 0.6016 | +0.0478 [+0.0111, +0.0857] | 0.0105 | 0.0315 | 36/166/23 |
| TM: Tomoro ColQwen3-4B | post-hoc 4: stronger retriever | 0.6360 [0.5777, 0.6926] | 0.7156 | 0.7822 | 0.6581 | 0.6192 | +0.0661 [+0.0298, +0.1033] | 0.0006 | 0.0024 | 39/168/18 |
| F_cq: RRF(ColPali, ColQwen2.5) | post-hoc 4: second visual retriever | 0.6125 [0.5543, 0.6689] | 0.6933 | 0.7333 | 0.6259 | 0.5914 | +0.0426 [+0.0203, +0.0673] | 0.0002 | 0.0010 | 32/183/10 |
| F_tm: RRF(ColPali, Tomoro) | post-hoc 4: second visual retriever | 0.6129 [0.5541, 0.6693] | 0.7022 | 0.7689 | 0.6345 | 0.5921 | +0.0430 [+0.0223, +0.0644] | 0.0001 | 0.0006 | 31/186/8 |
| F_q8: RRF(ColPali, Qwen3-VL-Embedding-8B) | post-hoc 4: second visual retriever | 0.5546 [0.4944, 0.6132] | 0.6356 | 0.6933 | 0.5740 | 0.5362 | -0.0153 [-0.0338, +0.0040] | 0.1185 | 0.1185 | 9/187/29 |
| B_vr8: ColPali top-N reranked by Qwen3-VL-Reranker-8B | post-hoc 4: visual reranker | 0.6018 [0.5416, 0.6598] | 0.6667 | 0.6800 | 0.6064 | 0.5821 | +0.0319 [+0.0050, +0.0607] | 0.0239 | 0.0478 | 29/175/21 |

## Pairs (post-hoc set 4): Qwen3-VL-Embedding-8B vs ColPali; Tomoro vs ColQwen2.5

| Pair | Δ [95% CI] | p (perm.) | W/T/L |
|---|---|---|---|
| Q8 vs B | -0.0689 [-0.1073, -0.0321] | 0.0007 | 19/161/45 |
| TM vs CQ | +0.0182 [-0.0171, +0.0548] | 0.3390 | 33/167/25 |

## By query type (exploratory, uncorrected)

| System | exact (n=93) | general (n=115) | visual (n=17) |
|---|---|---|---|
| B: ColPali v1.3 | 0.833 | 0.391 | 0.339 |
| Q8: Qwen3-VL-Embedding-8B | 0.749 | 0.342 | 0.221 |
| CQ: ColQwen2.5 (v0.2) | 0.874 (Δ +0.041, p=0.019) | 0.432 (Δ +0.041, p=0.172) | 0.472 (Δ +0.133, p=0.348) |
| TM: Tomoro ColQwen3-4B | 0.872 (Δ +0.039, p=0.035) | 0.481 (Δ +0.090, p=0.004) | 0.391 (Δ +0.052, p=0.567) |
| F_cq: RRF(ColPali, ColQwen2.5) | 0.866 (Δ +0.033, p=0.001) | 0.438 (Δ +0.047, p=0.017) | 0.405 (Δ +0.066, p=0.314) |
| F_tm: RRF(ColPali, Tomoro) | 0.861 (Δ +0.027, p=0.008) | 0.453 (Δ +0.062, p=0.000) | 0.339 (Δ +0.000, p=1.000) |
| F_q8: RRF(ColPali, Qwen3-VL-Embedding-8B) | 0.811 (Δ -0.022, p=0.111) | 0.380 (Δ -0.011, p=0.468) | 0.331 (Δ -0.008, p=0.746) |
| B_vr8: ColPali top-N reranked by Qwen3-VL-Reranker-8B | 0.895 (Δ +0.062, p=0.003) | 0.389 (Δ -0.002, p=0.901) | 0.437 (Δ +0.098, p=0.315) |

## Post-hoc: without duplicated query texts (n = 212)

| System | nDCG@5 | Δ vs B |
|---|---|---|
| B: ColPali v1.3 | 0.5977 | – |
| Q8: Qwen3-VL-Embedding-8B | 0.5196 | – |
| CQ: ColQwen2.5 (v0.2) | 0.6408 | +0.0432 |
| TM: Tomoro ColQwen3-4B | 0.6661 | +0.0684 |
| F_cq: RRF(ColPali, ColQwen2.5) | 0.6403 | +0.0426 |
| F_tm: RRF(ColPali, Tomoro) | 0.6397 | +0.0421 |
| F_q8: RRF(ColPali, Qwen3-VL-Embedding-8B) | 0.5812 | -0.0165 |
| B_vr8: ColPali top-N reranked by Qwen3-VL-Reranker-8B | 0.6320 | +0.0343 |

## Resources (per system; components measured on different devices, see note)

| System | Index time (s) | Index size (MB) | Query latency, sequential (ms) | parallel (ms) | Fusion overhead (ms) |
|---|---|---|---|---|---|
| B: ColPali v1.3 | 111.70 | 125.85 | 51.30 | 51.30 | 0.000 |
| Q8: Qwen3-VL-Embedding-8B | 574.55 | 7.81 | 76.72 | 76.72 | 0.000 |
| CQ: ColQwen2.5 (v0.2) | 213.52 | 92.55 | 164.47 | 164.47 | 0.000 |
| TM: Tomoro ColQwen3-4B | 256.98 | 372.64 | 68.44 | 68.44 | 0.000 |
| F_cq: RRF(ColPali, ColQwen2.5) | 325.22 | 218.41 | 215.90 | 164.60 | 0.133 |
| F_tm: RRF(ColPali, Tomoro) | 368.67 | 498.49 | 119.87 | 68.57 | 0.135 |
| F_q8: RRF(ColPali, Qwen3-VL-Embedding-8B) | 686.25 | 133.67 | 128.15 | 76.85 | 0.133 |
| B_vr8: ColPali top-N reranked by Qwen3-VL-Reranker-8B | 111.70 | 125.85 | 7462.00 | 7462.00 | 0.000 |

Component latencies were measured on different devices (ColPali: L4 GPU; BGE-M3: Mac MPS; BM25 and fusion: Mac CPU). Sequential = sum of components + fusion; parallel = slowest component + fusion.

## Top-5 headroom: CQ vs B

| Group | CQ_hit | B_hit | CQ_only | B_only |
|---|---|---|---|---|
| all (n=225) | 155 | 147 | 16 | 8 |
| exact (n=93) | 87 | 84 | 4 | 1 |
| general (n=115) | 58 | 56 | 8 | 6 |
| visual (n=17) | 10 | 7 | 4 | 1 |

## Top-5 headroom: TM vs B

| Group | TM_hit | B_hit | TM_only | B_only |
|---|---|---|---|---|
| all (n=225) | 161 | 147 | 22 | 8 |
| exact (n=93) | 86 | 84 | 3 | 1 |
| general (n=115) | 65 | 56 | 15 | 6 |
| visual (n=17) | 10 | 7 | 4 | 1 |

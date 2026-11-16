# POST-HOC set 3 test-split results (exploratory; main test run was 2026-11-04) (n = 225 queries), nDCG@5 unless stated

| System | Role | nDCG@5 [95% CI] | Recall@5 | Recall@10 | nDCG@10 | MRR@10 | Δ vs B [95% CI] | p (perm.) | p (Holm) | W/T/L vs B |
|---|---|---|---|---|---|---|---|---|---|---|
| A: BM25 (Tesseract text) | reference | 0.3818 [0.3231, 0.4410] | 0.4444 | 0.4978 | 0.3991 | 0.3686 | – | – | – | – |
| B: ColPali v1.3 | baseline | 0.5699 [0.5104, 0.6275] | 0.6533 | 0.6800 | 0.5789 | 0.5459 | – | – | – | – |
| A_vlm: BM25 over Qwen3.5-4B transcription | post-hoc 3: H3 component | 0.4277 [0.3653, 0.4869] | 0.4889 | 0.5422 | 0.4447 | 0.4144 | – | – | – | – |
| Q3E: Qwen3-VL-Embedding-2B | post-hoc 3: H4 component | 0.4709 [0.4107, 0.5311] | 0.5422 | 0.6133 | 0.4942 | 0.4570 | – | – | – | – |
| C: RRF(BM25, ColPali) | main C (frozen 2026-11-04) | 0.5162 [0.4574, 0.5740] | 0.6178 | 0.6667 | 0.5324 | 0.4895 | – | – | – | – |
| C_sel: C where BM25 margin >= tau, else B | post-hoc 3: H2 | 0.5677 [0.5083, 0.6252] | 0.6533 | 0.6800 | 0.5767 | 0.5430 | -0.0022 [-0.0067, +0.0000] | 1.0000 | 1.0000 | 0/224/1 |
| B_vr: ColPali top-N reranked by Qwen3-VL-Reranker-2B | post-hoc 3: H1 | 0.5809 [0.5224, 0.6390] | 0.6622 | 0.7200 | 0.6002 | 0.5622 | +0.0111 [-0.0182, +0.0391] | 0.4619 | 0.9237 | 28/173/24 |
| C_vlm: RRF(A_vlm, ColPali) | post-hoc 3: H3 | 0.5350 [0.4744, 0.5932] | 0.6178 | 0.6711 | 0.5524 | 0.5148 | -0.0349 [-0.0568, -0.0120] | 0.0027 | 0.0108 | 11/172/42 |
| E_ens: RRF(Q3E, ColPali) | post-hoc 3: H4 | 0.5580 [0.4975, 0.6168] | 0.6356 | 0.6889 | 0.5752 | 0.5396 | -0.0119 [-0.0312, +0.0081] | 0.2366 | 0.7097 | 13/183/29 |

## Pairs (post-hoc set 3): VLM transcription vs Tesseract for BM25; Qwen3-VL embedding vs ColPali

| Pair | Δ [95% CI] | p (perm.) | W/T/L |
|---|---|---|---|
| A_vlm vs A | +0.0458 [+0.0186, +0.0757] | 0.0016 | 22/191/12 |
| Q3E vs B | -0.0990 [-0.1395, -0.0604] | 0.0001 | 16/157/52 |

## By query type (exploratory, uncorrected)

| System | exact (n=93) | general (n=115) | visual (n=17) |
|---|---|---|---|
| A: BM25 (Tesseract text) | 0.617 | 0.231 | 0.118 |
| B: ColPali v1.3 | 0.833 | 0.391 | 0.339 |
| A_vlm: BM25 over Qwen3.5-4B transcription | 0.652 | 0.282 | 0.184 |
| Q3E: Qwen3-VL-Embedding-2B | 0.710 | 0.330 | 0.119 |
| C: RRF(BM25, ColPali) | 0.774 | 0.345 | 0.265 |
| C_sel: C where BM25 margin >= tau, else B | 0.828 (Δ -0.005, p=1.000) | 0.391 (Δ +0.000, p=1.000) | 0.339 (Δ +0.000, p=1.000) |
| B_vr: ColPali top-N reranked by Qwen3-VL-Reranker-2B | 0.856 (Δ +0.022, p=0.122) | 0.384 (Δ -0.007, p=0.774) | 0.409 (Δ +0.070, p=0.491) |
| C_vlm: RRF(A_vlm, ColPali) | 0.779 (Δ -0.054, p=0.000) | 0.372 (Δ -0.019, p=0.323) | 0.302 (Δ -0.037, p=0.507) |
| E_ens: RRF(Q3E, ColPali) | 0.810 (Δ -0.023, p=0.130) | 0.393 (Δ +0.002, p=0.912) | 0.298 (Δ -0.041, p=0.123) |

## Post-hoc: without duplicated query texts (n = 212)

| System | nDCG@5 | Δ vs B |
|---|---|---|
| A: BM25 (Tesseract text) | 0.4023 | – |
| B: ColPali v1.3 | 0.5977 | – |
| A_vlm: BM25 over Qwen3.5-4B transcription | 0.4492 | – |
| Q3E: Qwen3-VL-Embedding-2B | 0.4921 | – |
| C: RRF(BM25, ColPali) | 0.5416 | – |
| C_sel: C where BM25 margin >= tau, else B | 0.5953 | -0.0024 |
| B_vr: ColPali top-N reranked by Qwen3-VL-Reranker-2B | 0.6092 | +0.0115 |
| C_vlm: RRF(A_vlm, ColPali) | 0.5589 | -0.0388 |
| E_ens: RRF(Q3E, ColPali) | 0.5831 | -0.0146 |

## Resources (per system; components measured on different devices, see note)

| System | Index time (s) | Index size (MB) | Query latency, sequential (ms) | parallel (ms) | Fusion overhead (ms) |
|---|---|---|---|---|---|
| A: BM25 (Tesseract text) | 0.35 | 0.63 | 0.12 | 0.12 | 0.000 |
| B: ColPali v1.3 | 111.70 | 125.85 | 51.30 | 51.30 | 0.000 |
| A_vlm: BM25 over Qwen3.5-4B transcription | 0.46 | 0.60 | 0.13 | 0.13 | 0.000 |
| Q3E: Qwen3-VL-Embedding-2B | 286.29 | 3.91 | 51.69 | 51.69 | 0.000 |
| C: RRF(BM25, ColPali) | 112.04 | 126.48 | 51.56 | 51.43 | 0.136 |
| C_sel: C where BM25 margin >= tau, else B | 112.04 | 126.48 | 51.56 | 51.43 | 0.136 |
| B_vr: ColPali top-N reranked by Qwen3-VL-Reranker-2B | 111.70 | 125.85 | 7177.05 | 7177.05 | 0.000 |
| C_vlm: RRF(A_vlm, ColPali) | 112.15 | 126.45 | 51.56 | 51.43 | 0.135 |
| E_ens: RRF(Q3E, ColPali) | 397.99 | 129.76 | 103.13 | 51.83 | 0.141 |

Component latencies were measured on different devices (ColPali: L4 GPU; BGE-M3: Mac MPS; BM25 and fusion: Mac CPU). Sequential = sum of components + fusion; parallel = slowest component + fusion.

## Top-5 headroom: A_vlm vs B

| Group | A_vlm_hit | B_hit | A_vlm_only | B_only |
|---|---|---|---|---|
| all (n=225) | 110 | 147 | 3 | 40 |
| exact (n=93) | 65 | 84 | 0 | 19 |
| general (n=115) | 41 | 56 | 3 | 18 |
| visual (n=17) | 4 | 7 | 0 | 3 |

## Top-5 headroom: Q3E vs B

| Group | Q3E_hit | B_hit | Q3E_only | B_only |
|---|---|---|---|---|
| all (n=225) | 122 | 147 | 6 | 31 |
| exact (n=93) | 70 | 84 | 1 | 15 |
| general (n=115) | 49 | 56 | 5 | 12 |
| visual (n=17) | 3 | 7 | 0 | 4 |

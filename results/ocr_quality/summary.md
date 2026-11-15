# OCR quality vs retrieval (POST-HOC set 2, descriptive, full set n = 451)

Proxy: share of alphabetic OCR tokens found in the NLTK English word list (234377 words).
Pages: median 0.731, 10th–90th percentile 0.525–0.843; 4 pages with empty OCR.

| Gold-page proxy group | n | proxy range | BM25 Hit@5 | BGE-M3 Hit@5 | ColPali Hit@5 | BM25 nDCG@5 | BGE-M3 nDCG@5 | ColPali nDCG@5 |
|---|---|---|---|---|---|---|---|---|
| low | 149 | 0.100–0.685 | 0.416 | 0.282 | 0.631 | 0.369 | 0.253 | 0.565 |
| mid | 151 | 0.686–0.778 | 0.503 | 0.318 | 0.709 | 0.431 | 0.271 | 0.618 |
| high | 147 | 0.779–0.958 | 0.442 | 0.259 | 0.605 | 0.382 | 0.214 | 0.540 |
| empty OCR | 4 | – | 0.000 | 1.000 | 0.750 | 0.000 | 0.515 | 0.625 |

| System | Spearman ρ (proxy vs nDCG@5) | 95% CI |
|---|---|---|
| BM25 | +0.027 | [-0.065, +0.120] |
| BGE-M3 | -0.019 | [-0.108, +0.073] |
| ColPali | +0.008 | [-0.087, +0.104] |

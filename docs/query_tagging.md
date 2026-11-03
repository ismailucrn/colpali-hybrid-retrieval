# Query-type tagging rules and dev/test split

**Status: FROZEN on 3 November 2026**, together with `results/splits/docvqa_query_tags.csv` and `results/splits/docvqa_split.json`, in the same commit. This happened **before** any run of systems A, C, D or E. Any later change is added below as a dated note and is never made as a silent edit.

## Why

The failure analysis asks *for which query types* fusion helps. If tags were assigned after looking at system outputs, the analysis could be biased toward the hypothesis. So the tags are computed automatically from the **query text only**, using the fixed rules in `src/query_types.py`. No manual judgement is involved, which also makes the tags exactly reproducible.

Honest note on timing: System B (ColPali) per-query results already existed when these rules were written. The rules were designed by reading a random sample of query texts only. Per-query ColPali results were not consulted, and the rules use nothing but the query string.

## Tags (one per query; order of precedence: visual > exact > general)

| Tag | Rule (query text only) | Examples from DocVQA |
|---|---|---|
| `visual` | Contains a visual-element term: chart, graph, plot(ted), figure/fig, diagram, map, image, picture, photo(graph), logo, axis, x-/y-axis, legend, bar, pie, illustration, drawing/drawn, monogram, stamp, handwritten, colo(u)r, bold, symbol, icon. Case-insensitive; plural allowed. | "In fig 3, what is plotted along x axis?" · "Whose Logo is it?" |
| `exact` | Contains a **lexical anchor** that can be matched literally on the page: a digit; an acronym or code (≥ 2 capitals, `&` allowed); a quoted span; or a capitalised word that is not the first word (a name or a field label). | "dividend payout in 2012" · "full form of FERS" · "the Log-in No." · "Charles Muller" |
| `general` | None of the above. | "Who is the letter from?" · "What is the heading?" |

What `exact` means: the *query itself* carries a string that BM25 can match. It does **not** mean that the *answer* is a number. For example, "What is the page number?" is `general`, because the query contains nothing specific to match.

Known weaknesses (to report in the limitations section):

- Rare lowercase terms, e.g. "teaching of epidomiology", are not detected as anchors. Detecting them would need corpus statistics.
- Capitalised generic field labels ("Budget Estimate", "Correspondence") count as anchors.
- A quoted common word, e.g. ‘page’, counts as an anchor.

## Split

- Unit: unique query (451 after official deduplication). Each query goes to exactly one split.
- Stratified by tag. Within each tag, queries are shuffled with `numpy.random.RandomState(0)` and the first `round(n × 0.5)` go to dev. Command: `python scripts/make_split.py --seed 0 --dev-fraction 0.5`.
- **Dev (226):** used to choose RRF `k`, fusion weights and any other setting.
- **Test (225):** touched **once**, at the end, with the settings chosen on dev.
- The published-vs-measured reproduction (System B) uses the **full** 451-query set, as the paper does.

| Split | exact | general | visual | total | of which duplicated query texts |
|---|---|---|---|---|---|
| dev | 93 | 115 | 18 | 226 | 9 |
| test | 93 | 115 | 17 | 225 | 13 |
| all | 186 | 230 | 35 | 451 | 22 |

Rules that fired across all 451 queries: no rule 230, capitalised word 77, digit 40, acronym 35, visual term 35, quoted span 34.

*Limitation:* the visual group is small, about 17 queries per split, so per-type results for `visual` will have wide confidence intervals.

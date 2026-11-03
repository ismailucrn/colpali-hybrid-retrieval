# Query-type tagging rules

**Status: DRAFT.** These rules must be frozen and committed, and the tag file produced, **before any retrieval system is run** on the chosen task. After freezing, changes are added as dated notes, never silent edits.

## Why

The failure analysis asks *for which query types* fusion helps. If tags were assigned after looking at system outputs, the analysis could be biased toward the hypothesis. Tags therefore depend only on the query text (and, where a rule says so, on the gold page), never on retrieval results.

## Tags (one per query)

| Tag | Definition | Examples (illustrative, not from the dataset) |
|---|---|---|
| `exact` | The answer requires matching a specific token string: a number, date, year, percentage, monetary amount, code/ID, proper name, acronym, or rare technical term. | "revenue in 2019", "what does ISO 27001 require" |
| `visual` | The query is about a chart, plot, figure, diagram, table layout or other visual element, or the gold page is mainly non-text. | "trend shown in the bar chart", "which region is highlighted on the map" |
| `general` | Neither of the above: conceptual or descriptive questions answerable from running text. | "what are the benefits of the program" |

## Decision procedure

1. If the query explicitly refers to a visual element (chart, graph, figure, plot, diagram, map, image) → `visual`.
2. Else, if it contains a number, date, code/ID, proper name, acronym or rare technical term that must be matched → `exact`.
3. Else → `general`.

Rule order is fixed (visual before exact) so that "the 2019 value in the bar chart" is `visual`. *Open question to settle before freezing: whether a gold page that is mainly a figure should override rule 2.*

## Procedure

- Tagger: the project author (manual), optionally assisted by a regex pre-pass for numbers/acronyms; the final tag is manual.
- Output: `results/query_tags.csv` with columns `query_id, tag, rule_applied, note`.
- Freeze: commit the rules and the tag file together, before the first retrieval run; record the commit hash in `docs/experiment_log.md`.
- Report the tag distribution and a few ambiguous cases in the report.

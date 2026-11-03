"""Loading ViDoRe v1 QA-format tasks with the official query/qrels construction."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RetrievalTask:
    """One ViDoRe QA-format task as the official evaluator sees it.

    - `page_ids`: one id per row (the image filename), in dataset order.
    - `queries`: unique non-null queries in first-occurrence order (official deduplication).
    - `qrels`: {query: {page_id: 1}} with exactly one relevant page per query. For query texts that
      occur on several rows, the official code keeps the page of the *last* occurrence
      (dict-comprehension overwrite in `ViDoReEvaluatorQA._get_relevant_docs_results`); we do the same.
    - `duplicated_queries`: query texts that occur on more than one row (ambiguous gold page).
    """

    page_ids: list[str]
    queries: list[str]
    qrels: dict[str, dict[str, int]]
    duplicated_queries: set[str] = field(default_factory=set)
    images: list | None = None
    page_texts: list[str] | None = None


def build_task(query_column: list, page_ids: list[str]) -> RetrievalTask:
    if len(set(page_ids)) != len(page_ids):
        # The official evaluator merges rows sharing a filename with a max over scores; our tasks
        # have unique filenames, so we fail loudly instead of silently implementing that case.
        raise ValueError("duplicate page ids are not supported")

    queries, seen, counts = [], set(), {}
    for q in query_column:
        if q is None:
            continue
        counts[q] = counts.get(q, 0) + 1
        if q not in seen:
            seen.add(q)
            queries.append(q)

    qrels = {q: {pid: 1} for q, pid in zip(query_column, page_ids) if q is not None}
    duplicated = {q for q, c in counts.items() if c > 1}
    return RetrievalTask(page_ids=list(page_ids), queries=queries, qrels=qrels, duplicated_queries=duplicated)


def load_task(dataset: str, revision: str | None, split: str, with_images: bool = False, text_dataset: str | None = None,
              text_revision: str | None = None) -> RetrievalTask:
    """Load a ViDoRe QA task. Optionally attach page images and/or OCR page text from a `_tesseract` variant."""
    from datasets import load_dataset

    ds = load_dataset(dataset, split=split, revision=revision)
    columns = ["query", "image_filename"] + (["image"] if with_images else [])
    ds = ds.select_columns(columns)
    task = build_task(ds["query"], ds["image_filename"])

    if with_images:
        task.images = [img.convert("RGB") for img in ds["image"]]

    if text_dataset is not None:
        tds = load_dataset(text_dataset, split=split, revision=text_revision).select_columns(
            ["image_filename", "text_description"]
        )
        if tds["image_filename"] != task.page_ids:
            raise ValueError(f"{text_dataset} rows are not aligned with {dataset}")
        task.page_texts = [t or "" for t in tds["text_description"]]

    return task

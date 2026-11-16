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
    row_page: list[int] | None = None   # row -> index into page_ids (set when rows may share a page)
    row_texts: list[str] | None = None  # OCR text per row (BM25 indexes rows, as the official evaluator does)


def build_task(query_column: list, page_ids: list[str], allow_duplicate_pages: bool = False) -> RetrievalTask:
    """Official query/qrels construction. `page_ids` has one entry per dataset row.

    With allow_duplicate_pages=True, rows that share an image filename become one page (first-occurrence order) and
    `row_page` maps every row to its page; scores computed per row are merged with `max_over_rows`, as the official
    evaluator does (it keeps the max score over rows with the same filename).
    """
    if len(set(page_ids)) != len(page_ids) and not allow_duplicate_pages:
        # DocVQA and InfoVQA have unique filenames; other tasks must opt in explicitly.
        raise ValueError("duplicate page ids are not supported")
    unique = list(dict.fromkeys(page_ids))
    position = {pid: i for i, pid in enumerate(unique)}

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
    return RetrievalTask(page_ids=unique, queries=queries, qrels=qrels, duplicated_queries=duplicated,
                         row_page=[position[pid] for pid in page_ids])


def max_over_rows(row_scores, row_page: list[int], n_pages: int):
    """[n_queries, n_rows] -> [n_queries, n_pages], max over rows of the same page (official merge).

    The official loop seeds new filenames with 0 (`max(results.get(filename, 0), score)`), which only matters for
    negative scores; we refuse those instead of reproducing the quirk.
    """
    import numpy as np

    row_scores = np.asarray(row_scores)
    if (row_scores < 0).any():
        raise ValueError("negative scores: the official max-merge would clamp them to 0")
    out = np.full((row_scores.shape[0], n_pages), -np.inf)
    for r, p in enumerate(row_page):
        np.maximum(out[:, p], row_scores[:, r], out=out[:, p])
    return out


def load_task_columns(dataset: str, revision: str, split: str, text_dataset: str | None = None,
                      text_revision: str | None = None) -> RetrievalTask:
    """Light CPU loader: reads only the query / filename (/ OCR text) parquet columns from the Hub at a pinned revision,
    without downloading page images. Rows may share a page (see build_task). Row order = sorted shard order, as in
    `datasets`; every score file is checked against this order when it is loaded."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()

    def read(repo: str, rev: str, columns: list[str]) -> dict[str, list]:
        files = sorted(fs.glob(f"datasets/{repo}@{rev}/**/{split}-*.parquet"))
        if not files:
            raise FileNotFoundError(f"no {split} parquet files in {repo}@{rev}")
        out = {c: [] for c in columns}
        for f in files:
            with fs.open(f) as fh:
                table = pq.read_table(fh, columns=columns)
            for c in columns:
                out[c] += table[c].to_pylist()
        return out

    base = read(dataset, revision, ["query", "image_filename"])
    task = build_task(base["query"], base["image_filename"], allow_duplicate_pages=True)
    if text_dataset is not None:
        t = read(text_dataset, text_revision, ["query", "image_filename", "text_description"])
        if t["image_filename"] != base["image_filename"] or t["query"] != base["query"]:
            raise ValueError(f"{text_dataset} rows are not aligned with {dataset}")
        task.row_texts = [x or "" for x in t["text_description"]]
    return task


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

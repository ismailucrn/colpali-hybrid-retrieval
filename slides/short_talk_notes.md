# Speaker notes: 5-minute talk (`slides/short_talk.pdf`)

The instructor asked for a short talk that introduces the topic to classmates. The detail of what was tried and what worked belongs in the report. Aim for about 45 seconds per slide.

**1 · Title (10 s).** Name the topic in one sentence: searching document pages as images, and whether keyword search still helps.

**2 · The problem (50 s).**
- RAG over PDFs needs a retriever that finds the right page before any answer is generated.
- The usual way first turns the page into text: layout detection, OCR, chunks, then an embedding or BM25.
- This is slow, it loses tables, charts and layout, and every OCR mistake becomes a search mistake.
- Point out the two modalities: the document is an image, the question is text.

**3 · ColPali (60 s).**
- ColPali feeds the page *image* to a vision–language model (PaliGemma-3B), which gives about a thousand patch vectors per page.
- The question gives one vector per word.
- The score is MaxSim: each word looks for its best-matching patch, and the maxima are summed. The same idea is used for text in ColBERT.
- So a word on the page is matched where it is printed, without OCR.
- Published: on the ViDoRe benchmark it beats every OCR pipeline (81.3 vs 67.0 average nDCG@5). Newer models in the same family exist: ColQwen, Tomoro.

**4 · My question (60 s).**
- In text search, adding keyword search to semantic search ("hybrid search") is a cheap and standard improvement.
- My question: does adding BM25 over the page's OCR text still help ColPali? The intuition was exact strings such as invoice numbers or codes.
- How I tested it, in one breath:
  - I reproduced ColPali with the authors' code;
  - I fused it with BM25 and added a control with a dense text model;
  - I wrote the predictions down in advance, tuned on a dev split and used the test split once;
  - I followed up with stronger text pipelines, newer Qwen models and nine more ViDoRe tasks.

**5 · What came out (50 s).**
- Read the chart from left to right: text-only systems are well below ColPali, and adding BM25 to ColPali lowers it. Only newer visual models go above it.
- The reproduction matched the authors' number within one point.
- Adding text never helped, on DocVQA or on nine other tasks.
- A newer visual model (Tomoro) did help, and the gain held on a task that was never used for tuning.

**6 · Take-aways (50 s).**
- Visual document retrievers already "read" the text, so OCR keyword search mostly finds pages they already find.
- Fusion is not free. It helps when both rankers are comparably strong, and a much weaker one imports its errors. Text-retrieval research reports the same thing.
- Method: a control, a test set used once and a fresh task turned a negative result into a clear answer.
- Point to the repository for the report and the full details.

**If asked for detail:** the full numbers are in the report (sections 7.1–7.11), and the detailed deck is `slides/slides.pdf`. Its speaker notes and likely questions are in `slides/speaker_notes.md`.

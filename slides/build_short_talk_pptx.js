// Builds slides/short_talk.pptx: an editable PowerPoint version of the 5-minute talk (slides/short_talk.tex).
// Measured numbers are read from results/*.json at build time; published numbers are typed and cited on the slide.
// Usage (from the repository root):  npm install pptxgenjs && node slides/build_short_talk_pptx.js
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const load = (p) => JSON.parse(fs.readFileSync(path.join(ROOT, p), "utf8"));
const f3 = (x) => x.toFixed(3);
const sgn = (x) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(3);

const full = load("results/colpali_v1.3_docvqa/metrics.json");
const test = load("results/test_evaluation/metrics.json");
const p2 = load("results/posthoc2/metrics.json");
const p4 = load("results/posthoc4/metrics.json");
const fresh = load("results/posthoc4/fresh_task_infovqa.json");
const nd = (r) => r.metrics.ndcg_at_5;
const N = {
  fullB: f3(full.metrics.ndcg_at_5),
  pubCard: f3(full.published["ndcg_at_5_model_card_v1.3"]),
  deltaC: sgn(test.vs_baseline.C.delta),
  deltaTM: sgn(p4.vs_baseline.TM.delta),
  infoDelta: sgn(fresh.delta),
};
const BARS = [
  ["BM25 (OCR text)", nd(test.results.A)],
  ["Text RAG stack", nd(p2.results.T_rr)],
  ["ColPali + BM25", nd(test.results.C)],
  ["ColPali v1.3", nd(test.results.B)],
  ["ColPali + Tomoro", nd(p4.results.F_tm)],
  ["Tomoro ColQwen3-4B", nd(p4.results.TM)],
];

// ---- theme: same palette as the report figures (ColPali blue, text orange, visual green) -------------------
const THEME = {
  name: "ColPali talk",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1F1F1D", lt1: "FFFFFF", dk2: "16233A", lt2: "EEF4FC",
    accent1: "2A78D6", accent2: "EB6834", accent3: "1BAF7A", accent4: "7D7C78", accent5: "8A6D00", accent6: "6B6A66",
    hlink: "2A78D6", folHlink: "6B6A66",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.author = "Ismail Ucuran";
pres.title = "Searching Document Pages as Images";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
// Plain hex colours (no theme post-processing needed, so the script runs with pptxgenjs alone).
const C = { text1: HEX.dk1, text2: HEX.dk2, background1: HEX.lt1, background2: HEX.lt2,
            accent1: HEX.accent1, accent2: HEX.accent2, accent3: HEX.accent3, accent4: HEX.accent4, accent5: HEX.accent5, accent6: HEX.accent6 };

pres.defineSlideMaster({
  title: "DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 1.1, w: 8.8, h: 1.35, fontSize: 36, bold: true, color: C.background1, valign: "bottom", align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 2.55, w: 8.8, h: 0.6, fontSize: 20, color: C.background2, align: "left" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 9.0, h: 0.7, fontSize: 28, bold: true, color: C.text2, valign: "middle", align: "left" }, text: "" } },
  ],
  slideNumber: { x: 9.2, y: 5.25, w: 0.5, h: 0.25, fontSize: 10, color: C.accent6 },
});

const tag = (label, color) => ({ text: `[${label}] `, options: { fontSize: 11, color, bold: true } });
const PUB = () => tag("Published", C.accent1);
const MEA = () => tag("Measured", C.accent3);
const INT = () => tag("Interpretation", C.accent5);

function box(slide, text, x, y, w, h, opts = {}) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: 0.08, fill: { color: opts.fill || C.background1 },
    line: { color: opts.line || C.accent6, width: 1 }, objectName: opts.name || text,
  });
  slide.addText(text, { x, y, w, h, fontSize: opts.size || 13, color: C.text1, align: "center", valign: "middle", isTextBox: true, margin: 2, bold: !!opts.bold });
}
function arrow(slide, x1, y1, x2, y2) {
  slide.addShape(pres.shapes.LINE, { x: x1, y: y1, w: x2 - x1, h: y2 - y1, line: { color: C.accent6, width: 1.5, endArrowType: "triangle" }, flipV: y2 < y1 });
}

// ---- 1 title --------------------------------------------------------------------------------------------
pres.addSection({ title: "Talk" });
let s = pres.addSlide({ masterName: "DARK", sectionTitle: "Talk" });
s.addText("Searching Document Pages as Images", { placeholder: "title" });
s.addText("ColPali, and whether keyword search still helps it", { placeholder: "body" });
s.addText([
  { text: "İsmail UÇURAN  (26587009)", options: { breakLine: true, bold: true } },
  { text: "YZM5110 Multimodal Learning - Research Project  ·  19 November 2026", options: { breakLine: true } },
  { text: "github.com/ismailucrn/colpali-hybrid-retrieval", options: { color: C.accent1 } },
], { x: 0.6, y: 3.6, w: 8.8, h: 1.1, fontSize: 14, color: C.background1, isTextBox: true, margin: 0, valign: "top" });
s.addNotes("Name the topic in one sentence: searching document pages as images, and whether keyword search still helps.");

// ---- 2 problem ------------------------------------------------------------------------------------------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Talk" });
s.addText("The problem: finding the right page for RAG", { placeholder: "title" });
s.addText([
  { text: "Before an LLM can answer a question about a PDF, a " },
  { text: "retriever", options: { bold: true } },
  { text: " must find the page that holds the answer. The usual pipeline turns pages into text first:" },
], { x: 0.5, y: 1.05, w: 9.0, h: 0.6, fontSize: 15, color: C.text1, isTextBox: true, margin: 0 });
const steps = ["PDF page", "layout detection\n+ OCR", "chunks of text", "text embedding\nor BM25", "search"];
steps.forEach((t, i) => {
  const x = 0.5 + i * 1.86;
  box(s, t, x, 1.85, 1.5, 0.75, { size: 12, name: `pipeline step ${i + 1}` });
  if (i < steps.length - 1) arrow(s, x + 1.5, 2.225, x + 1.86, 2.225);
});
const issues = [
  ["Slow and fragile", "OCR, layout detection and captioning take seconds per page."],
  ["Visual content is lost", "Tables, charts, logos and layout are dropped or garbled."],
  ["OCR errors become search errors", "A misread number cannot be found again."],
];
issues.forEach(([h, d], i) => {
  const x = 0.5 + i * 3.05;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 3.0, w: 2.8, h: 1.45, rectRadius: 0.08, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: `issue ${i + 1}` });
  s.addText([{ text: h, options: { bold: true, breakLine: true, color: C.accent2 } }, { text: d }],
    { x: x + 0.15, y: 3.1, w: 2.5, h: 1.25, fontSize: 14, color: C.text1, isTextBox: true, margin: 0, valign: "top" });
});
s.addText("Two modalities meet here: the document is an image, the question is text.",
  { x: 0.5, y: 4.65, w: 9.0, h: 0.4, fontSize: 14, italic: true, color: C.accent6, isTextBox: true, margin: 0 });
s.addNotes("RAG over PDFs needs a retriever that finds the right page before any answer is generated. The usual way first turns the page into text: layout detection, OCR, chunks, then an embedding or BM25. This is slow, it loses tables, charts and layout, and every OCR mistake becomes a search mistake. Point out the two modalities: the document is an image, the question is text.");

// ---- 3 ColPali ------------------------------------------------------------------------------------------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Talk" });
s.addText("ColPali: search the page image directly", { placeholder: "title" });
box(s, "page image", 0.5, 1.25, 1.2, 0.65, { size: 12 });
box(s, "PaliGemma-3B\n(vision-language)", 2.05, 1.25, 1.75, 0.65, { size: 12, fill: C.background2, line: C.accent1 });
box(s, "≈1000 patch\nvectors", 4.15, 1.25, 1.35, 0.65, { size: 12 });
box(s, "question (text)", 0.5, 2.3, 1.2, 0.65, { size: 12 });
box(s, "same model", 2.05, 2.3, 1.75, 0.65, { size: 12, fill: C.background2, line: C.accent1 });
box(s, "one vector\nper word", 4.15, 2.3, 1.35, 0.65, { size: 12 });
box(s, "MaxSim\nscore", 5.75, 1.78, 0.9, 0.65, { size: 12, fill: C.background2, line: C.accent1, bold: true });
arrow(s, 1.7, 1.575, 2.05, 1.575); arrow(s, 3.8, 1.575, 4.15, 1.575);
arrow(s, 1.7, 2.625, 2.05, 2.625); arrow(s, 3.8, 2.625, 4.15, 2.625);
arrow(s, 5.5, 1.575, 5.75, 1.95); arrow(s, 5.5, 2.625, 5.75, 2.25);
s.addText("score(q, d) = Σ over question words of  max over page patches  ⟨qᵢ, dⱼ⟩",
  { x: 0.5, y: 3.2, w: 6.2, h: 0.4, fontSize: 14, color: C.text1, isTextBox: true, margin: 0 });
s.addText("Each question word finds the patch where it appears on the page. No OCR.",
  { x: 0.5, y: 3.6, w: 6.2, h: 0.4, fontSize: 13, italic: true, color: C.accent6, isTextBox: true, margin: 0 });
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.0, y: 1.25, w: 2.5, h: 1.7, rectRadius: 0.08, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: "ViDoRe stat" });
s.addText([
  { text: "81.3 vs 67.0", options: { fontSize: 28, bold: true, color: C.accent1, breakLine: true } },
  { text: "average nDCG@5 on ViDoRe: ColPali vs the best OCR-based pipeline", options: { fontSize: 12, color: C.text1 } },
], { x: 7.1, y: 1.35, w: 2.3, h: 1.5, isTextBox: true, margin: 0, valign: "top" });
s.addText([
  PUB(), { text: "Faysse et al., ColPali, ICLR 2025 (Table 2).", options: { breakLine: true } },
  { text: "Same scoring idea as ColBERT in text search, applied to image patches. Newer models in the family: ColQwen, Tomoro." },
], { x: 0.5, y: 4.15, w: 9.0, h: 0.8, fontSize: 14, color: C.text1, isTextBox: true, margin: 0, valign: "top" });
s.addNotes("ColPali feeds the page image to a vision-language model (PaliGemma-3B), which gives about a thousand patch vectors per page. The question gives one vector per word. The score is MaxSim: each word looks for its best-matching patch, and the maxima are summed; the same idea as ColBERT in text search. So a word on the page is matched where it is printed, without OCR. Published: on ViDoRe it beats every OCR pipeline, 81.3 vs 67.0 average nDCG@5.");

// ---- 4 question and method ------------------------------------------------------------------------------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Talk" });
s.addText("My question: does keyword search still help?", { placeholder: "title" });
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y: 1.15, w: 4.1, h: 3.05, rectRadius: 0.08, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: "question card" });
s.addText([
  { text: "In text search, combining keyword search (BM25) with semantic search is a cheap, standard improvement (\"hybrid search\").", options: { breakLine: true } },
  { text: " ", options: { breakLine: true, fontSize: 6 } },
  { text: "Visual retrievers might miss exact strings such as invoice numbers, codes and names.", options: { breakLine: true } },
  { text: " ", options: { breakLine: true, fontSize: 6 } },
  { text: "Question: ", options: { bold: true, color: C.accent2 } },
  { text: "if we add BM25 over the page's OCR text to ColPali, does page retrieval improve, and for which questions?" },
], { x: 0.7, y: 1.3, w: 3.7, h: 2.8, fontSize: 15, color: C.text1, isTextBox: true, margin: 0, valign: "top" });
const did = [
  ["Reproduce", "ColPali on ViDoRe DocVQA with the authors' code and checkpoint."],
  ["Fuse", "with BM25 by reciprocal rank fusion, plus a control: a dense text model instead of BM25."],
  ["Test fairly", "predictions fixed in advance, tuning on a dev split, test split used once, significance tests."],
  ["Follow up", "stronger text pipelines, newer Qwen models, and the nine other ViDoRe tasks."],
];
did.forEach(([h, d], i) => {
  const y = 1.15 + i * 0.95;
  s.addShape(pres.shapes.OVAL, { x: 4.95, y: y + 0.05, w: 0.5, h: 0.5, fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: `step ${i + 1}` });
  s.addText(String(i + 1), { x: 4.95, y: y + 0.05, w: 0.5, h: 0.5, fontSize: 16, bold: true, color: C.background1, align: "center", valign: "middle", isTextBox: true, margin: 0 });
  s.addText([{ text: h + ": ", options: { bold: true } }, { text: d }],
    { x: 5.6, y, w: 3.9, h: 0.85, fontSize: 14, color: C.text1, isTextBox: true, margin: 0, valign: "top" });
});
s.addNotes("In text search, adding keyword search to semantic search is a cheap and standard improvement. My question: does adding BM25 over the page's OCR text still help ColPali? The intuition was exact strings such as invoice numbers or codes. How I tested it: I reproduced ColPali with the authors' code; I fused it with BM25 and added a control with a dense text model; I wrote the predictions down in advance, tuned on a dev split and used the test split once; I followed up with stronger text pipelines, newer Qwen models and nine more ViDoRe tasks (infographics, arXiv figures, French tables, financial reports, and topic collections on energy, government, healthcare, AI and French environmental reports).");

// ---- 5 results ------------------------------------------------------------------------------------------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Talk" });
s.addText("What came out", { placeholder: "title" });
s.addChart(pres.charts.BAR, [{ name: "DocVQA test nDCG@5", labels: BARS.map((b) => b[0]), values: BARS.map((b) => +b[1].toFixed(3)) }], {
  x: 0.4, y: 1.05, w: 5.9, h: 4.0, barDir: "col",
  chartColors: [HEX.accent2, HEX.accent2, HEX.accent2, HEX.accent1, HEX.accent3, HEX.accent4],
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", dataLabelFontSize: 11, dataLabelFontFace: "+mn-lt", dataLabelColor: HEX.dk1,
  showTitle: true, title: "DocVQA test (225 queries), nDCG@5", titleFontSize: 13, titleFontFace: "+mn-lt", titleColor: HEX.dk1,
  showLegend: false, valAxisMinVal: 0, valAxisMaxVal: 0.7, valAxisLabelFormatCode: "0.0",
  valAxisLabelColor: HEX.accent6, catAxisLabelColor: HEX.dk1, catAxisLabelFontSize: 10, valAxisLabelFontSize: 10,
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
  valGridLine: { color: "E6E5E1", size: 0.5 }, catGridLine: { style: "none" }, objectName: "results chart",
});
const stats = [
  [`${N.fullB} vs ${N.pubCard}`, "Reproduction: our ColPali nDCG@5 vs the authors' number", C.accent1],
  [`${N.deltaC}`, "Adding BM25 to ColPali (DocVQA test); no gain on nine more ViDoRe tasks either", C.accent2],
  [`${N.deltaTM} / ${N.infoDelta}`, "Newer visual model (Tomoro): DocVQA test / InfoVQA, a task never used for tuning", C.accent3],
];
stats.forEach(([v, l, col], i) => {
  const y = 1.1 + i * 1.33;
  s.addText([{ text: v, options: { fontSize: 24, bold: true, color: col, breakLine: true } }, { text: l, options: { fontSize: 12, color: C.text1 } }],
    { x: 6.55, y, w: 3.0, h: 1.2, isTextBox: true, margin: 0, valign: "top" });
});
s.addText([MEA(), { text: "all numbers on this slide", options: { fontSize: 11, color: C.accent6 } }],
  { x: 6.55, y: 5.05, w: 2.6, h: 0.3, isTextBox: true, margin: 0 });
s.addNotes("Read the chart from left to right: text-only systems are well below ColPali, and adding BM25 to ColPali lowers it. Only newer visual models go above it. The reproduction matched the authors' number within one point. Adding text never helped, on DocVQA or on the nine other tasks. A newer visual model (Tomoro) did help, and the gain held on a task that was never used for tuning.");

// ---- 6 take-aways ---------------------------------------------------------------------------------------
s = pres.addSlide({ masterName: "DARK", sectionTitle: "Talk" });
s.addText("Take-aways for your own projects", { x: 0.6, y: 0.35, w: 8.8, h: 0.7, fontSize: 28, bold: true, color: C.background1, isTextBox: true, margin: 0, fontFace: THEME.headFontFace });
const take = [
  ["Visual retrievers already read the text", "OCR-based keyword search mostly finds pages ColPali finds anyway."],
  ["Fusion is not free", "Two rankers help when both are comparably strong; a much weaker one imports its errors (as in text retrieval: Bruch et al., TOIS 2024)."],
  ["Method matters as much as the score", "A control, a test set used once and a fresh task turned a negative result into a clear answer."],
];
take.forEach(([h, d], i) => {
  const x = 0.6 + i * 2.97;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.3, w: 2.75, h: 2.75, rectRadius: 0.1, fill: { color: C.background1 }, line: { color: C.background1 }, objectName: `take-away ${i + 1}` });
  s.addText([{ text: h, options: { bold: true, fontSize: 16, color: C.text2, breakLine: true } }, { text: " ", options: { fontSize: 6, breakLine: true } }, { text: d, options: { fontSize: 14, color: C.text1 } }],
    { x: x + 0.18, y: 1.45, w: 2.4, h: 2.5, isTextBox: true, margin: 0, valign: "top" });
});
s.addText([
  { text: "Report, code and all results: ", options: { color: C.background2 } },
  { text: "github.com/ismailucrn/colpali-hybrid-retrieval", options: { color: C.accent1, bold: true } },
  { text: "  (tag v1.0-submission)", options: { color: C.background2 } },
], { x: 0.6, y: 4.45, w: 8.8, h: 0.5, fontSize: 14, isTextBox: true, margin: 0 });
s.addNotes("Visual document retrievers already read the text, so OCR keyword search mostly finds pages they already find. Fusion is not free: it helps when both rankers are comparably strong, and a much weaker one imports its errors; text-retrieval research reports the same thing. Method: a control, a test set used once and a fresh task turned a negative result into a clear answer. Point to the repository for the report and the full details.");

(async () => {
  const out = path.join(__dirname, "short_talk.pptx");
  await pres.writeFile({ fileName: out });
  // Drop the ZIP's folder entries: cloud viewers (OneDrive, Google Drive) can refuse files that contain them.
  try { require("child_process").execSync(`zip -dq "${out}" '*/'`); } catch (e) { /* none present */ }
  console.log("wrote " + path.relative(ROOT, out));
})();

# 📚 StudyMate — Local AI Student Assistant

A fully **local**, no-external-API study assistant that turns a lecture PDF or scanned image into a chat-based Q&A assistant, auto-generated quizzes, and a concept mind map — with a custom-trained document layout model powering the OCR.

**Notebook:** [`StudyMate.ipynb`](./StudyMate.ipynb) · **UI:** interactive Gradio dashboard (bilingual — English & Arabic)

## Why this project

Most "chat with your PDF" tools either call an external LLM API or run plain OCR on the whole page (which breaks down on lecture slides with tables/figures mixed into the text). This project does two things differently:

1. **A custom-trained layout detection model (YOLOv8)** finds where the actual text is on a page — separating it from tables, figures, and images — *before* OCR runs, instead of OCR-ing the raw page.
2. **Everything runs locally.** No Gemini/OpenAI API calls: local OCR (Tesseract), a local summarizer (BART), a local embedding model + vector DB (SentenceTransformers + ChromaDB) for RAG, and a local LLM (Qwen2.5-1.5B-Instruct) for chat, quiz, and mind-map generation.

## Pipeline

```
PDF / Image
   │
   ├─ digital PDF text? ──────────────────► use as-is
   │
   └─ scanned page / image
        │
        ▼
   YOLOv8 Layout Detection  ──► bounding boxes: text / title / list / table / figure
        │
        ▼
   Tesseract OCR (per text region, skips tables/figures)
        │
        ▼
   Extracted text
        │
        ├─► BART (Map-Reduce) ──────────► Summary
        ├─► ChromaDB + SentenceTransformers ─► RAG index ──► Qwen2.5-1.5B ─► Chat / Q&A
        ├─► Qwen2.5-1.5B ────────────────► MCQ Quiz
        ├─► Qwen2.5-1.5B ────────────────► Mind Map (PyVis)
        └─► TextRank (PageRank over sentence embeddings) ─► Extractive summary (no LLM)
        └─► spaCy POS/NER ───────────────► Cloze (fill-in-the-blank) quiz (no LLM)
```

## Tech stack

| Component | Tool |
|---|---|
| Layout detection | YOLOv8 (Ultralytics), custom-trained |
| OCR | Tesseract (`eng+ara`) |
| Summarization | BART (`facebook/bart-large-cnn`), Map-Reduce |
| Embeddings / RAG | SentenceTransformers (`all-MiniLM-L6-v2`) + ChromaDB |
| Local LLM | Qwen2.5-1.5B-Instruct |
| Classical NLP (no-LLM path) | TextRank (PageRank) summarization, spaCy POS/NER cloze quiz generation |
| Mind maps | PyVis |
| UI | Gradio |

## The layout detection model (`yolo/`)

A YOLOv8n model trained from scratch (transfer learning from COCO weights) to detect 5 document-layout classes: `text`, `title`, `list`, `table`, `figure`.

- **Dataset:** an 800-image sample of [PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet) (scientific-paper layouts), 7,994 annotations across the 5 classes. Only a sample was used (not the full ~335k-image dataset), so results are a proof of concept rather than a state-of-the-art layout model. `yolo/annotations_sample.json` has the COCO-format sample annotations.
- **Training:** 50 epochs, image size 640, YOLOv8n backbone. Full config in [`yolo/layout_detection_v1/args.yaml`](./yolo/layout_detection_v1/args.yaml).
- **Final validation metrics:** precision 0.91, recall 0.89, mAP50 0.94, mAP50-95 0.84 — see [`results.csv`](./yolo/layout_detection_v1/results.csv), [`results.png`](./yolo/layout_detection_v1/results.png), [`confusion_matrix.png`](./yolo/layout_detection_v1/confusion_matrix.png), and the PR/F1 curves in the same folder.
- **Weights:** [`yolo/layout_detection_v1/weights/best.pt`](./yolo/layout_detection_v1/weights/best.pt).
- **Sample predictions:** [`yolo/inference_test/`](./yolo/inference_test/) — the model's layout predictions drawn on unseen pages.
- **Training/data code:** [`yolo/src/`](./yolo/src/) — `data_prep/` (COCO→YOLO conversion, PubLayNet sampling, label visualization), `training/` (training script), `inference/` (batch inference / test script).

The raw PubLayNet images aren't included in this repo (large, third-party-licensed dataset) — only the sampled annotations, the trained weights, and the training results.

## Running the notebook

The notebook was built and run in **Google Colab** (it mounts Google Drive to load `best.pt` and installs system-level Tesseract via `apt-get`). To run it:

1. Open `StudyMate.ipynb` in Colab.
2. Put `layout_detection_v1/weights/best.pt` in a Drive folder and update `DRIVE_FOLDER` in Cell 2 to point at it (the notebook falls back to full-page OCR automatically if `best.pt` isn't found).
3. Run all cells — the last cell launches the Gradio dashboard.

For a local (non-Colab) environment, install the dependencies below plus the Tesseract system binary (`tesseract-ocr`, `tesseract-ocr-ara` on Debian/Ubuntu) and skip the Drive-mount cell.

```bash
pip install -r requirements.txt
```

## Notes

- The chatbot understands both English and Arabic questions (keyword routing + Tesseract's `eng+ara` OCR), since it was originally built for Arabic-speaking students.
- `demo.launch(share=True)` creates a temporary public Gradio link when run in Colab — for a permanent deployment, this would need to move to a persistent host (e.g. Hugging Face Spaces).

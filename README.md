# Research Paper OCR & Deep Metadata Extraction

A Python starter implementation for the FrontierAtlas AI/Data Engineering Intern trial task. It retrieves metadata and PDFs for five arXiv papers, extracts PDF text, and creates structured JSON records.

## Features

- Fetches title, authors, abstract, and submission date from the arXiv API.
- Downloads PDFs and extracts text with PyMuPDF.
- Saves extracted text locally for inspection.
- Uses conservative, transparent rules for models, datasets, code links, and benchmark claims.
- Writes structured JSON and reports papers that failed processing.

> **Important:** Heuristic extraction is not guaranteed to be correct. Review every repository URL, entity, and benchmark claim against the source PDF before submitting. The assignment weights extraction accuracy most heavily. A paper with no extractable text may require OCR, such as Tesseract, which is not bundled into this starter project.

## Requirements

- Python 3.11+
- Internet connection to access arXiv
- Install dependencies:

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```bash
python extract_paper_meta.py
```

Optional output path:

```bash
python extract_paper_meta.py --output extracted_papers.json
```

Process one paper while developing:

```bash
python extract_paper_meta.py --ids 2503.18382
```

The script writes PDFs to `data/pdfs/`, extracted text to `data/extracted_text/`, and JSON to `extracted_papers.json`. Local PDFs and extracted text are ignored by Git by default.

## Output schema

Each JSON object contains:

- `arxiv_id`, `title`, `authors`, `abstract`, `publication_date`
- `arxiv_pdf_url`, `arxiv_page_url`
- `github_url`, `is_official_code`
- `tasks`, `methods`, `models`, `datasets`
- `sota_claims`, where each claim has `benchmark`, `metric`, `value`, and `comparison_target`

## Official repository detection

The script normalizes GitHub URLs and compares links found in the abstract/PDF with a paper-specific candidate list. It avoids treating arbitrary GitHub links as official code merely because they appear in a paper. However, a URL match is not sufficient proof by itself: verify the paper's code-availability statement and repository ownership manually. If the source link is not detected, the script leaves `github_url` as `null` rather than guessing.

## Limitations and next improvements

1. Regex-based entity extraction can miss tables, captions, unusual formatting, and synonyms.
2. The starter claim patterns cover known claims in the five assignment papers; expand and validate them against the full text.
3. The current script does not perform OCR on image-only pages.
4. Some task/method tags are predefined per paper and should be checked before use.
5. Add schema validation, broader unit tests, and manual ground-truth comparisons before claiming accuracy.

## Tests

Run:

```bash
pytest -v
```

## Assignment papers

- https://arxiv.org/abs/2503.18382
- https://arxiv.org/abs/2510.13998
- https://arxiv.org/abs/2607.18859
- https://arxiv.org/abs/2507.23370
- https://arxiv.org/abs/2503.11651

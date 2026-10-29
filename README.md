
# Research Paper OCR & Deep Metadata Extraction

A Python implementation for the FrontierAtlas AI/Data Engineering Intern trial task. It retrieves metadata and PDFs for five arXiv papers, extracts PDF text using PyMuPDF, and creates structured JSON records.

## Features

- Fetches paper titles, authors, abstracts, and submission dates from the arXiv API.
- Downloads research paper PDFs and extracts their text.
- Saves extracted text locally for inspection.
- Identifies models, datasets, GitHub repositories, and benchmark claims using transparent extraction rules.
- Produces structured JSON and reports papers that fail during processing.

> **Note:** Heuristic extraction is not guaranteed to be correct. Repository URLs, models, datasets, and benchmark claims should be reviewed against the original papers. Scanned or image-only pages may require a separate OCR engine, which is not included.

## Requirements

- Python 3.11 or later
- Internet connection
- Dependencies listed in `requirements.txt`

## Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Running the Extractor

Process all five configured papers:

```bash
python extract_paper_meta.py
```

Specify an output path:

```bash
python extract_paper_meta.py --output extracted_papers.json
```

Process one paper during development:

```bash
python extract_paper_meta.py --ids 2503.18382
```

The script saves downloaded PDFs in `data/pdfs/`, extracted text in `data/extracted_text/`, and metadata in `extracted_papers.json`.

## Output Schema

Each paper record contains:

- `arxiv_id`: arXiv identifier
- `title`: Paper title
- `authors`: List of author names
- `publication_date`: Date in `YYYY-MM-DD` format
- `arxiv_pdf_url`: PDF URL
- `arxiv_page_url`: arXiv abstract page URL
- `github_url`: Official code repository URL or `null`
- `is_official_code`: Whether the repository is identified as official
- `tasks`: Research tasks
- `methods`: Methods used or proposed
- `models`: Models identified
- `datasets`: Datasets identified
- `sota_claims`: Quantitative benchmark claims
- `abstract`: Paper abstract

Each item in `sota_claims` contains `benchmark`, `metric`, `value`, and `comparison_target`.

## Official Repository Detection

The extractor normalizes GitHub URLs and compares URLs found in the paper's abstract and extracted PDF text against a paper-specific list of candidate repositories. This helps avoid treating arbitrary dependency links as the paper's official implementation.

If no candidate URL is detected, the script returns `null` for `github_url` and `false` for `is_official_code`.

This approach is heuristic. A URL match alone does not prove that a repository is officially associated with a paper. Repository ownership and the paper's code-availability statement should be checked manually.

## Testing

Run the unit tests:

```bash
python -m pytest -v
```

GitHub Actions runs the automated tests and extraction workflow. A successful run confirms that the configured checks passed, but does not guarantee scientific accuracy.

## Limitations

1. Regex-based extraction may miss entities in tables, captions, or unusual PDF layouts.
2. Benchmark claims may require manual verification against experimental results.
3. Image-only pages are not processed by a dedicated OCR engine.
4. Some task and method tags are predefined for the five target papers.
5. Passing unit tests does not guarantee complete or scientifically accurate metadata.

## Assignment Papers

1. [PP-FormulaNet — arXiv:2503.18382](https://arxiv.org/abs/2503.18382)
2. [BitNet Distillation — arXiv:2510.13998](https://arxiv.org/abs/2510.13998)
3. [PhoenixRepair — arXiv:2607.18859](https://arxiv.org/abs/2607.18859)
4. [Trae Agent — arXiv:2507.23370](https://arxiv.org/abs/2507.23370)
5. [VGGT — arXiv:2503.11651](https://arxiv.org/abs/2503.11651)

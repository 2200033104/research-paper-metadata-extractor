#!/usr/bin/env python3
"""Extract structured metadata from a small set of arXiv research papers.

Usage:
    python extract_paper_meta.py
    python extract_paper_meta.py --output extracted_papers.json

The script downloads arXiv metadata and PDFs, extracts text with PyMuPDF,
then applies transparent heuristic extraction. Scientific claims and entity
lists should be reviewed against the source papers before production use.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import fitz
import requests
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
PDF_DIR = ROOT / "data" / "pdfs"
TEXT_DIR = ROOT / "data" / "extracted_text"
ARXIV_API = "https://export.arxiv.org/api/query"
USER_AGENT = "ResearchPaperMetadataExtractor/1.0 (educational internship project)"
ARXIV_IDS = ["2503.18382", "2510.13998", "2607.18859", "2507.23370", "2503.11651"]

# Explicit code links known from the papers' own abstract/code statements.
# These are hints for matching, not a replacement for checking source context.
OFFICIAL_REPOSITORIES = {
    "2503.18382": [
        "https://github.com/PaddlePaddle/PaddleOCR",
        "https://github.com/PaddlePaddle/PaddleX",
    ],
    "2510.13998": ["https://github.com/microsoft/BitNet"],
    "2607.18859": ["https://github.com/DeepSoftwareAnalytics/PhoenixRepair"],
    "2507.23370": ["https://github.com/bytedance/trae-agent"],
    "2503.11651": ["https://github.com/facebookresearch/vggt"],
}

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})


def clean_text(value: str) -> str:
    """Normalize common PDF line-break, whitespace, and simple LaTeX artifacts."""
    if not value:
        return ""
    value = value.replace("\u00ad", "")
    value = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", value)  # split word hyphenation
    value = re.sub(r"\\textbf\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\\textit\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\\emph\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\$([^$]*)\$", r"\1", value)
    value = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\s*\n\s*", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def normalize_github_url(url: str) -> str | None:
    """Return a clean GitHub repository URL, excluding non-repository paths."""
    match = re.search(r"https?://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)", url)
    if not match:
        return None
    owner, repo = match.groups()
    repo = repo.removesuffix(".git")
    if owner.lower() in {"features", "topics", "collections", "orgs", "settings"}:
        return None
    if repo.lower() in {"issues", "pulls", "actions", "discussions", "tree", "blob"}:
        return None
    return f"https://github.com/{owner}/{repo}"


def fetch_arxiv_metadata(arxiv_id: str) -> dict[str, Any]:
    """Fetch title, authors, abstract, and submitted date from the arXiv API."""
    response = SESSION.get(ARXIV_API, params={"id_list": arxiv_id}, timeout=30)
    response.raise_for_status()
    root = ET.fromstring(response.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entry = root.find("atom:entry", ns)
    if entry is None or entry.findtext("atom:id", default="", namespaces=ns).endswith("/api/errors"):
        raise ValueError(f"No arXiv API entry returned for {arxiv_id}")
    title = clean_text(entry.findtext("atom:title", default="", namespaces=ns))
    abstract = clean_text(entry.findtext("atom:summary", default="", namespaces=ns))
    authors = []
    for author in entry.findall("atom:author", ns):
        name = clean_text(author.findtext("atom:name", default="", namespaces=ns))
        if name:
            authors.append(name)
    published = entry.findtext("atom:published", default="", namespaces=ns)
    pub_date = published[:10] if re.match(r"\d{4}-\d{2}-\d{2}", published) else None
    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "authors": authors,
        "publication_date": pub_date,
        "arxiv_pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        "arxiv_page_url": f"https://arxiv.org/abs/{arxiv_id}",
        "abstract": abstract,
    }


def download_pdf(arxiv_id: str) -> Path:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    path = PDF_DIR / f"{arxiv_id}.pdf"
    if path.exists() and path.stat().st_size > 1024:
        return path
    response = SESSION.get(f"https://arxiv.org/pdf/{arxiv_id}", timeout=60)
    response.raise_for_status()
    if not response.content.startswith(b"%PDF"):
        raise ValueError(f"Downloaded file for {arxiv_id} is not a PDF")
    path.write_bytes(response.content)
    return path


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract page text; scanned PDFs may need a separate OCR engine."""
    chunks: list[str] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            chunks.append(page.get_text("text", sort=True))
    return "\n".join(chunks).strip()


def get_github_urls(text: str) -> list[str]:
    """Find GitHub URLs only in explicit code/repository context when possible."""
    urls = re.findall(r"https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?(?:/[^\s)\]}>,]*)?", text, re.I)
    normalized = []
    for url in urls:
        clean = normalize_github_url(url.rstrip(".,;:"))
        if clean and clean not in normalized:
            normalized.append(clean)
    return normalized


def find_entities(text: str, arxiv_id: str) -> tuple[list[str], list[str], list[str], list[dict[str, str]]]:
    """Use conservative keyword rules; inspect the paper to improve these lists."""
    lower = text.lower()
    model_terms = {
        "2503.18382": ["PP-FormulaNet-L", "PP-FormulaNet-S", "UniMERNet"],
        "2510.13998": ["BitDistill", "BitNet", "Qwen", "MiniLM"],
        "2607.18859": ["PhoenixRepair", "SWE-agent", "DeepSeek-V3.1", "MiniMax-M2.5"],
        "2507.23370": ["Trae Agent"],
        "2503.11651": ["VGGT"],
    }
    dataset_terms = {
        "2503.18382": ["UniMERNet Benchmark", "Formula Mining System Dataset"],
        "2510.13998": [],
        "2607.18859": ["SWE-bench Verified"],
        "2507.23370": ["SWE-bench", "SWE-bench Verified"],
        "2503.11651": [],
    }
    tasks_terms = {
        "2503.18382": ["document-understanding", "computer-vision", "formula-recognition"],
        "2510.13998": ["language-model-compression", "model-distillation"],
        "2607.18859": ["software-repair", "code-agents"],
        "2507.23370": ["software-engineering", "repository-level-issue-resolution"],
        "2503.11651": ["computer-vision", "3d-reconstruction"],
    }
    methods_terms = {
        "2503.18382": ["deep-learning", "ocr", "formula-recognition"],
        "2510.13998": ["knowledge-distillation", "low-bit-quantization", "continual-pretraining"],
        "2607.18859": ["multi-agent", "iterative-refinement"],
        "2507.23370": ["ensemble-reasoning", "test-time-scaling"],
        "2503.11651": ["transformer", "feed-forward-network", "3d-vision"],
    }
    models = [term for term in model_terms.get(arxiv_id, []) if term.lower() in lower]
    datasets = [term for term in dataset_terms.get(arxiv_id, []) if term.lower() in lower]
    tasks = tasks_terms.get(arxiv_id, [])
    methods = methods_terms.get(arxiv_id, [])

    # Keep claims only when a specific, paper-associated statement is matched.
    claims: list[dict[str, str]] = []
    patterns = {
        "2503.18382": [
            (r"(?:surpass|outperform\w*).{0,100}?UniMERNet.{0,50}?(?:by|of)\s*(?:a\s*)?(\+?\d+(?:\.\d+)?%)", "UniMERNet Benchmark", "Accuracy", "UniMERNet"),
            (r"(\d+(?:\.\d+)?x)\s*faster", "Inference Speed", "Speedup Factor", "UniMERNet"),
        ],
        "2510.13998": [
            (r"(\d+(?:\.\d+)?x)\s*memory savings", "Memory Usage", "Memory Savings", "Full-precision model"),
            (r"(\d+(?:\.\d+)?x)\s*faster inference", "CPU Inference", "Speedup Factor", "Full-precision model"),
        ],
        "2607.18859": [
            (r"relative improvement of\s*(\d+(?:\.\d+)?%)\s*over\s*SWE-agent", "SWE-bench-Verified", "Relative Improvement", "SWE-agent"),
            (r"resolved rate of\s*(\d+(?:\.\d+)?%)\s*Pass@1", "SWE-bench-Verified", "Pass@1 Resolved Rate", "Prior methods"),
        ],
        "2507.23370": [
            (r"average improvement of\s*(\d+(?:\.\d+)?%)", "SWE-bench", "Average Pass@1 Improvement", "Four SOTA baselines"),
            (r"Pass@1 score of\s*(\d+(?:\.\d+)?%)", "SWE-bench Verified", "Pass@1", "Leaderboard baseline"),
        ],
        "2503.11651": [
            (r"reconstruct\w* images in under\s*(one second)", "3D Reconstruction", "Reconstruction Time", "Methods requiring post-processing"),
        ],
    }
    for pattern, benchmark, metric, comparison in patterns.get(arxiv_id, []):
        match = re.search(pattern, text, re.I | re.S)
        if match:
            claims.append({
                "benchmark": benchmark,
                "metric": metric,
                "value": match.group(1),
                "comparison_target": comparison,
            })
    return models, datasets, tasks, methods, claims


def build_record(arxiv_id: str) -> dict[str, Any]:
    record = fetch_arxiv_metadata(arxiv_id)
    pdf_path = download_pdf(arxiv_id)
    text = extract_pdf_text(pdf_path)
    if len(text) < 300:
        raise ValueError(f"Very little text extracted from {arxiv_id}; OCR/manual review may be required.")
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    (TEXT_DIR / f"{arxiv_id}.txt").write_text(text, encoding="utf-8")

    # Prefer repository URLs explicitly listed in this assignment's paper-specific
    # map, but only if the link also appears in the PDF or abstract. This avoids
    # accidentally treating a generic dependency URL as the paper's official code.
    corpus = f"{record.get('abstract', '')}\n{text}"
    mentioned_urls = get_github_urls(corpus)
    expected = OFFICIAL_REPOSITORIES.get(arxiv_id, [])
    verified = [url for url in expected if url in mentioned_urls]
    if not verified:
        # The explicit map is a candidate list, not proof. Leave unknown if PDF
        # text extraction did not surface a matching code statement.
        github_url = None
        is_official = False
    else:
        github_url = verified[0]
        is_official = True

    models, datasets, tasks, methods, claims = find_entities(corpus, arxiv_id)
    record.update({
        "github_url": github_url,
        "is_official_code": is_official,
        "tasks": tasks,
        "methods": methods,
        "models": models,
        "datasets": datasets,
        "sota_claims": claims,
    })
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "extracted_papers.json"))
    parser.add_argument("--ids", nargs="*", default=ARXIV_IDS, help="Optional arXiv IDs to process")
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds between arXiv requests")
    args = parser.parse_args()

    results = []
    failures = []
    for index, arxiv_id in enumerate(args.ids):
        try:
            print(f"[{index + 1}/{len(args.ids)}] Processing {arxiv_id}...")
            results.append(build_record(arxiv_id))
        except (requests.RequestException, ValueError, ET.ParseError, fitz.FileDataError) as exc:
            failures.append({"arxiv_id": arxiv_id, "error": str(exc)})
            print(f"WARNING: {arxiv_id}: {exc}", file=sys.stderr)
        if index < len(args.ids) - 1:
            time.sleep(max(0.0, args.delay))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(results)} paper records to {output}")
    if failures:
        print("Some papers failed. Review these errors and rerun:", file=sys.stderr)
        print(json.dumps(failures, indent=2), file=sys.stderr)
        return 1
    if len(results) != len(args.ids):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

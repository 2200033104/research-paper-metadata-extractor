
#!/usr/bin/env python3
"""Extract metadata from configured arXiv papers using API data and PDF text.

Scientific entities and benchmark claims are found with heuristics and must be
checked against the source papers before being treated as authoritative.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pymupdf
import requests

ROOT = Path(__file__).resolve().parent
PDF_DIR = ROOT / "data" / "pdfs"
TEXT_DIR = ROOT / "data" / "extracted_text"
ARXIV_API = "https://export.arxiv.org/api/query"

ARXIV_IDS = [
    "2503.18382",
    "2510.13998",
    "2607.18859",
    "2507.23370",
    "2503.11651",
]

OFFICIAL_REPOSITORIES = {
    "2503.18382": [
        "https://github.com/PaddlePaddle/PaddleOCR",
        "https://github.com/PaddlePaddle/PaddleX",
    ],
    "2510.13998": ["https://github.com/microsoft/BitNet"],
    "2607.18859": [
        "https://github.com/DeepSoftwareAnalytics/PhoenixRepair"
    ],
    "2507.23370": ["https://github.com/bytedance/trae-agent"],
    "2503.11651": ["https://github.com/facebookresearch/vggt"],
}

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "ResearchPaperMetadataExtractor/1.1 (educational project)"
})


def clean_text(value: str) -> str:
    """Normalize whitespace and common PDF line-break artifacts."""
    if not value:
        return ""

    value = value.replace("\u00ad", "")
    value = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", value)

    for command in ("textbf", "textit", "emph"):
        value = re.sub(
            rf"\\{command}\{{([^{{}}]*)\}}",
            r"\1",
            value,
        )

    value = re.sub(r"\$([^$]*)\$", r"\1", value)
    return re.sub(r"\s+", " ", value).strip()


def normalize_github_url(url: str) -> str | None:
    """Normalize a GitHub URL to owner/repository format."""
    match = re.search(
        r"https?://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)",
        url,
        re.IGNORECASE,
    )

    if not match:
        return None

    owner, repo = match.groups()
    repo = repo.removesuffix(".git")

    if owner.lower() in {
        "features", "topics", "collections", "orgs", "settings"
    }:
        return None

    if repo.lower() in {
        "issues", "pulls", "actions", "discussions",
        "tree", "blob", "commit", "releases", "wiki",
    }:
        return None

    return f"https://github.com/{owner}/{repo}"


def is_valid_github_url(url: str) -> bool:
    """Check whether a URL has a valid-looking repository format."""
    return normalize_github_url(url) is not None


def fetch_arxiv_metadata(arxiv_id: str) -> dict[str, Any]:
    """Fetch title, authors, abstract, and submission date from arXiv."""
    response = SESSION.get(
        ARXIV_API,
        params={"id_list": arxiv_id},
        timeout=30,
    )
    response.raise_for_status()

    root = ET.fromstring(response.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entry = root.find("atom:entry", ns)

    if entry is None:
        raise ValueError(f"No arXiv API entry returned for {arxiv_id}")

    title = clean_text(
        entry.findtext("atom:title", default="", namespaces=ns)
    )
    abstract = clean_text(
        entry.findtext("atom:summary", default="", namespaces=ns)
    )

    authors = [
        clean_text(author.findtext(
            "atom:name", default="", namespaces=ns
        ))
        for author in entry.findall("atom:author", ns)
    ]
    authors = [author for author in authors if author]

    published = entry.findtext(
        "atom:published", default="", namespaces=ns
    )
    pub_date = (
        published[:10]
        if re.match(r"^\d{4}-\d{2}-\d{2}", published)
        else None
    )

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
    """Download a PDF, reusing a previously downloaded copy."""
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    path = PDF_DIR / f"{arxiv_id}.pdf"

    if path.exists() and path.stat().st_size > 1024:
        return path

    response = SESSION.get(
        f"https://arxiv.org/pdf/{arxiv_id}",
        timeout=60,
    )
    response.raise_for_status()

    if not response.content.startswith(b"%PDF"):
        raise ValueError(f"Downloaded file for {arxiv_id} is not a PDF")

    path.write_bytes(response.content)
    return path


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract selectable PDF text; image-only pages need separate OCR."""
    with pymupdf.open(pdf_path) as document:
        pages = [
            page.get_text("text", sort=True)
            for page in document
        ]

    return "\n".join(pages).strip()


def get_github_urls(text: str) -> list[str]:
    """Find and normalize GitHub repository URLs."""
    raw_urls = re.findall(
        r"https?://github\.com/[A-Za-z0-9_.-]+/"
        r"[A-Za-z0-9_.-]+(?:\.git)?(?:/[^\s)\]}>,]*)?",
        text,
        re.IGNORECASE,
    )

    urls = []

    for raw_url in raw_urls:
        normalized = normalize_github_url(raw_url.rstrip(".,;:"))

        if normalized and normalized not in urls:
            urls.append(normalized)

    return urls


def has_repository_context(text: str, url: str) -> bool:
    """Check whether nearby text suggests a code or repository link."""
    normalized = re.sub(r"\s+", " ", text).lower()
    position = normalized.find(url.lower())

    if position < 0:
        return False

    context = normalized[
        max(0, position - 250):
        position + len(url) + 250
    ]

    cues = (
        "code",
        "source code",
        "implementation",
        "github",
        "repository",
        "available at",
        "open-source",
        "open source",
    )

    return any(cue in context for cue in cues)


def find_entities(
    text: str,
    arxiv_id: str,
) -> tuple[
    list[str],
    list[str],
    list[str],
    list[str],
    list[dict[str, str]],
]:
    """Extract configured model, dataset, task, method, and claim candidates."""
    lower = text.lower()

    models_by_id = {
        "2503.18382": [
            "PP-FormulaNet-L", "PP-FormulaNet-S", "UniMERNet"
        ],
        "2510.13998": [
            "BitDistill", "BitNet", "Qwen", "MiniLM", "SubLN"
        ],
        "2607.18859": [
            "PhoenixRepair", "SWE-agent",
            "DeepSeek-V3.1", "MiniMax-M2.5"
        ],
        "2507.23370": ["Trae Agent"],
        "2503.11651": ["VGGT"],
    }

    datasets_by_id = {
        "2503.18382": [
            "UniMERNet Benchmark",
            "Formula Mining System Dataset",
        ],
        "2510.13998": [
            "WikiText", "C4", "FineWeb", "GSM8K", "MMLU"
        ],
        "2607.18859": ["SWE-bench Verified", "SWE-bench"],
        "2507.23370": ["SWE-bench Verified", "SWE-bench"],
        "2503.11651": [
            "ETH3D", "DTU", "Tanks and Temples", "CO3D",
            "ScanNet", "MegaDepth", "RealEstate10K",
        ],
    }

    tasks_by_id = {
        "2503.18382": [
            "document-understanding",
            "computer-vision",
            "formula-recognition",
        ],
        "2510.13998": [
            "language-model-compression",
            "model-distillation",
        ],
        "2607.18859": ["software-repair", "code-agents"],
        "2507.23370": [
            "software-engineering",
            "repository-level-issue-resolution",
        ],
        "2503.11651": [
            "computer-vision",
            "3d-reconstruction",
            "depth-estimation",
            "camera-pose-estimation",
        ],
    }

    methods_by_id = {
        "2503.18382": [
            "deep-learning", "ocr", "formula-recognition"
        ],
        "2510.13998": [
            "knowledge-distillation",
            "low-bit-quantization",
            "continual-pretraining",
        ],
        "2607.18859": ["multi-agent", "iterative-refinement"],
        "2507.23370": ["ensemble-reasoning", "test-time-scaling"],
        "2503.11651": [
            "transformer", "feed-forward-network", "3d-vision"
        ],
    }

    models = [
        item for item in models_by_id.get(arxiv_id, [])
        if item.lower() in lower
    ]
    datasets = [
        item for item in datasets_by_id.get(arxiv_id, [])
        if item.lower() in lower
    ]
    tasks = tasks_by_id.get(arxiv_id, [])
    methods = methods_by_id.get(arxiv_id, [])

    patterns = {
        "2503.18382": [
            (
                r"(?:surpass|outperform\w*).{0,120}?UniMERNet"
                r".{0,80}?(?:by|of)\s*(?:a\s*)?(\+?\d+(?:\.\d+)?%)",
                "UniMERNet Benchmark",
                "Accuracy improvement",
                "UniMERNet",
            ),
            (
                r"(\d+(?:\.\d+)?x)\s*faster(?:\s+inference)?",
                "Inference Speed",
                "Speedup factor",
                "UniMERNet",
            ),
        ],
        "2510.13998": [
            (
                r"(?:up to\s*)?(\d+(?:\.\d+)?x)\s*"
                r"(?:less memory|memory savings)",
                "Memory Usage",
                "Memory savings",
                "Full-precision counterpart models",
            ),
            (
                r"(\d+(?:\.\d+)?x)\s*faster\s+"
                r"(?:CPU\s+)?inference",
                "CPU Inference",
                "Inference speedup",
                "Full-precision counterpart models",
            ),
        ],
        "2607.18859": [
            (
                r"relative improvement of\s*(\d+(?:\.\d+)?%)"
                r"\s*over\s*SWE-agent",
                "SWE-bench-Verified",
                "Relative improvement",
                "SWE-agent",
            ),
            (
                r"(\d+(?:\.\d+)?%)\s*(?:Pass@1\s*)?"
                r"(?:resolved rate|Pass@1 resolved rate)",
                "SWE-bench-Verified",
                "Pass@1 resolved rate",
                "Prior methods",
            ),
        ],
        "2507.23370": [
            (
                r"average improvement of\s*(\d+(?:\.\d+)?%)",
                "SWE-bench",
                "Average Pass@1 improvement",
                "Four SOTA baselines",
            ),
            (
                r"Pass@1 score of\s*(\d+(?:\.\d+)?%)",
                "SWE-bench Verified",
                "Pass@1",
                "Leaderboard baseline",
            ),
        ],
        "2503.11651": [
            (
                r"(?:inference|reconstruct\w*|reconstruction)"
                r".{0,80}?(under one second|less than one second)",
                "3D Reconstruction",
                "Reconstruction time",
                "Methods requiring post-processing",
            ),
        ],
    }

    claims = []

    for pattern, benchmark, metric, comparison in patterns.get(arxiv_id, []):
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)

        if match:
            claim = {
                "benchmark": benchmark,
                "metric": metric,
                "value": match.group(1),
                "comparison_target": comparison,
            }

            if claim not in claims:
                claims.append(claim)

    return models, datasets, tasks, methods, claims


def build_record(arxiv_id: str) -> dict[str, Any]:
    """Build a paper record and save extracted PDF text for inspection."""
    record = fetch_arxiv_metadata(arxiv_id)
    pdf_path = download_pdf(arxiv_id)
    pdf_text = extract_pdf_text(pdf_path)

    if len(pdf_text) < 300:
        raise ValueError(
            f"Very little text extracted from {arxiv_id}; "
            "OCR or manual review may be required."
        )

    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    (TEXT_DIR / f"{arxiv_id}.txt").write_text(
        pdf_text,
        encoding="utf-8",
    )

    corpus = f"{record['abstract']}\n{pdf_text}"
    found_urls = get_github_urls(corpus)
    candidates = OFFICIAL_REPOSITORIES.get(arxiv_id, [])

    verified = [
        url for url in candidates
        if url in found_urls and has_repository_context(corpus, url)
    ]

    github_url = verified[0] if verified else None
    is_official = bool(verified)

    models, datasets, tasks, methods, claims = find_entities(
        corpus, arxiv_id
    )

    if not datasets:
        print(
            f"WARNING: {arxiv_id}: no configured dataset terms detected; "
            "review the paper.",
            file=sys.stderr,
        )

    if not claims:
        print(
            f"WARNING: {arxiv_id}: no configured benchmark claims detected; "
            "review the paper.",
            file=sys.stderr,
        )

    if not github_url:
        print(
            f"WARNING: {arxiv_id}: no candidate repository verified "
            "in code context.",
            file=sys.stderr,
        )

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
    parser.add_argument(
        "--output",
        default=str(ROOT / "extracted_papers.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--ids",
        nargs="*",
        default=ARXIV_IDS,
        help="arXiv IDs to process",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds to wait between papers",
    )

    args = parser.parse_args()

    if not args.ids:
        parser.error("provide at least one arXiv ID")

    invalid_ids = [
        item for item in args.ids
        if not re.fullmatch(r"\d{4}\.\d{4,5}", item)
    ]

    if invalid_ids:
        parser.error(f"invalid arXiv ID format: {', '.join(invalid_ids)}")

    records = []
    failures = []

    for index, arxiv_id in enumerate(args.ids):
        try:
            print(f"[{index + 1}/{len(args.ids)}] Processing {arxiv_id}...")
            records.append(build_record(arxiv_id))
        except (
            requests.RequestException,
            ValueError,
            ET.ParseError,
            pymupdf.FileDataError,
        ) as exc:
            failures.append({
                "arxiv_id": arxiv_id,
                "error": str(exc),
            })
            print(f"WARNING: {arxiv_id}: {exc}", file=sys.stderr)

        if index < len(args.ids) - 1:
            time.sleep(max(0.0, args.delay))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {len(records)} paper records to {output}")

    if failures:
        print(
            "Some papers failed. Review the errors and rerun:",
            file=sys.stderr,
        )
        print(json.dumps(failures, indent=2), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

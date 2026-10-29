
import json
import re
from datetime import date
from pathlib import Path

import pytest

from extract_paper_meta import (
    clean_text,
    normalize_github_url,
    is_valid_github_url,
)


def test_clean_text_removes_extra_whitespace():
    assert clean_text("  Hello   world \n test  ") == "Hello world test"


def test_clean_text_handles_empty_values():
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_normalize_github_url():
    result = normalize_github_url(
        "https://github.com/example/project/"
    )
    assert result == "https://github.com/example/project"


def test_reject_invalid_github_url():
    assert not is_valid_github_url(
        "https://example.com/not-github"
    )


def test_publication_date_format():
    publication_date = "2025-03-18"

    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2}", publication_date
    )

    date.fromisoformat(publication_date)


def test_json_round_trip(tmp_path):
    sample = {
        "arxiv_id": "2503.18382",
        "title": "Example Paper",
        "authors": ["Author One"],
        "publication_date": "2025-03-18",
        "sota_claims": [],
    }

    output_file = tmp_path / "test.json"
    output_file.write_text(
        json.dumps(sample, ensure_ascii=False),
        encoding="utf-8",
    )

    loaded = json.loads(
        output_file.read_text(encoding="utf-8")
    )

    assert loaded == sample

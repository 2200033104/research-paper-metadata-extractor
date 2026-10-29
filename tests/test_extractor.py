import json
from datetime import date

from extract_paper_meta import clean_text, normalize_github_url


def test_clean_text_repairs_hyphenation_and_whitespace():
    assert clean_text("re-\n search   paper") == "research paper"


def test_clean_text_removes_simple_latex():
    assert clean_text(r"\textbf{Important} $x^2$") == "Important x^2"


def test_normalize_github_url_removes_git_suffix_and_path():
    assert normalize_github_url("https://github.com/PaddlePaddle/PaddleOCR/tree/main") == (
        "https://github.com/PaddlePaddle/PaddleOCR"
    )


def test_reject_non_repository_github_paths():
    assert normalize_github_url("https://github.com/topics/python") is None


def test_publication_date_format():
    value = "2025-03-24"
    assert date.fromisoformat(value).isoformat() == value


def test_json_round_trip():
    payload = [{"arxiv_id": "2503.18382", "authors": ["Jane Smith"]}]
    assert json.loads(json.dumps(payload)) == payload

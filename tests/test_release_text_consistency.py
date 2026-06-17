from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def issue_workflow_trigger_types() -> list[str]:
    text = read(".github/workflows/assetpack-issue-generate.yml")
    match = re.search(r"types:\s*\[(.*?)\]", text)
    assert match, "issue generation workflow must declare issue event types"
    return [item.strip() for item in match.group(1).split(",")]


def test_issue_generation_workflow_uses_label_based_trigger() -> None:
    assert issue_workflow_trigger_types() == ["labeled", "edited", "reopened"]


def test_issue_template_applies_only_request_label() -> None:
    data = yaml.safe_load(read(".github/ISSUE_TEMPLATE/generate.yml"))
    assert data["labels"] == ["asset-request"]


def test_readme_release_summary_matches_validation_record() -> None:
    readme = read("README.md")
    assert "fresh smoke Issue" in readme
    assert "duplicate replay check" in readme
    assert "invalid Issue check" in readme
    assert "manual smoke Issue" in readme


def test_docs_index_links_final_text_audit() -> None:
    docs_index = read("docs/README.md")
    assert "[Final text audit](final-text-audit.md)" in docs_index


def test_storage_policy_describes_duplicates_as_existing_assets() -> None:
    storage = read("docs/storage-policy.md")
    assert "Duplicate recipe ids are handled as existing generated records" in storage
    assert "duplicate recipe IDs before image generation" not in storage


def test_user_facing_docs_do_not_use_stale_label_instructions() -> None:
    docs = [
        "README.md",
        "docs/user-manual.md",
        "docs/template-build-and-use.md",
        "docs/example-issues.md",
    ]
    for path in docs:
        text = read(path)
        assert "needs-validation" not in text, path
        assert "Apply the configured request label." not in text, path
        assert "Apply the `asset-request` label." not in text, path


def test_issue_workflow_docs_match_current_trigger_contract() -> None:
    docs = [
        "README.md",
        "docs/user-manual.md",
        "docs/issue-driven-generation.md",
        "docs/implementation-status.md",
    ]
    for path in docs:
        text = read(path)
        assert "labeled" in text, path
    issue_doc = read("docs/issue-driven-generation.md")
    assert "does not listen to `opened` events" in issue_doc

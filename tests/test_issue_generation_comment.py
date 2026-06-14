import json
import subprocess
import sys
from pathlib import Path


def test_success_comment_uses_committed_asset_language(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "request.json").write_text(json.dumps({
        "recipe_id": "assetpack-test",
        "selected_model_id": "sdxl-turbo-quality",
    }), encoding="utf-8")
    (out_dir / "report.json").write_text(json.dumps({
        "summary": {"passed": 1, "failed": 0}
    }), encoding="utf-8")

    subprocess.run([
        sys.executable,
        "scripts/write_issue_generation_comment.py",
        "--out-dir", str(out_dir),
        "--outcome", "success",
    ], check=True)

    body = (out_dir / "generation-comment.md").read_text(encoding="utf-8")
    assert "generated and committed" in body
    assert "not committed to Git by default" not in body
    assert "assetpack-test" in body
    assert "sdxl-turbo-quality" in body


def test_failure_comment_reports_reason(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "request.json").write_text(json.dumps({
        "recipe_id": "assetpack-test",
        "selected_model_id": "sdxl-turbo-quality",
    }), encoding="utf-8")
    (out_dir / "report.json").write_text(json.dumps({
        "reason": "generation subprocess failed",
        "summary": {"passed": 0, "failed": 1}
    }), encoding="utf-8")

    subprocess.run([
        sys.executable,
        "scripts/write_issue_generation_comment.py",
        "--out-dir", str(out_dir),
        "--outcome", "failure",
    ], check=True)

    body = (out_dir / "generation-comment.md").read_text(encoding="utf-8")
    assert "generation incomplete" in body
    assert "generation subprocess failed" in body

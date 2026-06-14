import json
import subprocess
import sys
from pathlib import Path


def write_config(path: Path, output_root: Path) -> None:
    path.write_text(
        f"""
issue_generation:
  committed_output_root: {output_root.as_posix()}
input_policy:
  ascii_only: true
  ascii_fields:
    - subject
    - scene
    - audience
    - constraints
prompt_policy:
  required_terms:
    - black outline
    - white background
""",
        encoding="utf-8",
    )


def run_policy_with_comment(tmp_path: Path, request: dict) -> tuple[dict, str]:
    cfg = tmp_path / "assetpack.yml"
    request_json = tmp_path / "request.json"
    comment_file = tmp_path / "comment.md"
    output_root = tmp_path / "assets" / "generated"
    write_config(cfg, output_root)
    request_json.write_text(json.dumps(request), encoding="utf-8")
    subprocess.run([
        sys.executable,
        "scripts/validate_issue_policy.py",
        "--config", str(cfg),
        "--request-json", str(request_json),
        "--comment-file", str(comment_file),
    ], check=True)
    comment = comment_file.read_text(encoding="utf-8") if comment_file.exists() else ""
    return json.loads(request_json.read_text(encoding="utf-8")), comment


def run_policy(tmp_path: Path, request: dict) -> dict:
    result, _ = run_policy_with_comment(tmp_path, request)
    return result


def valid_request() -> dict:
    return {
        "valid": True,
        "recipe_id": "assetpack-test",
        "prompt": "cat, black outline, white background",
        "fields": {
            "subject": "cat",
            "scene": "sitting",
            "audience": "children",
            "constraints": "simple",
        },
    }


def test_rejects_non_ascii_field(tmp_path):
    request = valid_request()
    request["fields"]["subject"] = "caf" + chr(233)
    result = run_policy(tmp_path, request)
    assert result["valid"] is False
    assert any("non-ASCII" in error for error in result["errors"])


def test_rejects_missing_required_term(tmp_path):
    request = valid_request()
    request["prompt"] = "cat, black outline"
    result = run_policy(tmp_path, request)
    assert result["valid"] is False
    assert "white background" in result["missing_terms"]


def test_duplicate_recipe_id_is_existing_asset_status(tmp_path):
    duplicate_dir = tmp_path / "assets" / "generated" / "issue-000001" / "assetpack-test"
    duplicate_dir.mkdir(parents=True)
    result, comment = run_policy_with_comment(tmp_path, valid_request())
    assert result["valid"] is False
    assert result["policy_status"] == "duplicate"
    assert result["duplicate_path"] is not None
    assert "already exists" in comment
    assert "assetpack-test" in comment

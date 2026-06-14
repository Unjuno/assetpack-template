import json
import subprocess
import sys
from pathlib import Path


def test_generated_assets_index_lists_records(tmp_path):
    root = tmp_path / "assets" / "generated"
    record = root / "issue-000034" / "assetpack-test"
    record.mkdir(parents=True)
    (record / "image.png").write_bytes(b"fake-png")
    (record / "prompt.txt").write_text("cat prompt\n", encoding="utf-8")
    (record / "metadata.json").write_text(json.dumps({
        "issue_number": 34,
        "recipe_id": "assetpack-test",
        "selected_model_id": "sdxl-turbo-quality",
    }), encoding="utf-8")

    subprocess.run([
        sys.executable,
        "scripts/update_generated_assets_index.py",
        "--root", str(root),
    ], check=True)

    body = (root / "README.md").read_text(encoding="utf-8")
    assert "# Generated assets" in body
    assert "#34" in body
    assert "assetpack-test" in body
    assert "sdxl-turbo-quality" in body
    assert "issue-000034/assetpack-test/image.png" in body
    assert "issue-000034/assetpack-test/prompt.txt" in body


def test_generated_assets_index_handles_empty_root(tmp_path):
    root = tmp_path / "assets" / "generated"

    subprocess.run([
        sys.executable,
        "scripts/update_generated_assets_index.py",
        "--root", str(root),
    ], check=True)

    body = (root / "README.md").read_text(encoding="utf-8")
    assert "No generated assets are committed yet" in body

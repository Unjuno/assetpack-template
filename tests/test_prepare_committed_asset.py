import json
import subprocess
import sys
from pathlib import Path


def write_config(path: Path) -> None:
    path.write_text(
        """
prompt_policy:
  required_terms:
    - black outline
    - white background
issue_generation:
  committed_output_root: assets/generated
""",
        encoding="utf-8",
    )


def test_prepare_committed_asset_stages_record(tmp_path):
    cfg = tmp_path / "assetpack.yml"
    write_config(cfg)

    out_dir = tmp_path / "out"
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True)
    (image_dir / "image.png").write_bytes(b"fake-png")

    request = {
        "valid": True,
        "issue_number": "34",
        "recipe_id": "assetpack-test",
        "selected_model_id": "sdxl-turbo-quality",
        "prompt": "cat, black outline, white background",
        "negative_prompt": "low quality",
    }
    report = {
        "seed": 12345,
        "summary": {"passed": 1, "failed": 0},
        "candidates": [
            {
                "candidate": {
                    "width": 256,
                    "height": 256,
                    "steps": 1,
                    "scheduler": "test-scheduler",
                    "method": "diffusers_text_to_image",
                    "pipeline_class": "AutoPipelineForText2Image",
                    "model_ref": "stabilityai/sdxl-turbo",
                },
                "width": 256,
                "height": 256,
                "steps": 1,
                "image_sha256": "abc123",
                "generate_seconds": 1.25,
                "load_seconds": 2.5,
            }
        ],
    }
    request_json = out_dir / "request.json"
    report_json = out_dir / "report.json"
    request_json.write_text(json.dumps(request), encoding="utf-8")
    report_json.write_text(json.dumps(report), encoding="utf-8")

    repo_output_root = tmp_path / "assets" / "generated"
    github_output = tmp_path / "github-output.txt"

    subprocess.run([
        sys.executable,
        "scripts/prepare_committed_asset.py",
        "--config", str(cfg),
        "--request-json", str(request_json),
        "--report-json", str(report_json),
        "--source-out-dir", str(out_dir),
        "--repo-output-root", str(repo_output_root),
        "--github-run-id", "123",
        "--github-sha", "abc",
        "--github-output", str(github_output),
    ], check=True)

    dest = repo_output_root / "issue-000034" / "assetpack-test"
    assert (dest / "image.png").read_bytes() == b"fake-png"
    assert "black outline" in (dest / "prompt.txt").read_text(encoding="utf-8")
    assert (dest / "request.json").exists()
    assert (dest / "report.json").exists()
    metadata = json.loads((dest / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["image_file_size_bytes"] == len(b"fake-png")
    assert metadata["image_width"] == 256
    assert metadata["image_height"] == 256
    assert metadata["seed"] == 12345
    assert metadata["steps"] == 1
    assert metadata["scheduler"] == "test-scheduler"
    assert metadata["model_ref"] == "stabilityai/sdxl-turbo"
    assert metadata["image_sha256"] == "abc123"
    assert (dest / "README.md").exists()
    assert "committed_asset_dir=" in github_output.read_text(encoding="utf-8")


def test_prepare_committed_asset_rejects_missing_required_term(tmp_path):
    cfg = tmp_path / "assetpack.yml"
    write_config(cfg)

    out_dir = tmp_path / "out"
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True)
    (image_dir / "image.png").write_bytes(b"fake-png")

    request = {
        "valid": True,
        "issue_number": "34",
        "recipe_id": "assetpack-test",
        "selected_model_id": "sdxl-turbo-quality",
        "prompt": "cat, black outline",
    }
    request_json = out_dir / "request.json"
    report_json = out_dir / "report.json"
    request_json.write_text(json.dumps(request), encoding="utf-8")
    report_json.write_text(json.dumps({}), encoding="utf-8")

    result = subprocess.run([
        sys.executable,
        "scripts/prepare_committed_asset.py",
        "--config", str(cfg),
        "--request-json", str(request_json),
        "--report-json", str(report_json),
        "--source-out-dir", str(out_dir),
        "--repo-output-root", str(tmp_path / "assets" / "generated"),
    ], text=True, capture_output=True)

    assert result.returncode != 0
    assert "missing required terms" in (result.stderr + result.stdout)

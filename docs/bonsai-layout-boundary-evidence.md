# Bonsai layout boundary evidence

Canonical manifest entry:

```text
docs/bonsai-lowbit-verification-manifest.json
key: bonsai_repo_layout_boundary_preflight
```

## Verified CI run

```text
run_id: 27406318749
head_sha: 4d6601d208df7c7ff68ecd63d813c0cd517506c7
job: layout-boundary
job_conclusion: success
artifact: bonsai-layout-boundary-report-json / 7587605829
artifact_sha256: 468f3168e1066365df96924341547ff0ac1c028a02e4cea07dcad97fd583ac4f
```

The report completed with:

```json
{
  "status": "passed",
  "ci_conclusion": "success",
  "required_milestone": "layout_probe",
  "milestone_reached": "layout_probe",
  "layout_probe_mode": "boundary_files",
  "download_weights": false,
  "claim_promotable_to_manifest": true
}
```

## Verified boundary files

The CI job verified the following repository boundary files without downloading model weights, exporting ONNX, or running ONNX Runtime:

```text
model_index.json
scheduler/scheduler_config.json
text_encoder/config.json
tokenizer/tokenizer_config.json
transformer/config.json
vae/config.json
```

The CI job also recorded these boundary files as absent:

```text
text_encoder_2/config.json
tokenizer_2/tokenizer_config.json
```

## Allowed claim

```text
bonsai_repo_layout_boundary_files_verified_without_weight_download_onnx_export_or_runtime_execution
```

This is a repo-layout boundary claim only. It confirms the expected top-level component configuration files are present for the Bonsai model repository. It does not claim component execution.

## Non-claims

This evidence does not verify:

- full Bonsai ONNX pipeline execution;
- real transformer block ONNX execution;
- text encoder execution;
- tokenizer execution;
- scheduler execution;
- VAE execution;
- prompt-to-image generation;
- single monolithic multi-block ONNX execution.

Any stronger claim requires a new manifest key and a new artifact report.

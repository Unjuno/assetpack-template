# Scripts

This directory contains Python utilities for validation, smoke tests, model experiments, and evidence probes.

The scripts are implementation tools. The stable template configuration remains `assetpack.yml`.

## Important scripts

| Script | Purpose |
|---|---|
| `run_image_model_ci_benchmark.py` | Runs Diffusers and Diffusers-LoRA image-generation candidates from experiment YAML files. Writes `report.json` and image artifacts. |
| `run_diffusers_smoke.py` | Lightweight Diffusers smoke-test helper. |
| `run_image_model_onnx_feasibility.py` | Records ONNX feasibility boundaries for image-model candidates. |
| `run_bonsai_onnx_probe.py` | Probes Bonsai-related ONNX boundaries. |
| `run_bonsai_combined_component_probe.py` | Records combined Bonsai component evidence. |
| `run_bonsai_lowbit_probe.py` | Probes Bonsai low-bit paths. |
| `run_bonsai_transformer_load_probe.py` | Probes transformer loading and minimal export evidence. |

## Benchmark runner contract

`run_image_model_ci_benchmark.py` reads a YAML config with:

```text
experiment_id
prompt
negative_prompt
seed
runtime
candidates
```

It writes:

```text
report.json
images/<candidate-id>/cat.png
```

The output image path still uses `cat.png` in some historical runs because the runner started as a cat benchmark. Downstream bundle scripts may rename the image for review, but the runner output path is part of the historical experiment behavior.

## Candidate methods

Supported image candidate methods include:

```text
diffusers_text_to_image
diffusers_load_only
diffusers_lora_text_to_image
diffusers_lora_load_only
```

Unsupported or placeholder candidates should produce explicit skipped or failed reports rather than pretending to be production-ready.

## Timeout control

Candidate timeout can be configured with:

```text
IMAGE_MODEL_CANDIDATE_TIMEOUT_SECONDS
```

or by passing:

```text
--candidate-timeout-seconds
```

## Production direction

Future production generation scripts should read from `assetpack.yml`, especially:

```text
models.image_generation.default_model_id
models.image_generation.allowed_model_ids
models.image_generation.runtime_override.environment_variable
```

The experiment runner can remain available for evidence and regression tests.

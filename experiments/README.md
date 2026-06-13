# Experiment configurations

This directory contains reproducibility records for model-selection and feasibility experiments.

These files are not the main template configuration contract. The main template contract is:

```text
assetpack.yml
```

## Purpose

Experiment YAML files describe fixed CI runs used to answer specific questions, such as:

- can this candidate load on a GitHub-hosted CPU runner;
- can it generate one image under a timeout;
- does it preserve subject identity under a fixed prompt;
- how does it perform on hard morphology stress tests;
- which model should be selected for the template default.

## Current production-facing result

The experiment history selected this pair for the template concept:

```text
default:   sdxl-turbo-quality
alternate: ssd-1b-lcm-lora-quality
```

The durable decision is recorded in:

```text
docs/ci/image-model-final-selection.md
docs/ci/image-model-final-selection.json
assetpack.yml
```

## Artifact policy

Experiment-generated images should stay in GitHub Actions artifacts.

Commit only:

- experiment configs;
- compact reports;
- evidence summaries;
- model-selection decisions.

Do not commit generated image batches to this directory.

## How to read these files

Most experiment configs contain:

| Field | Meaning |
|---|---|
| `experiment_id` | Stable ID for the experiment. |
| `prompt` | Fixed prompt used for that run. |
| `negative_prompt` | Fixed negative prompt when supported. |
| `runtime` | CPU/runtime constraints. |
| `candidates` | Candidate models and pipeline settings. |
| `selection_policy` | Why candidates were included or excluded. |

## Relationship to workflows

Workflow files in `.github/workflows/` may read these configs and upload one artifact per candidate or per subject/candidate pair.

The final runoff workflow generated 60 expected images:

```text
3 final candidates × 20 hard subjects
```

That result is evidence for the current model selection, not a permanent requirement that every derived repository rerun all experiments.

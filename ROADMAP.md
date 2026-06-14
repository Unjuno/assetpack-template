# Roadmap

This repository is now an Issue-driven prompt/image asset repository template.

The active contract is:

```text
GitHub Issue
  -> structured fields
  -> repository policy validation
  -> mechanical prompt
  -> selected image model
  -> committed prompt/image record under assets/generated/
  -> Issue feedback
```

## Phase 0 — Stabilize the active generation path

- [x] Use `issues` events as the active trigger.
- [x] Gate requests with the configured `asset-request` label.
- [x] Validate required fields, license, model, URLs, and unknown sections.
- [x] Enforce ASCII-only structured text fields.
- [x] Enforce `prompt_policy.required_terms` before generation.
- [x] Reject duplicate `recipe_id` records before generation.
- [x] Generate at least one smoke asset from a valid Issue.
- [x] Commit the generated PNG and prompt metadata under `assets/generated/`.
- [x] Comment generation success or failure back to the Issue.
- [x] Remove obsolete artifact-only success wording from validation and generation comments.

## Phase 1 — Keep the template surface clean

- [x] Remove legacy Bonsai / lowbit / ONNX / image-model-ci files from the active search surface.
- [x] Simplify README around Issue-driven asset generation.
- [x] Simplify CI docs to the selected model pair and active artifact policy.
- [x] Keep selected image models limited to `sdxl-turbo-quality` and `ssd-1b-lcm-lora-quality`.
- [ ] Confirm with a local `git ls-files | grep -Ei 'bonsai|lowbit|onnx|image-model-ci'` pass.

## Phase 2 — Harden duplicate and replay behavior

- [x] Add duplicate `recipe_id` detection against `assets/generated/**/<recipe_id>`.
- [ ] Add a targeted replay test for re-labeling or editing an already-generated request.
- [ ] Decide whether duplicate requests should remain hard failures or become explicit no-op comments.
- [ ] Add a workflow-level note explaining why repeated Issue events may occur.

## Phase 3 — Improve generated asset records

- [x] Commit `image.png`, `prompt.txt`, `negative_prompt.txt`, `request.json`, `metadata.json`, `report.json`, and `README.md` together.
- [x] Include a stable repository link to the committed asset directory in the success comment.
- [x] Add image size and file size to `metadata.json`.
- [ ] Add seed, scheduler, steps, width, and height to `metadata.json` when available from the generator.
- [ ] Add a generated index file for `assets/generated/` after multiple assets exist.

## Phase 4 — Test and CI hygiene

- [x] Add tests for ASCII-only rejection.
- [x] Add tests for missing required terms.
- [x] Add tests for duplicate recipe IDs.
- [x] Add a lightweight CI workflow that runs tests without invoking image generation.
- [x] Add a fixture-based test for `prepare_committed_asset.py`.
- [x] Add a fixture-based test for `write_issue_generation_comment.py`.

## Phase 5 — Documentation for derived repositories

- [x] Document persistent storage in `docs/storage-policy.md`.
- [ ] Add a derived-repository setup guide.
- [ ] Add a prompt-policy guide with required-term examples.
- [ ] Add an operations guide for failed Issues and duplicate recipes.
- [ ] Add an example Issue request that is expected to pass.
- [ ] Add an example Issue request that is expected to fail ASCII-only validation.

## Current known smoke record

Issue #34 generated and committed:

```text
assets/generated/issue-000034/assetpack-1261d48d85f2711b/
```

This proves the active Issue -> generation -> commit path works at least once.

# Roadmap

This repository is an Issue-driven prompt/image asset repository template.

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

## Status

Release-ready for template use on `main`.

The remaining items in this roadmap are optional hardening items, not blockers for the current template release.

## Phase 0 — Stabilize the active generation path

- [x] Use `issues` events as the active trigger.
- [x] Gate requests with the configured request label.
- [x] Validate required fields, license, model, URLs, and unknown sections.
- [x] Enforce ASCII-only structured text fields.
- [x] Enforce configured required terms before generation.
- [x] Reject duplicate recipe records before generation.
- [x] Generate at least one smoke asset from a valid Issue.
- [x] Commit the generated PNG and prompt metadata under `assets/generated/`.
- [x] Comment generation success or failure back to the Issue.
- [x] Propagate generation subprocess failures through the guarded Issue runner.
- [x] Use a neutral Issue generation runner entry point for active generation.

## Phase 1 — Keep the template surface clean

- [x] Simplify README around Issue-driven asset generation.
- [x] Simplify CI docs to the selected model pair and active artifact policy.
- [x] Keep selected image models limited to the configured default and alternate IDs.
- [x] Move active generation off the old runner implementation.
- [x] Tombstone obsolete probe, smoke, benchmark, ONNX, Bonsai, and lowbit helpers where deletion is not appropriate or is blocked.
- [x] Tombstone obsolete generated evidence records under `docs/ci`.
- [x] Disable legacy experiment workflows with inert `workflow_dispatch` workflows.
- [x] Verify that major legacy search terms no longer resolve to current active implementation paths, except active/generated records.

## Phase 2 — Harden duplicate and replay behavior

- [x] Add duplicate recipe detection against committed generated records.
- [x] Add a policy replay test for an already-generated request.
- [x] Treat duplicate requests as existing-asset status comments instead of generation attempts.
- [x] Add a workflow note explaining why repeated Issue events may occur.
- [x] Narrow the Issue workflow trigger to avoid duplicate `opened`/`labeled` runs.

## Phase 3 — Improve generated asset records

- [x] Commit `image.png`, `prompt.txt`, `negative_prompt.txt`, `request.json`, `metadata.json`, `report.json`, and `README.md` together.
- [x] Include a stable repository link to the committed asset directory in the success comment.
- [x] Add image size and file size to `metadata.json`.
- [x] Add seed, scheduler, steps, width, and height to `metadata.json` when available from the generator.
- [x] Add a generated index file for `assets/generated/` after multiple assets exist.

## Phase 4 — Test and CI hygiene

- [x] Add tests for ASCII-only rejection.
- [x] Add tests for missing required terms.
- [x] Add tests for duplicate recipe IDs.
- [x] Add a lightweight CI workflow that runs tests without invoking image generation.
- [x] Add a fixture-based test for `prepare_committed_asset.py`.
- [x] Add a fixture-based test for `write_issue_generation_comment.py`.

## Phase 5 — Documentation for external and derived repository users

- [x] Document persistent storage in `docs/storage-policy.md`.
- [x] Add a derived-repository setup guide.
- [x] Add a prompt-policy guide with required-term examples.
- [x] Add an operations guide for failed Issues and duplicate recipes.
- [x] Add example Issue requests.
- [x] Add an external user manual.
- [x] Add a maintainer release checklist for publishing a derived repository.
- [x] Confirm that README, docs index, Issue template, and manual describe the same request fields and model options.
- [x] Document how derived repositories should reset template-generated sample assets.

## Release validation

- [x] Submit one fresh smoke Issue after documentation stabilization.
- [x] Confirm the smoke Issue generates and commits an asset.
- [x] Confirm duplicate replay behavior after the fresh smoke Issue.
- [x] Confirm one intentionally invalid Issue fails before generation.
- [x] Confirm a manual smoke Issue generated and committed an asset.
- [x] Document the release validation record.

Evidence is recorded in:

```text
docs/release-validation.md
```

## Optional hardening after this release

- [ ] Add a separate `required_issue_terms` policy if a derived repository needs specific words to appear in the user-submitted Issue fields.
- [ ] Add a more direct workflow-run status reference for the lightweight test workflow if a future automation surface exposes run IDs.

## Current known smoke records

Issue #34 generated and committed:

```text
assets/generated/issue-000034/assetpack-1261d48d85f2711b/
```

Issue #37 generated and committed:

```text
assets/generated/issue-000037/assetpack-d5d6ce9df8725f4c/
```

Issue #39 generated and committed:

```text
assets/generated/issue-000039/assetpack-315db6d7e5f10403/
```

These records prove the active Issue -> generation -> commit path works on `main`.

# Roadmap

This roadmap tracks the active template surface for issue-driven image asset generation.

## P0 — production correctness

- [x] Confirm valid Issue requests can generate an image.
- [x] Confirm generated `prompt.txt` and `image.png` are committed under `assets/generated/`.
- [x] Remove stale artifact-only wording from validation acceptance comments.
- [x] Remove stale artifact-only wording from generation result comments.
- [x] Prevent duplicate generation/comments when one Issue emits multiple events.
- [x] Make duplicate `recipe_id` rejection deterministic under concurrent issue events by serializing workflow runs per Issue.
- [ ] Verify failure comments for ASCII-only rejection, missing required terms, and duplicate recipe IDs.

## P1 — repository contract hardening

- [x] Keep `assetpack.yml` as the single source of truth for generation policy.
- [x] Document the exact Issue form field contract.
- [x] Document committed asset directory layout with an example.
- [x] Document storage constraints and expected image size discipline.
- [x] Add a cleanup policy for smoke-test Issues and generated smoke assets.

## P2 — test and CI coverage

- [x] Add tests for ASCII-only rejection.
- [x] Add tests for missing required terms.
- [x] Add tests for duplicate recipe IDs.
- [x] Add a dry-run validation command documented in README.
- [ ] Add a workflow syntax guard or lightweight lint check.
- [x] Add a smoke-test procedure that does not create duplicate generated assets.

## P3 — template polish

- [x] Remove Bonsai / lowbit / ONNX research surface from current code search.
- [x] Simplify README to the active Issue-driven asset repository contract.
- [ ] Review `docs/` for stale references after the generation success path is finalized.
- [ ] Review `scripts/` names and comments for template readability.
- [ ] Decide whether smoke-generated assets should remain as examples or be removed before template release.

## Current known facts

- Issue #34 generated a committed asset record.
- Generated asset path: `assets/generated/issue-000034/assetpack-1261d48d85f2711b/`.
- The final prompt contains all configured required terms.
- The generated PNG exists in Git.

## Next action

Verify failure comments for ASCII-only rejection and duplicate recipe IDs. Missing required-term verification remains covered by policy tests unless the prompt template is intentionally changed in a test branch.

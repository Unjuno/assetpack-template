# Roadmap

This roadmap tracks the active template surface for issue-driven image asset generation.

## P0 — production correctness

- [x] Confirm valid Issue requests can generate an image.
- [x] Confirm generated `prompt.txt` and `image.png` are committed under `assets/generated/`.
- [x] Remove stale artifact-only wording from validation acceptance comments.
- [x] Remove stale artifact-only wording from generation result comments.
- [ ] Prevent duplicate generation/comments when one Issue emits multiple events.
- [ ] Make duplicate `recipe_id` rejection deterministic under concurrent issue events.
- [ ] Verify failure comments for ASCII-only rejection, missing required terms, and duplicate recipe IDs.

## P1 — repository contract hardening

- [ ] Keep `assetpack.yml` as the single source of truth for generation policy.
- [ ] Document the exact Issue form field contract.
- [ ] Document committed asset directory layout with an example.
- [ ] Document storage constraints and expected image size discipline.
- [ ] Add a cleanup policy for smoke-test Issues and generated smoke assets.

## P2 — test and CI coverage

- [x] Add tests for ASCII-only rejection.
- [x] Add tests for missing required terms.
- [x] Add tests for duplicate recipe IDs.
- [ ] Add a dry-run validation command documented in README.
- [ ] Add a workflow syntax guard or lightweight lint check.
- [ ] Add a smoke-test procedure that does not create duplicate generated assets.

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

Serialize issue generation per Issue number so `opened` and `labeled` events cannot produce duplicate image/comment records for the same request.

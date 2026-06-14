# TODO: assetpack-template cleanup and issue generation hardening

## P0 — workflow correctness

- [x] Restore a safe label gate for issue-triggered generation without breaking issue-form triggers.
- [x] Ensure validation failures produce an Issue comment.
- [x] Ensure required-term failures stop generation before dependency install.
- [x] Ensure generation failures produce an Issue comment.
- [x] Ensure successful generation produces an Issue comment with committed asset paths.

## P0 — validation policy

- [x] Add ASCII-only validation for structured text fields.
- [x] Keep license and model allowlist validation.
- [x] Keep URL and unknown-section rejection.
- [x] Validate final mechanical prompt contains every configured required term.
- [x] Reject duplicate recipe IDs before generation.

## P0 — repo-persistent asset output

- [x] Add a script that stages generated PNG, prompt, request, report, and metadata under `assets/generated/`.
- [x] Change workflow permissions to allow committing generated assets.
- [x] Add commit/push step with retry.

## P1 — repository cleanup

- [x] Remove active Bonsai / lowbit / ONNX probe workflows found in the active workflow surface.
- [~] Remove Bonsai / lowbit / ONNX scripts from active `scripts/` surface. Some files were blocked by tool safety checks and remain.
- [~] Remove Bonsai / lowbit / ONNX docs and experiments from active docs surface. Some files were blocked by tool safety checks or search/index uncertainty and remain.
- [x] Rewrite README around Issue-driven asset repository generation.
- [x] Rewrite docs to match repo-persistent generated assets where tool safety allowed.

## P2 — verification

- [ ] Add or update tests for ASCII-only rejection.
- [ ] Add or update tests for missing required terms.
- [ ] Add or update tests for duplicate recipe IDs.

## Blocked follow-up

- `scripts/bonsai_prompt_z.py` could not be deleted or stubbed by the connector safety checks.
- Some legacy docs/scripts may remain because GitHub search returned stale commit-index results and several delete/update calls were blocked.

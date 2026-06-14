# TODO: assetpack-template cleanup and issue generation hardening

## P0 — workflow correctness

- [ ] Restore a safe label gate for issue-triggered generation without breaking issue-form triggers.
- [ ] Ensure validation failures always produce an Issue comment.
- [ ] Ensure required-term failures stop generation before dependency install.
- [ ] Ensure generation failures produce an Issue comment.
- [ ] Ensure successful generation produces an Issue comment with committed asset paths.

## P0 — validation policy

- [ ] Add ASCII-only validation for structured text fields.
- [ ] Keep license and model allowlist validation.
- [ ] Keep URL and unknown-section rejection.
- [ ] Validate final mechanical prompt contains every configured required term.
- [ ] Reject duplicate recipe IDs before generation.

## P0 — repo-persistent asset output

- [ ] Add a script that stages generated PNG, prompt, request, report, and metadata under `assets/generated/`.
- [ ] Change workflow permissions to allow committing generated assets.
- [ ] Add commit/push step with retry.

## P1 — repository cleanup

- [ ] Remove active Bonsai / lowbit / ONNX probe workflows from the template surface.
- [ ] Remove Bonsai / lowbit / ONNX scripts from active `scripts/` surface.
- [ ] Remove Bonsai / lowbit / ONNX docs and experiments from active docs surface.
- [ ] Rewrite README around Issue-driven prompt/image corpus generation.
- [ ] Rewrite docs to match repo-persistent generated assets.

## P2 — verification

- [ ] Add or update tests for ASCII-only rejection.
- [ ] Add or update tests for missing required terms.
- [ ] Add or update tests for duplicate recipe IDs.

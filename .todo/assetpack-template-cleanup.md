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

- [x] Remove active legacy probe and model-selection workflows from the template surface.
- [x] Remove many legacy research scripts from active `scripts/` surface.
- [x] Remove legacy experiment YAML files found through current cleanup passes.
- [x] Simplify model-selection docs to the selected model pair only.
- [x] Rewrite README around Issue-driven asset repository generation.
- [x] Rewrite docs to match repo-persistent generated assets where tool safety allowed.

## P2 — verification

- [x] Add or update tests for ASCII-only rejection.
- [x] Add or update tests for missing required terms.
- [x] Add or update tests for duplicate recipe IDs.

## Follow-up

- GitHub code search may still show stale results from older commits. Confirm remaining files with `git ls-files` or `fetch_file` before further deletion.
- Some old CI evidence JSON directories may still remain under `docs/ci/` if not directly required by `assetpack.yml`.

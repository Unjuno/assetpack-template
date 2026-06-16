# Final text audit

This document records the final text-level audit for the template release.

Audit date: 2026-06-16

## Scope

Reviewed public and maintainer-facing text in:

- `README.md`
- `ROADMAP.md`
- `docs/README.md`
- `docs/user-manual.md`
- `docs/template-build-and-use.md`
- `docs/issue-driven-generation.md`
- `docs/storage-policy.md`
- `docs/derived-repository-setup.md`
- `docs/prompt-policy-guide.md`
- `docs/operations-guide.md`
- `docs/example-issues.md`
- `docs/maintainer-release-checklist.md`
- `docs/implementation-status.md`
- `docs/release-validation.md`
- `.github/ISSUE_TEMPLATE/generate.yml`
- `.github/workflows/assetpack-issue-generate.yml`
- `.github/workflows/assetpack-tests.yml`
- `assetpack.yml`

## Findings fixed before release

- Removed stale references to the Issue `opened` event as a generation trigger.
- Clarified that generation starts from `labeled`, `edited`, and `reopened` Issue events.
- Removed the unused `needs-validation` label from the Issue template.
- Clarified that the Issue template normally applies `asset-request`, and users should confirm the label is present rather than applying it blindly.
- Clarified that duplicate recipe ids resolve to existing generated assets instead of starting a new generation attempt.
- Added derived-repository reset guidance for template-generated sample records.
- Added maintainer checklist items for resetting `assets/generated/issue-*` records in derived repositories.
- Documented the manual smoke Issue #39 and the follow-up workflow trigger narrowing.
- Aligned README release validation summary with the release validation evidence.

## Current verified behavior

- The active Issue generation workflow listens to `labeled`, `edited`, and `reopened` events.
- The `Asset request` Issue template applies only the `asset-request` label.
- Valid structured requests can generate committed image records under `assets/generated/`.
- Duplicate structured requests resolve to an existing asset path.
- Invalid structured requests fail before generation.
- Derived repositories should reset copied generated sample records before accepting real requests.

## Known non-blocking notes

- `prompt_policy.required_terms` applies to the generated final prompt, not to user-submitted Issue fields.
- A separate `required_issue_terms` policy can be added later if a derived repository needs specific words to appear in user-submitted Issue fields.
- The lightweight test workflow currently runs on `push` to `main` and `pull_request`; local test commands are documented in the template guide.

## Release decision

No release-blocking text inconsistencies remain after this audit.

The repository is suitable for release as an Issue-driven image asset generation template.

# Release validation

This document records the final release validation performed against `main`.

Validation date: 2026-06-16

## Summary

The Issue-driven generation path has been validated end to end.

| Check | Result | Evidence |
| --- | --- | --- |
| Fresh smoke Issue | PASS | Issue #37 generated and committed `assets/generated/issue-000037/assetpack-d5d6ce9df8725f4c`. |
| Duplicate replay | PASS | Issue #38 resolved to the existing asset from Issue #37. |
| Invalid request | PASS | Issue #36 was rejected before generation because the `scene` field contained a URL. |
| Existing duplicate | PASS | Issue #35 resolved to the existing asset from Issue #34. |

## Fresh smoke Issue

Issue #37 used this structured request:

```md
### Subject
round owl with tiny glasses

### Scene
perched on a simple branch

### Audience
children

### Constraints
simple composition, friendly pose, clear large shapes

### Model
default

### License
CC0-1.0
```

The workflow generated an image using `sdxl-turbo-quality` and committed:

```text
assets/generated/issue-000037/assetpack-d5d6ce9df8725f4c
```

Expected committed paths were reported by the workflow:

```text
assets/generated/issue-000037/assetpack-d5d6ce9df8725f4c/image.png
assets/generated/issue-000037/assetpack-d5d6ce9df8725f4c/prompt.txt
```

## Duplicate replay

Issue #38 repeated the same structured request as Issue #37.

The workflow reported that the asset already exists:

```text
assets/generated/issue-000037/assetpack-d5d6ce9df8725f4c
```

This confirms that duplicate detection prevents unnecessary regeneration for the same recipe id.

## Invalid request

Issue #36 included a URL in the `scene` field.

The workflow rejected the request before image generation with:

```text
URL found in field: scene
```

No generated asset should be committed for this invalid request.

## Existing duplicate

Issue #35 repeated an older request and resolved to the existing asset:

```text
assets/generated/issue-000034/assetpack-1261d48d85f2711b
```

## Known caveats

These caveats are documented but are not blockers for this template release:

- An Issue that is opened and labeled in the same interaction can produce duplicate bot comments because both `opened` and `labeled` events may run.
- `prompt_policy.required_terms` applies to the generated final prompt. It is not an Issue-field required-word policy.
- A derived repository should reset template-generated sample records before accepting real requests.
- The lightweight test workflow exists and is part of the repository, but this release validation record focuses on the Issue-driven generation path.

## Release judgment

The repository is suitable for release as an Issue-driven image asset generation template.

Recommended follow-up hardening:

- Reduce duplicate bot comments when an Issue is both opened and labeled in the same interaction.
- Optionally add a separate `required_issue_terms` policy if a derived repository wants specific words to appear in the user-submitted Issue fields, not just in the final generated prompt.

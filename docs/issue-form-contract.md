# Issue form contract

Asset requests are structured GitHub Issues. They are not free-form prompts.

## Required fields

- `subject`
- `scene`
- `audience`
- `license`

## Optional fields

- `constraints`
- `model`

## Policy

- Text fields are ASCII-only.
- URLs are rejected.
- Unknown nonempty sections are rejected.
- `license` must be one of the configured values in `assetpack.yml`.
- `model` must be `default` or one of the configured allowed model IDs.
- The final mechanical prompt must contain every configured `prompt_policy.required_terms` value.
- A `recipe_id` that already exists under `assets/generated/` is rejected before generation.

## Persistent output

Successful generation stores a record under:

```text
assets/generated/issue-<issue-number>/<recipe-id>/
```

Each committed record should keep the image, prompt, request, metadata, and report together.

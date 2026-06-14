# assetpack-template

Issue-driven asset repository template.

Use GitHub Issues for structured requests. CI validates each request with `assetpack.yml`, builds a fixed recipe, runs the selected generator, and stores the resulting record under `assets/generated/`.

## Rules

- Structured fields only.
- ASCII-only text fields.
- Required terms must be present in the final recipe.
- Duplicate recipe IDs are rejected.
- Success and failure are reported back to the Issue.

## Main path

```text
.github/ISSUE_TEMPLATE/generate.yml
.github/workflows/assetpack-issue-generate.yml
assets/generated/
```

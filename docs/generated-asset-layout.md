# Generated asset layout

Committed generated assets live under:

```text
assets/generated/issue-<issue-number>/<recipe-id>/
```

Example:

```text
assets/generated/issue-000034/assetpack-1261d48d85f2711b/
  image.png
  prompt.txt
  negative_prompt.txt
  request.json
  metadata.json
  report.json
  README.md
```

## Required files

- `image.png`: generated PNG image.
- `prompt.txt`: final mechanical prompt.
- `negative_prompt.txt`: negative prompt passed to the generator.
- `request.json`: parsed Issue fields and validation data.
- `metadata.json`: workflow and model metadata.
- `report.json`: generation result summary.
- `README.md`: local human-readable summary.

A generated PNG without its prompt and metadata is not a complete persistent record.

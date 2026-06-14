# Operations guide

This guide covers normal operator decisions for Issue-driven asset generation.

## Normal success path

A valid request with the configured label should produce:

```text
assets/generated/issue-<number>/<recipe_id>/
```

The directory should contain:

```text
image.png
prompt.txt
negative_prompt.txt
request.json
metadata.json
report.json
README.md
```

The Issue should receive a success comment with the committed asset path.

## Validation failure

Validation stops before generation. Typical causes:

- Missing required section.
- Invalid license.
- Invalid model id.
- URL in a structured field.
- Unknown non-empty section.
- Non-ASCII text in a checked field.
- Required term missing from the final prompt.

Action: edit the Issue fields and re-run by editing or re-labeling the Issue.

## Duplicate recipe

A repeated request can happen when an Issue is opened, edited, reopened, or labeled.

If the request resolves to an existing `recipe_id`, the workflow treats it as an existing asset status. It should not generate another image. The Issue comment should point at the existing directory under `assets/generated/`.

Action: use the existing asset, or change one of the structured fields if a new asset is required.

## Generation failure

Generation can fail after validation because of model download, timeout, dependency, or runner resource limits.

Action:

1. Open the workflow run.
2. Check the generation step logs.
3. Check the uploaded artifact for `request.json` and `report.json`.
4. If the failure is transient, re-run the workflow.
5. If the failure is deterministic, update `assetpack.yml`, model config, or generation limits.

## Commit failure

The generation step can pass while the commit step fails because the branch moved or repository permissions changed.

Action:

1. Confirm the workflow token has `contents: write`.
2. Confirm branch protection allows the workflow commit pattern.
3. Re-run the workflow after the branch is stable.

## Manual cleanup

Do not delete committed generated records unless the repository policy requires it. Each generated asset is intended to be a durable prompt/image record.

If a record must be removed, remove the full directory and update `assets/generated/README.md`.

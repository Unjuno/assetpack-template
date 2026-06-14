# Storage constraints

This repository intentionally commits generated prompt/image records to Git.

The policy is deliberate: generated assets are part of the persistent asset corpus.

## Practical constraints

- Keep one generated PNG per accepted Issue request.
- Keep generated images small enough for normal Git repository use.
- Keep the prompt, image, request, metadata, and report together.
- Do not overwrite existing generated records.
- Reject duplicate `recipe_id` records before image generation.

## Recommended defaults

- `images_per_prompt`: `1`
- committed output root: `assets/generated/`
- generated record layout: `assets/generated/issue-<issue-number>/<recipe-id>/`

## Cleanup

Smoke-test generated assets may be kept as examples while validating the template. Before release, decide explicitly whether to keep them as examples or remove them.

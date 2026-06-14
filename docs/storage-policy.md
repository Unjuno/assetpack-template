# Storage policy

Generated records are stored under `assets/generated/`.

Each generated record must keep the generated image, final prompt, request data, metadata, and report together.

Validation must reject non-ASCII structured text fields, missing required prompt terms, and duplicate recipe IDs before image generation.

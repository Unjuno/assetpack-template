# Storage policy

Generated records are stored under `assets/generated/`.

Each generated record must keep the generated image, final prompt, request data, metadata, and report together.

A successful generated record should contain:

```text
image.png
prompt.txt
negative_prompt.txt
request.json
metadata.json
report.json
README.md
```

Validation must reject non-ASCII structured text fields and missing required prompt terms before image generation.

Duplicate recipe ids are handled as existing generated records, not as generation attempts. If a matching `recipe_id` already exists under `assets/generated/`, the workflow should comment with the existing asset path instead of generating another image.

Derived repositories should reset template-generated sample records before accepting real requests.

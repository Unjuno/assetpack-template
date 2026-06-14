# Smoke-test procedure

Use smoke-test Issues to verify the active workflow.

## Valid generation smoke

Expected result:

- validation passes;
- image generation runs;
- prompt/image record is committed under `assets/generated/`;
- Issue comment includes committed paths.

## ASCII rejection smoke

Use a non-ASCII character in a structured text field.

Expected result:

- validation fails;
- no image is generated;
- no files are committed under `assets/generated/`;
- Issue comment explains the ASCII-only policy failure.

## Duplicate recipe smoke

Submit the same structured fields as an already committed generated asset.

Expected result:

- validation fails before generation;
- no new image is generated;
- Issue comment reports duplicate `recipe_id` and the existing path.

## Required term failure

This is covered by tests unless the prompt template is intentionally changed in a test branch, because normal prompts inject `prompt_policy.required_terms` mechanically.

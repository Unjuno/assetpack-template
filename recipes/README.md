# Recipes

This directory is for prompt recipes generated from structured input.

A recipe is not just a raw prompt. It should preserve enough structured information to explain how the prompt was produced.

## Suggested layout

```text
recipes/
  generated/
  approved/
```

| Path | Meaning |
|---|---|
| `generated/` | Machine-generated recipe candidates. |
| `approved/` | Human-reviewed recipes accepted for use. |

## Recipe contents

A recipe should include:

- stable recipe ID;
- source structured input;
- generated prompt;
- required terms applied;
- constraint checks;
- dedupe hash;
- selected model ID if generation was attempted;
- output report reference if available.

Generated images should not be stored here by default. Store recipe metadata and keep images as CI artifacts unless a reviewed publish step is added.

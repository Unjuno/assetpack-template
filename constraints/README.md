# Constraints

This directory is for validation and prompt-policy constraints.

A derived repository should use constraints to keep generation aligned with the fixed theme.

## Typical files

| File | Purpose |
|---|---|
| `required_terms.json` | Terms that must appear in generated prompts. |
| `banned_terms.json` | Terms or patterns that should be rejected. |
| `dedupe.json` | Dedupe settings or thresholds. |
| `schema.recipe.json` | JSON Schema for generated prompt recipes. |

## Design rule

Constraints should be checked before image generation is attempted.

The goal is to keep CI runs predictable and to prevent arbitrary prompt expansion from becoming the default behavior.

# Prompt templates

This directory is for mechanical prompt templates.

A derived repository should not rely on arbitrary free-form prompts by default. It should define fixed templates that combine structured fields with required terms and constraints.

## Intended pattern

A template should define stable slots such as:

```text
subject
scene
style
audience
constraints
license
```

Example concept:

```text
Create a {category} asset of {subject} in {scene}, using {style}, with {constraints}.
```

The exact format can be YAML, JSON, or another structured form, but it should be deterministic and reviewable.

## Relationship to `assetpack.yml`

`assetpack.yml` defines the theme and prompt policy. Files in this directory provide the reusable prompt structures used by recipe builders.

## Non-goal

Do not store one-off prompts here as production logic. One-off prompts belong in experiments or local scratch work, not in the template contract.

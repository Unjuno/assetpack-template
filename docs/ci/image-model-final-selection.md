# Final image model selection

## Decision

Use `sdxl-turbo-quality` as the default image-generation model.

Keep `ssd-1b-lcm-lora-quality` as the configurable alternate.

## Configuration source

```text
assetpack.yml
```

Allowed runtime values:

```text
sdxl-turbo-quality
ssd-1b-lcm-lora-quality
```

Runtime override key:

```text
ASSETPACK_IMAGE_MODEL_ID
```

## Scope

This is a repository-specific model decision. It is not a general image-model benchmark.

Generated Issue assets are committed with their prompt and metadata under `assets/generated/`.

# Final image model selection

## Decision

Use `sdxl-turbo-quality` as the default image-generation model for this repository's current fixed-theme asset-generation concept.

Keep `ssd-1b-lcm-lora-quality` as the configurable alternate when visual asset style is prioritized.

Do not select `segmind-vega-quality` as the default or alternate after the final runoff. It remains a useful experimental reference, but it is less reliable on hard morphology and target-identity preservation than the selected pair.

## Selected configuration

Machine-readable selection source:

```text
assetpack.yml
```

Evidence record:

```text
docs/ci/image-model-final-selection.json
```

Runtime override key:

```text
ASSETPACK_IMAGE_MODEL_ID
```

Allowed runtime values:

```text
sdxl-turbo-quality
ssd-1b-lcm-lora-quality
```

Default:

```text
sdxl-turbo-quality
```

Alternate:

```text
ssd-1b-lcm-lora-quality
```

## Evidence summary

Final runoff workflow:

```text
image-model-ci-final-runoff-hard-subjects
```

Run:

```text
27446158099
```

Head SHA:

```text
dfd09fe5cc7f964583072dff3782ad81cbbb4814
```

The final runoff used 3 candidates across 20 intentionally difficult subjects:

```text
3 candidates × 20 subjects = 60 expected images
```

Artifact/image extraction result:

```text
expected artifacts: 60
downloaded artifacts: 60
expected images: 60
extracted PNGs: 57
no PNG: 3
```

No-PNG cases:

```text
stag-beetle / segmind-vega-quality
lacewing / sdxl-turbo-quality
pangolin / ssd-1b-lcm-lora-quality
```

## Final ranking

| Rank | Candidate | Status | Rationale |
|---:|---|---|---|
| 1 | `sdxl-turbo-quality` | selected default | Best overall target identity and topology preservation across the 20 hard-subject runoff. |
| 2 | `ssd-1b-lcm-lora-quality` | configurable alternate | Close second; strong visual asset quality and good enough morphology retention to remain selectable. |
| 3 | `segmind-vega-quality` | not selected | Often visually attractive, but less reliable on hard morphology and subject identity. |

## Evaluation principle

This selection prioritizes:

1. target identity;
2. topology and anatomy;
3. small parts, such as legs, wings, antennae, tentacles, and symmetry;
4. contact with the bonsai branch, root, pot, or water surface;
5. usable asset appearance.

Attractive images alone were not enough to win the runoff.

## Scope of claim

This is an evidence-based selection for this repository's current fixed-theme asset-generation concept.

It is not a general model-quality benchmark and does not claim universal superiority over other image-generation models.

Generated images are not committed to Git by default. They remain CI artifacts. The repository stores configuration and evidence records only.

# Assetpack policy

`assetpack.yml` is the source of truth for this template.

It defines:

- the locked theme;
- Issue generation settings;
- structured field limits;
- ASCII-only input policy;
- prompt recipe;
- required prompt terms;
- license allowlist;
- selected image model IDs;
- committed output root.

Derived repositories should change `assetpack.yml` first, then update Issue forms and documentation to match.

Do not treat Issue text as a free prompt. The final prompt is mechanical and repository-controlled.

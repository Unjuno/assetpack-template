# assetpack-template

Issue-driven image asset generation template for GitHub repositories.

This template lets a repository accept structured GitHub Issues, validate them against a fixed policy, generate an image with a configured model, and commit the generated image record under `assets/generated/`.

```text
GitHub Issue + asset-request label
  -> request validation
  -> policy check
  -> fixed prompt recipe
  -> configured image model
  -> committed asset record under assets/generated/
  -> Issue success/failure comment
```

## Status

Release-ready for template use on `main`.

The release validation passed:

1. fresh smoke Issue,
2. duplicate replay check,
3. invalid Issue check,
4. manual smoke Issue.

See [Release validation](docs/release-validation.md) and [Implementation status](docs/implementation-status.md).

Recommended follow-up hardening:

- optionally add a separate Issue-field required-term policy for derived repositories that need user-submitted words, not only configured final-prompt terms.

## What this template provides

- A structured Issue template for image asset requests.
- A GitHub Actions workflow that runs from Issues.
- Label-gated generation using `asset-request`.
- Validation for required fields, license, model id, URLs, unknown sections, and ASCII-only fields.
- A fixed prompt recipe controlled by `assetpack.yml`.
- Required prompt terms injected by configuration.
- Default and alternate image model configuration.
- Generated asset commits under `assets/generated/`.
- Success, duplicate, validation-failure, and generation-failure Issue comments.
- User and maintainer documentation.

## Quick start for a derived repository

1. Create a new repository from this template or clone it into a new repository.
2. Reset template-generated sample assets before accepting requests:

   ```bash
   python scripts/reset_template_generated_assets.py --yes
   git add assets/generated
   git commit -m "Reset generated assets for this repository"
   ```

   This removes `assets/generated/issue-*` directories copied from the template and resets `assets/generated/README.md` to an empty generated-asset index. Keep the `assets/generated/` root itself.

3. Enable GitHub Actions.
4. Create the request label:

   ```text
   asset-request
   ```

5. Review `assetpack.yml` and adjust the repository theme, prompt recipe, policy, license list, and model settings.
6. Run the lightweight test workflow:

   ```text
   .github/workflows/assetpack-tests.yml
   ```

7. Open a new Issue using the `Asset request` template.
8. Confirm that the `asset-request` label is present. If the template did not add it automatically, apply it after checking the fields.
9. Wait for the generation workflow to finish.
10. Confirm that the generated record appears under:

   ```text
   assets/generated/issue-<issue-number>/<recipe_id>/
   ```

11. Confirm that the Issue received a success or failure comment.

## Trigger behavior

The generation workflow intentionally starts from Issue `labeled`, `edited`, and `reopened` events. It does not start from `opened` events. This prevents duplicate generation runs when an Issue is created with the request label already attached.

The normal manual path is:

```text
create or edit Issue -> confirm or apply asset-request label -> wait for CI result comment
```

## Example Issue request

Use the Issue form fields, not a free-form prompt.

```md
### Subject
sleeping fox

### Scene
sitting on a white rug

### Audience
children

### Constraints
simple composition, friendly pose

### Model
default

### License
CC0-1.0
```

## Request policy

Requests are intentionally structured. The repository does not accept arbitrary free prompts.

Required fields:

- `subject`
- `scene`
- `audience`
- `license`

Optional fields:

- `constraints`
- `model`

Default policy highlights:

- structured fields only,
- ASCII-only checked fields,
- URLs rejected from request fields,
- unknown non-empty sections rejected,
- license must be one of the configured values,
- model must be `default` or one of the configured model ids,
- final prompt must contain configured required terms.

The default required prompt terms are injected by `assetpack.yml`:

```text
black outline
white background
closed regions
no text
```

Normal users do not need to type these terms manually. The configured prompt recipe inserts them into the final prompt, and policy validation confirms they are present before generation.

## Models

The default template currently exposes:

| Role | Model id |
| --- | --- |
| Default | `sdxl-turbo-quality` |
| Alternate | `ssd-1b-lcm-lora-quality` |

Use `default` in normal Issue requests unless a maintainer asks for a specific model id.

## Generated output

A successful generation commits a directory like:

```text
assets/generated/issue-000034/assetpack-1261d48d85f2711b/
```

Expected files:

```text
image.png
prompt.txt
negative_prompt.txt
request.json
metadata.json
report.json
README.md
```

The generated asset index is:

```text
assets/generated/README.md
```

## Duplicate behavior

The same structured request resolves to the same recipe id.

If that recipe already exists under `assets/generated/`, the workflow should not generate a second image. It should comment with the existing asset path instead.

## Failure behavior

If validation fails, no image is generated and no generated asset record is committed.

Common validation failures include:

- missing required field,
- invalid license,
- invalid model id,
- URL in a structured field,
- unknown non-empty section,
- non-ASCII text in a checked field,
- missing required term in the final prompt.

If generation or commit fails, the workflow comments back to the Issue and uploads the issue-generation artifact for inspection.

## Main files

```text
assetpack.yml
.github/ISSUE_TEMPLATE/generate.yml
.github/workflows/assetpack-issue-generate.yml
.github/workflows/assetpack-tests.yml
scripts/validate_issue_request.py
scripts/validate_issue_policy.py
scripts/run_issue_safe.py
scripts/run_issue_image_generation.py
scripts/prepare_committed_asset.py
scripts/write_issue_generation_comment.py
scripts/reset_template_generated_assets.py
assets/generated/
```

## Documentation

- [User manual](docs/user-manual.md)
- [Template build and use guide](docs/template-build-and-use.md)
- [Maintainer release checklist](docs/maintainer-release-checklist.md)
- [Release validation](docs/release-validation.md)
- [Implementation status](docs/implementation-status.md)
- [Documentation index](docs/README.md)
- [Roadmap](ROADMAP.md)
- [Generated assets index](assets/generated/README.md)

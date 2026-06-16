# Derived repository setup

Use this template when a repository should store generated prompt/image records through GitHub Issues.

## 1. Copy the template

Create a new repository from this template or copy the files into an existing repository.

Keep these paths:

```text
.github/workflows/assetpack-issue-generate.yml
.github/workflows/assetpack-tests.yml
assetpack.yml
scripts/
tests/
docs/
assets/generated/
```

## 2. Reset generated sample records

A derived repository should start with its own generated records, not the template repository's validation records.

Run this after creating the derived repository:

```bash
python scripts/reset_template_generated_assets.py --yes
git add assets/generated
git commit -m "Reset generated assets for this repository"
```

Dry run:

```bash
python scripts/reset_template_generated_assets.py
```

The command targets generated issue record directories under `assets/generated/issue-*` and keeps the `assets/generated/` root available for future CI commits.

## 3. Configure the repository contract

Edit `assetpack.yml` before accepting real requests.

Minimum fields to review:

- `theme.description`
- `prompt_recipe.template`
- `prompt_policy.required_terms`
- `input_policy.ascii_fields`
- `issue_generation.required_label`
- `issue_generation.committed_output_root`
- `models.image_generation.default_model_id`
- `models.image_generation.allowed_model_ids`

The default repository contract assumes generated records are committed under `assets/generated/`.

## 4. Configure labels

Create the label configured by `issue_generation.required_label`.

Default:

```text
asset-request
```

Issues without this label are ignored by the generation workflow.

## 5. Enable GitHub Actions

The repository needs Actions enabled and a workflow token that can write repository contents and Issue comments.

The generation workflow uses:

```yaml
permissions:
  contents: write
  issues: write
```

The lightweight test workflow uses read-only repository permissions.

## 6. Submit a smoke request

Create an Issue with the required sections and apply the request label.

The first smoke request should use a simple subject and should keep all text ASCII-only.

On success, the workflow commits a directory like:

```text
assets/generated/issue-000001/<recipe_id>/
```

The Issue receives a comment with the committed asset path.

## 7. Review generated records

Each generated asset directory should contain:

```text
image.png
prompt.txt
negative_prompt.txt
request.json
metadata.json
report.json
README.md
```

The generated root index is stored at:

```text
assets/generated/README.md
```

Run the test workflow after setup changes before accepting normal requests.

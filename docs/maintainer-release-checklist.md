# Maintainer release checklist

Use this checklist before publishing a repository derived from `assetpack-template` for external users.

## 1. Repository settings

- [ ] GitHub Actions are enabled.
- [ ] The workflow token can write repository contents.
- [ ] The workflow token can comment on Issues.
- [ ] Branch protection allows the generation workflow to commit generated records, or an equivalent merge path is configured.
- [ ] The request label exists. The default label is `asset-request`.
- [ ] The `Asset request` Issue template is visible to users.

## 2. Policy configuration

- [ ] `assetpack.yml` describes the repository theme.
- [ ] `prompt_recipe.template` matches the desired asset style.
- [ ] `prompt_policy.required_terms` contains only terms that should always be present.
- [ ] `input_policy.ascii_only` and `ascii_fields` match the expected user population.
- [ ] `issue_generation.required_label` matches the actual GitHub label.
- [ ] `issue_generation.committed_output_root` points to `assets/generated` unless the repository intentionally uses a different path.

## 3. Model configuration

- [ ] The default model is appropriate for normal requests.
- [ ] The alternate model is intentionally exposed, or removed from user-facing options.
- [ ] The Issue template model dropdown matches the allowed model ids.
- [ ] The runner has enough resources for the configured model.
- [ ] Required model credentials, if any, are available as repository secrets.

## 4. Documentation consistency

- [ ] `README.md` links to `docs/user-manual.md`.
- [ ] `docs/README.md` links to `docs/user-manual.md`.
- [ ] `docs/user-manual.md` matches the current Issue template fields.
- [ ] `docs/user-manual.md` lists the current model options.
- [ ] `docs/template-build-and-use.md` matches `assetpack.yml` and the active workflow.
- [ ] `docs/operations-guide.md` describes the actual failure and duplicate-request behavior.

## 5. Smoke test

Submit one valid smoke Issue.

Expected result:

- [ ] The workflow starts after the `asset-request` label is present.
- [ ] Validation passes.
- [ ] One image is generated.
- [ ] A directory is committed under `assets/generated/issue-<number>/<recipe_id>/`.
- [ ] The directory contains `image.png`, `prompt.txt`, `negative_prompt.txt`, `request.json`, `metadata.json`, `report.json`, and `README.md`.
- [ ] The Issue receives a success comment with the committed asset path.
- [ ] `assets/generated/README.md` includes the new asset.

## 6. Duplicate test

Re-submit or re-trigger the same structured request.

Expected result:

- [ ] The workflow detects the existing recipe id.
- [ ] No duplicate image is generated.
- [ ] The Issue receives a comment pointing to the existing asset path.

## 7. Failure test

Submit or edit one intentionally invalid request.

Expected result:

- [ ] Validation fails before image generation.
- [ ] The Issue receives a clear failure comment.
- [ ] No partial generated asset directory is committed.

## 8. Public-use decision

Do not advertise the repository to external users until all of these are true:

- [ ] At least one valid smoke Issue has succeeded.
- [ ] Duplicate handling has been observed.
- [ ] One invalid request has failed safely.
- [ ] The user manual matches the live Issue template.
- [ ] The repository owner accepts the model cost, runtime, and content-policy risk.

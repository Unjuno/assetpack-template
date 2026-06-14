# Dry-run validation

Use the validation scripts locally before relying on GitHub Actions.

## Minimal flow

```bash
python scripts/validate_issue_request.py \
  --config assetpack.yml \
  --issue-body-file /path/to/issue-body.md \
  --issue-number 0 \
  --out-dir /tmp/assetpack-issue-dry-run

python scripts/validate_issue_policy.py \
  --config assetpack.yml \
  --request-json /tmp/assetpack-issue-dry-run/request.json \
  --comment-file /tmp/assetpack-issue-dry-run/validation-comment.md
```

Inspect:

```text
/tmp/assetpack-issue-dry-run/request.json
/tmp/assetpack-issue-dry-run/validation-comment.md
```

This dry run does not install image-generation dependencies and does not create a PNG.

# Smoke-test policy

Smoke-test Issues are used to verify the template workflow.

## Smoke-test categories

- Valid generation smoke: should generate and commit a prompt/image record.
- ASCII rejection smoke: should fail validation and leave no generated asset.
- Duplicate recipe smoke: should fail validation when the same `recipe_id` already exists.

## Cleanup policy

- Smoke-test Issues should be clearly titled with `Workflow smoke:`.
- Smoke-test generated assets may remain temporarily as examples.
- Before template release, decide whether smoke assets should be retained as examples or removed.
- If retained, document them as examples rather than production corpus entries.

## Current known smoke asset

```text
assets/generated/issue-000034/assetpack-1261d48d85f2711b/
```

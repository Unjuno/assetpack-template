# Issue-driven workflow

GitHub Issues provide structured requests.

```text
Issue -> label gate -> validation -> fixed recipe -> assets/generated -> Issue comment
```

Validation rejects missing fields, non-ASCII fields, invalid licenses, invalid models, URLs, unknown sections, and missing required terms.

## Repeated Issue events

The workflow listens to `opened`, `edited`, `reopened`, and `labeled` events. A single request can therefore be evaluated more than once.

If the same structured request resolves to a `recipe_id` that already exists under `assets/generated/`, policy validation treats the event as an existing asset status instead of starting another generation run. The Issue receives a comment pointing at the existing asset directory.

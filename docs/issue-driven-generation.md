# Issue-driven workflow

GitHub Issues provide structured requests.

```text
Issue -> request label -> validation -> fixed recipe -> assets/generated -> Issue comment
```

Validation rejects missing fields, non-ASCII fields, invalid licenses, invalid models, URLs, unknown sections, and missing required terms in the final prompt.

## Trigger events

The workflow listens to `labeled`, `edited`, and `reopened` Issue events.

The request label, usually `asset-request`, is the normal generation trigger. `edited` and `reopened` allow maintainers to retry after correcting a request.

The workflow intentionally does not listen to `opened` events. This avoids duplicate runs when an Issue template creates an Issue with labels already attached.

## Repeated Issue events

A single request can still be evaluated more than once if it is edited, reopened, or re-labeled.

If the same structured request resolves to a `recipe_id` that already exists under `assets/generated/`, policy validation treats the event as an existing asset status instead of starting another generation run. The Issue receives a comment pointing at the existing asset directory.

# Example Issues

Use these examples to verify request formatting.

## Expected to pass

Apply the configured request label, usually `asset-request`.

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

Expected behavior:

- Validation passes.
- The final prompt includes the required terms configured in `assetpack.yml`.
- The workflow attempts image generation.
- On success, a new record is committed under `assets/generated/`.

## Expected to fail ASCII-only validation

Apply the configured request label, then place a non-ASCII character in one of the checked fields such as Subject, Scene, Audience, or Constraints.

Expected behavior:

- Validation fails before generation.
- No image is generated.
- No files are committed.
- The Issue receives a policy rejection comment.

## Expected duplicate behavior

Submit the same valid request after its asset record already exists.

Expected behavior:

- Policy validation detects the existing `recipe_id` under `assets/generated/`.
- No new generation run starts.
- The Issue receives a comment pointing at the existing asset directory.

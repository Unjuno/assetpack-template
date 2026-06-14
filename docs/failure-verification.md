# Failure verification

Failure verification uses Issue smoke tests.

## ASCII-only rejection

Create an Issue with a non-ASCII character in a structured text field.

Expected result:

- workflow posts a validation failure comment;
- no image is generated;
- no committed record is created.

## Duplicate recipe rejection

Create an Issue with the same structured fields as an existing generated asset.

Expected result:

- workflow posts a duplicate recipe validation failure comment;
- no image is generated;
- no committed record is created.

## Missing required terms

This is covered by `tests/test_issue_policy.py` because normal Issue requests mechanically inject configured required terms.

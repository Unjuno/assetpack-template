# Corpus

This directory is for structured text inputs used by prompt recipe generation.

A derived repository may keep approved subjects, scenes, modifiers, styles, and other controlled terms here.

## Example files

```text
subjects.jsonl
scenes.jsonl
modifiers.jsonl
styles.jsonl
```

JSON Lines is a good default because it is easy to diff and validate.

## Design rule

Corpus files should support mechanical prompt generation. They are not a place for unreviewed one-off prompt text.

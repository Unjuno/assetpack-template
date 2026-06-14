# Workflow syntax guard

GitHub Actions validates workflow files on push.

Known rule from this template cleanup:

- Do not use `runner.*` context in job-level `env`.
- Prefer runtime extraction from `GITHUB_EVENT_PATH` for Issue-specific data.
- Keep the Issue generation workflow triggered only by `issues` events.
- Keep artifact names simple and avoid complex fallback expressions.

Current Issue workflow:

```text
.github/workflows/assetpack-issue-generate.yml
```

Before changing the workflow, check for these risky patterns:

```bash
grep -n "runner\." .github/workflows/*.yml
grep -n "||" .github/workflows/*.yml
```

If either appears in a workflow expression, review it before committing.

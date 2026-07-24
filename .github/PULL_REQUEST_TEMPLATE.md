## Summary

Briefly describe what this pull request changes and why. Link any related issues
(e.g. `Closes #123`).

## Changes

- <what you changed, at a high level>
-

## Tests

Describe how the change is covered and how you verified it.

- [ ] Added or updated tests for the change
- [ ] `pytest -q` passes locally
- [ ] `ruff check src tests` is clean

## Checklist

- [ ] The change preserves the public data contract `(unique_id, ds, y)` and is
      backward compatible (existing tests pass unmodified).
- [ ] No new required runtime dependency was added (core stays numpy/pandas/scipy;
      heavy backends remain optional extras).
- [ ] Public APIs have docstrings and any user-facing behavior change is documented.
- [ ] Commits are scoped and the PR description explains the motivation.

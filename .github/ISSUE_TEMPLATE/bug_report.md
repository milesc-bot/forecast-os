---
name: Bug report
about: Report a defect so we can reproduce and fix it
title: "[Bug] "
labels: bug
assignees: ""
---

## Summary

A clear, one-sentence description of the bug.

## Steps to reproduce

A minimal, self-contained snippet that triggers the problem. Please use
synthetic data (e.g. `forecast_os.generate_series(...)`) so we can run it as-is.

```python
import forecast_os as f

# ... minimal reproduction ...
```

## Expected behavior

What you expected to happen.

## Actual behavior

What actually happened. Include the full traceback if an exception was raised.

```
<paste traceback / output here>
```

## Version

- forecast-os version: <output of `python -c "import forecast_os; print(forecast_os.__version__)"`>
- Installed with extras (if any): <e.g. none / `[dev]` / `[nixtla]`>

## Environment

- Python version: <output of `python --version`>
- Operating system: <e.g. macOS 14.5 / Ubuntu 22.04 / Windows 11>
- numpy / pandas / scipy versions: <output of `pip show numpy pandas scipy | grep -E "Name|Version"`>

## Additional context

Anything else that might help — related issues, workarounds you tried, or notes
on how often it reproduces.

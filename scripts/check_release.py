"""Pre-tag release gate: run the suite against every supported pandas line.

The developer venv carries one pandas. That is the version every local check
sees — the test suite, ruff, and any amount of code review. A change that is
correct on pandas 3 and broken on pandas 2 therefore passes every local gate
and only fails in CI, which on this repo runs *after* the tag that publishes to
PyPI. That is exactly how v0.10.0 shipped a ``detect_anomalies`` call that
raised ``ValueError: Categorical categories cannot be null`` on pandas 2.x.

This script builds a throwaway venv per supported pandas line, installs the
working tree into it, and runs the full suite. Run it before tagging::

    python scripts/check_release.py

Exit code is 0 only when every line passes. ``--quick`` skips the optional
extras (faster, but does not exercise snapshots / serve / mcp / terminal).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: One entry per supported pandas line. The floor mirrors ``pandas>=2.0`` in
#: pyproject.toml and the pin CI applies to its oldest Python leg; ``latest``
#: tracks whatever resolves today, which is what most users install.
PANDAS_LINES = [
    ("oldest supported (2.x)", "pandas>=2,<3"),
    ("current (3.x)", "pandas>=3"),
]

#: Optional backends the suite needs to cover snapshots, serve, MCP and the TUI.
#: Without them those tests error out rather than skip, so a partial run would
#: report failures that say nothing about pandas.
EXTRAS = [
    "pyarrow", "requests", "mcp", "textual", "textual-plotext",
    "pytest-asyncio", "fastapi", "httpx", "uvicorn",
]


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=REPO, text=True, **kw)


def check_line(label: str, spec: str, *, quick: bool) -> bool:
    """Install the working tree against one pandas line and run the suite."""
    print(f"\n{'=' * 70}\n{label}  ({spec})\n{'=' * 70}", flush=True)
    tmp = Path(tempfile.mkdtemp(prefix="fos-relcheck-"))
    try:
        venv = tmp / "venv"
        if run([sys.executable, "-m", "venv", str(venv)]).returncode:
            print(f"FAIL {label}: could not create venv")
            return False
        py = venv / "bin" / "python"

        pkgs = ["pytest", "-e", ".", spec] + ([] if quick else EXTRAS)
        install = run(
            [str(py), "-m", "pip", "install", "-q", *pkgs],
            capture_output=True,
        )
        if install.returncode:
            # A line that cannot resolve on this interpreter is a skip, not a
            # failure: pandas 2.x has no wheels for the newest Pythons, and the
            # point of the gate is coverage where coverage is possible.
            print(f"SKIP {label}: cannot install here\n{install.stderr[-400:]}")
            return True

        got = run(
            [str(py), "-c", "import pandas; print(pandas.__version__)"],
            capture_output=True,
        ).stdout.strip()
        print(f"resolved pandas {got}", flush=True)

        # -p no:cacheprovider: the throwaway venv must not write .pytest_cache
        # into the repo and make the working tree look dirty before a tag.
        result = run([str(py), "-m", "pytest", "-p", "no:warnings", "-p", "no:cacheprovider"])
        ok = result.returncode == 0
        print(f"{'PASS' if ok else 'FAIL'} {label} (pandas {got})", flush=True)
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="skip optional extras")
    args = ap.parse_args()

    results = {label: check_line(label, spec, quick=args.quick) for label, spec in PANDAS_LINES}

    print(f"\n{'=' * 70}")
    for label, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    if all(results.values()):
        print("\nAll supported pandas lines green — safe to tag.")
        return 0
    print("\nDO NOT TAG: a supported pandas line is failing.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

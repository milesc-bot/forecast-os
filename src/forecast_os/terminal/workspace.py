"""The persisted terminal workspace: sources, watchlist, settings, alerts.

A :class:`Workspace` is the whole state of the always-on console, stored as
``workspace.json`` under a home directory (``$FORECAST_OS_HOME`` or
``~/.forecast-os`` by default)::

    ws = Workspace.load()            # defaults when no file exists yet
    ws.settings["h"] = 12
    ws.save()                        # atomic write (tmp + rename)

Loading is tolerant: missing keys fall back to defaults, so a workspace file
written by an older version keeps working. This module imports no textual —
it is plain stdlib and usable from any front end.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from ..core.exceptions import ForecastOSError

__all__ = ["Workspace", "default_settings"]

#: Environment variable overriding the default workspace home directory.
HOME_ENV_VAR = "FORECAST_OS_HOME"

#: Name of the workspace file inside the home directory.
WORKSPACE_FILE = "workspace.json"

_DEFAULT_SETTINGS: dict = {
    "model": "auto_ets",
    "h": 6,
    "level": 80,
    "refresh_seconds": 0,  # 0 = manual refresh only
    "season_length": None,
}

_SOURCE_DEFAULTS: dict = {"path": None, "mapping": None, "overrides": {}}

#: How each known setting is consumed downstream: ``(kind, may be null)``.
#: The console reads ``model`` as a registry name and coerces the rest with
#: ``int()``, so a value that cannot do that kills the app during mount.
#: Unlisted keys are left alone, so a newer version's settings still load.
_SETTING_SPECS: dict[str, tuple[type, bool]] = {
    "model": (str, False),
    "h": (int, False),
    "level": (int, False),
    "refresh_seconds": (int, True),  # null and 0 both mean "manual refresh only"
    "season_length": (int, True),  # null means "no seasonality"
}


def default_settings() -> dict:
    """A fresh copy of the default console settings."""
    return dict(_DEFAULT_SETTINGS)


def _require(raw: dict, key: str, kind: type, target: Path):
    """Fetch ``raw[key]`` (or ``None``) after checking it is a ``kind``."""
    value = raw.get(key)
    if value is not None and not isinstance(value, kind):
        raise ForecastOSError(
            f"workspace file {target}: field {key!r} must be a "
            f"{kind.__name__}, got {type(value).__name__}"
        )
    return value


def _require_dicts(items, key: str, target: Path) -> list[dict]:
    """Validate every element of a list field is a JSON object."""
    out = []
    for i, item in enumerate(items or []):
        if not isinstance(item, dict):
            raise ForecastOSError(
                f"workspace file {target}: {key}[{i}] must be a JSON object, "
                f"got {type(item).__name__}"
            )
        out.append(item)
    return out


def _require_usable_settings(settings: dict, target: Path) -> dict:
    """Check the known settings hold values the console can actually use.

    Type-checking the ``settings`` container alone was not enough: the values
    were merged verbatim, so a hand-edited ``"refresh_seconds": "every 5
    minutes"`` loaded "successfully" and then killed the app during mount
    with a bare ``ValueError`` from ``int()``. Raising here instead lets
    ``terminal.app.main`` report it as a clean one-line exit, the same way it
    already does for a wrong-typed ``sources``/``watch``/``alerts``.

    Deliberately permissive: anything ``int()`` accepts (including ``"12"``
    and ``80.0``) passes, nulls pass where the default is nullable, and
    unknown keys are not inspected at all. The one deliberate exception is
    ``bool``: it is an ``int`` subclass, so ``{"h": true}`` would coerce to a
    1-period horizon and ``{"level": true}`` to a 1% confidence level — values
    no one writing JSON booleans could have meant, so they are refused rather
    than silently accepted.
    """
    for key, (kind, nullable) in _SETTING_SPECS.items():
        if key not in settings:
            continue
        value = settings[key]
        if value is None and nullable:
            continue
        if kind is str:
            ok = isinstance(value, str)
        else:
            ok = not isinstance(value, bool)
            if ok:
                try:
                    int(value)
                except (TypeError, ValueError):
                    ok = False
        if not ok:
            expected = "a string" if kind is str else "an integer"
            raise ForecastOSError(
                f"workspace file {target}: setting {key!r} must be "
                f"{expected}, got {value!r}"
            )
    return settings


@dataclass
class Workspace:
    """The console state: data sources, watchlist, settings, and alert rules.

    ``sources`` holds one dict per CSV source: ``{"path": ..., "mapping":
    <registered mapping name or None>, "overrides": {...}}`` (overrides are
    passed to the mapping at apply time). ``watch`` pins ``unique_id``\\ s to
    the dashboard (empty = show all). ``settings`` drives the compute layer
    (model, horizon, confidence level, refresh cadence, season length).
    ``alerts`` holds rules like ``{"kind": "forecast_below", "series": "*",
    "threshold": 100.0}`` — kinds are ``forecast_below`` and
    ``coverage_below``, with ``series`` either one ``unique_id`` or ``"*"``.
    ``home`` is the directory the workspace persists to.
    """

    sources: list[dict] = field(default_factory=list)
    watch: list[str] = field(default_factory=list)
    settings: dict = field(default_factory=default_settings)
    alerts: list[dict] = field(default_factory=list)
    home: Path | None = None

    @staticmethod
    def resolve_home(home: str | os.PathLike | None = None) -> Path:
        """The workspace home: ``home`` arg, ``$FORECAST_OS_HOME``, or ``~/.forecast-os``."""
        if home is None:
            home = os.environ.get(HOME_ENV_VAR) or Path.home() / ".forecast-os"
        return Path(home).expanduser()

    @property
    def path(self) -> Path:
        """The workspace file path (``<home>/workspace.json``)."""
        if self.home is None:
            raise ForecastOSError(
                "workspace has no home directory; construct with home=... "
                "or use Workspace.load()"
            )
        return Path(self.home) / WORKSPACE_FILE

    @classmethod
    def load(cls, home: str | os.PathLike | None = None) -> Workspace:
        """Read the workspace under ``home`` (defaults when no file exists).

        Missing keys in the file are merged with defaults, so partial or
        older workspace files load cleanly. A file that is not valid JSON
        (or not a JSON object) raises :class:`ForecastOSError` naming it, as
        does a known setting holding a value the console cannot use (see
        :func:`_require_usable_settings`) — better a named error here than a
        traceback out of the app's mount.
        """
        resolved = cls.resolve_home(home)
        ws = cls(home=resolved)
        target = resolved / WORKSPACE_FILE
        if not target.exists():
            return ws
        try:
            raw = json.loads(target.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            raise ForecastOSError(f"cannot read workspace file {target}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ForecastOSError(
                f"workspace file {target} must hold a JSON object, "
                f"got {type(raw).__name__}"
            )
        sources = _require_dicts(_require(raw, "sources", list, target), "sources", target)
        watch = _require(raw, "watch", list, target)
        settings = _require(raw, "settings", dict, target)
        alerts = _require_dicts(_require(raw, "alerts", list, target), "alerts", target)
        ws.sources = [
            {**_SOURCE_DEFAULTS, **src, "overrides": dict(src.get("overrides") or {})}
            for src in sources
        ]
        ws.watch = [str(uid) for uid in watch or []]
        ws.settings = _require_usable_settings(
            {**_DEFAULT_SETTINGS, **(settings or {})}, target
        )
        ws.alerts = [dict(alert) for alert in alerts]
        return ws

    def save(self) -> Path:
        """Write ``workspace.json`` atomically (tmp file + rename); return its path."""
        target = self.path  # raises when home is unset
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "sources": self.sources,
            "watch": self.watch,
            "settings": self.settings,
            "alerts": self.alerts,
        }
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, target)
        return target

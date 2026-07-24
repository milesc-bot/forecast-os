"""Point-in-time snapshot store: parquet files plus a JSON manifest index.

A :class:`SnapshotStore` is a directory on disk. Each call to
:meth:`~SnapshotStore.snapshot` freezes a frame — a validated panel or a
committed forecast — as an ``<as_of>.parquet`` file under a per-kind
sub-directory and appends one row to ``manifest.json``. Snapshots are
append-only: re-snapshotting the same ``(as_of, kind)`` writes a
numerically suffixed sibling file and the manifest keeps every version, so
the store is a faithful week-over-week and forecast-accuracy audit trail.

Reading and writing parquet needs a pandas parquet engine; pyarrow is the
optional ``[snapshots]`` extra. Importing this module always succeeds; the
guard fires at :class:`SnapshotStore` construction, raising ``ImportError``
with the install hint when pyarrow is missing.
"""

from __future__ import annotations

import json
from os import PathLike
from pathlib import Path
from typing import Any

import pandas as pd

from ..core.exceptions import DataContractError, ForecastOSError
from ..core.types import ID_COL, TIME_COL, validate_panel

try:
    import pyarrow  # noqa: F401

    _HAS_PYARROW = True
except ImportError:  # pragma: no cover - pyarrow is installed in dev
    _HAS_PYARROW = False

__all__ = ["SnapshotStore"]

_SNAPSHOTS_HINT = (
    "the snapshot store needs the optional 'pyarrow' dependency; install it "
    'with pip install "forecast-os[snapshots]"'
)

#: Frame kinds the store understands and how each is validated.
_KINDS = ("panel", "forecast", "deals")

#: Columns a ``forecast`` snapshot must carry (the committed forecast shape).
_FORECAST_COLS = (ID_COL, TIME_COL, "yhat")

#: Columns a ``deals`` snapshot must carry (the opportunity contract). A deal
#: table is DEAL-GRAIN (one row per opportunity), not the (unique_id, ds, y)
#: panel, so it is stored verbatim rather than normalized.
_DEALS_COLS = ("opp_id", "amount", "stage")

_MANIFEST_NAME = "manifest.json"

#: Manifest column order for :meth:`SnapshotStore.manifest`.
_MANIFEST_COLS = ["as_of", "kind", "label", "file", "rows", "created_at"]


class SnapshotStore:
    """An append-only, point-in-time store of panel and forecast snapshots.

    ``path`` is the store directory (created if absent); it holds one
    ``<kind>/<as_of>.parquet`` file per snapshot and a ``manifest.json``
    index. Construction requires the optional ``pyarrow`` dependency and
    raises ``ImportError`` with the ``[snapshots]`` install hint without it.
    """

    def __init__(self, path: str | PathLike):
        if not _HAS_PYARROW:
            raise ImportError(_SNAPSHOTS_HINT)
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    # -- internals -----------------------------------------------------------

    @property
    def _manifest_path(self) -> Path:
        return self.path / _MANIFEST_NAME

    def _read_manifest(self) -> list[dict[str, Any]]:
        """The manifest entries in append order (empty list for a fresh store)."""
        if not self._manifest_path.exists():
            return []
        try:
            with open(self._manifest_path, encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as exc:
            raise ForecastOSError(
                f"corrupt snapshot manifest at {self._manifest_path}: {exc}"
            ) from exc

    @staticmethod
    def _restore_ds_unit(frame: pd.DataFrame, entry: dict[str, Any]) -> pd.DataFrame:
        """Restore the datetime ``ds`` resolution recorded at write time.

        Parquet promotes second-resolution timestamps to milliseconds, so the
        original unit is stored in the manifest and re-applied on load, keeping
        round-trips dtype-stable.
        """
        unit = entry.get("ds_unit")
        if (
            unit
            and TIME_COL in frame.columns
            and pd.api.types.is_datetime64_any_dtype(frame[TIME_COL])
        ):
            frame[TIME_COL] = frame[TIME_COL].astype(f"datetime64[{unit}]")
        return frame

    def _write_manifest(self, entries: list[dict[str, Any]]) -> None:
        with open(self._manifest_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)

    @staticmethod
    def _validate_frame(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
        """Validate ``frame`` for ``kind`` and return the frame to persist.

        ``panel`` kinds are run through :func:`validate_panel` (and the
        normalized copy is stored); ``forecast`` kinds must carry the
        ``(unique_id, ds, yhat)`` columns and ``deals`` kinds the deal-grain
        ``(opp_id, amount, stage)`` opportunity contract. Both non-panel kinds
        are stored as-is (a copy), preserving any extra feature/segment columns.
        """
        if kind == "panel":
            return validate_panel(frame)
        if not isinstance(frame, pd.DataFrame):
            raise DataContractError(
                f"expected a pandas DataFrame, got {type(frame).__name__}"
            )
        if kind == "deals":
            missing = [c for c in _DEALS_COLS if c not in frame.columns]
            if missing:
                raise DataContractError(
                    f"deals snapshot is missing required column(s) {missing}; the "
                    f"opportunity contract is (opp_id, amount, stage)"
                )
            if len(frame) == 0:
                raise DataContractError("deals snapshot is empty")
            return frame.copy()
        missing = [c for c in _FORECAST_COLS if c not in frame.columns]
        if missing:
            raise DataContractError(
                f"forecast snapshot is missing required column(s) {missing}; the "
                f"contract is (unique_id, ds, yhat)"
            )
        if len(frame) == 0:
            raise DataContractError("forecast snapshot is empty")
        return frame.copy()

    @staticmethod
    def _coerce_created_at(created_at: Any, as_of_ts: pd.Timestamp) -> str:
        """The ISO ``created_at`` string: caller-supplied, else derived from ``as_of``.

        ``datetime.now`` is never called (snapshots must be deterministic);
        the default is the ISO form of the normalized ``as_of`` timestamp.
        """
        if created_at is None:
            return as_of_ts.isoformat()
        if isinstance(created_at, str):
            return created_at
        return pd.Timestamp(created_at).isoformat()

    # -- writing -------------------------------------------------------------

    def snapshot(
        self,
        frame: pd.DataFrame,
        as_of: Any,
        kind: str = "panel",
        label: str | None = None,
        created_at: Any = None,
    ) -> dict[str, Any]:
        """Freeze ``frame`` as the ``as_of`` snapshot of ``kind`` and index it.

        ``panel`` frames must pass :func:`~forecast_os.core.types.validate_panel`
        (the normalized copy is stored); ``forecast`` frames must carry
        ``(unique_id, ds, yhat)`` and ``deals`` frames the deal-grain
        ``(opp_id, amount, stage)`` opportunity contract (stored verbatim).
        ``as_of`` is coerced to a midnight-normalized
        :class:`pandas.Timestamp`; the frame is written to
        ``<path>/<kind>/<as_of:%Y%m%d>[-<n>].parquet`` (a numeric suffix keeps
        earlier same-day snapshots) and a manifest row is appended.

        ``created_at`` is an ISO timestamp string; when omitted it is derived
        deterministically from ``as_of`` (never ``datetime.now``). Returns the
        appended manifest entry.
        """
        if kind not in _KINDS:
            raise ForecastOSError(
                f"unknown snapshot kind {kind!r}; expected one of {_KINDS}"
            )
        to_write = self._validate_frame(frame, kind)
        as_of_ts = pd.Timestamp(as_of).normalize()
        as_of_iso = as_of_ts.isoformat()
        created_at_iso = self._coerce_created_at(created_at, as_of_ts)

        entries = self._read_manifest()
        kind_dir = self.path / kind
        kind_dir.mkdir(parents=True, exist_ok=True)
        stem = as_of_ts.strftime("%Y%m%d")
        n = sum(1 for e in entries if e["kind"] == kind and e["as_of"] == as_of_iso)
        name = f"{stem}.parquet" if n == 0 else f"{stem}-{n}.parquet"
        file_path = kind_dir / name
        while file_path.exists():  # never overwrite an existing snapshot file
            n += 1
            name = f"{stem}-{n}.parquet"
            file_path = kind_dir / name

        to_write.to_parquet(file_path, index=False)

        ds_unit = None
        if TIME_COL in to_write.columns and pd.api.types.is_datetime64_any_dtype(
            to_write[TIME_COL]
        ):
            ds_unit = to_write[TIME_COL].dt.unit

        entry = {
            "as_of": as_of_iso,
            "kind": kind,
            "label": label,
            "file": f"{kind}/{name}",
            "rows": int(len(to_write)),
            "created_at": created_at_iso,
            "ds_unit": ds_unit,
        }
        entries.append(entry)
        self._write_manifest(entries)
        return entry

    # -- reading -------------------------------------------------------------

    def as_of_dates(self, kind: str = "panel") -> list[pd.Timestamp]:
        """The sorted unique ``as_of`` timestamps recorded for ``kind``."""
        seen = {
            pd.Timestamp(e["as_of"])
            for e in self._read_manifest()
            if e["kind"] == kind
        }
        return sorted(seen)

    def load(self, as_of: Any = None, kind: str = "panel") -> pd.DataFrame:
        """Load the frame for ``as_of`` (the latest same-day snapshot if several).

        ``as_of=None`` loads the most recent ``as_of`` recorded for ``kind``.
        Raises :class:`ForecastOSError` when the store holds no matching
        snapshot.
        """
        entries = [e for e in self._read_manifest() if e["kind"] == kind]
        if not entries:
            raise ForecastOSError(f"no {kind} snapshots in {str(self.path)!r}")
        if as_of is None:
            target = max(pd.Timestamp(e["as_of"]) for e in entries)
        else:
            target = pd.Timestamp(as_of).normalize()
        target_iso = target.isoformat()
        matches = [e for e in entries if e["as_of"] == target_iso]
        if not matches:
            raise ForecastOSError(
                f"no {kind} snapshot for as_of {target.date()} in {str(self.path)!r}"
            )
        # manifest is append-ordered, so the last match is the latest snapshot
        return self._restore_ds_unit(
            pd.read_parquet(self.path / matches[-1]["file"]), matches[-1]
        )

    def history(
        self, kind: str = "panel", series: str | list[str] | None = None
    ) -> pd.DataFrame:
        """Every ``kind`` snapshot stacked, with an added ``as_of`` column.

        Optionally filtered to the ``unique_id`` value(s) in ``series``; the
        result is sorted by whichever of ``(as_of, unique_id, ds)`` the frames
        carry — deal-grain ``deals`` snapshots have neither ``unique_id`` nor
        ``ds`` and are sorted by ``as_of`` alone. Returns an empty frame when
        the store holds no matching snapshots.
        """
        entries = [e for e in self._read_manifest() if e["kind"] == kind]
        if not entries:
            return pd.DataFrame()
        if series is not None and isinstance(series, str):
            series = [series]
        frames = []
        for e in entries:
            frame = self._restore_ds_unit(pd.read_parquet(self.path / e["file"]), e)
            frame = frame.copy()
            frame["as_of"] = pd.Timestamp(e["as_of"])
            if series is not None:
                frame = frame[frame[ID_COL].isin(series)]
            frames.append(frame)
        stacked = pd.concat(frames, ignore_index=True)
        sort_cols = [c for c in ("as_of", ID_COL, TIME_COL) if c in stacked.columns]
        return stacked.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    def manifest(self) -> pd.DataFrame:
        """The snapshot index as a DataFrame (``as_of`` parsed to datetime)."""
        entries = self._read_manifest()
        if not entries:
            return pd.DataFrame(columns=_MANIFEST_COLS)
        df = pd.DataFrame(entries)
        df["as_of"] = pd.to_datetime(df["as_of"])
        return df[_MANIFEST_COLS]

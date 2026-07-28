"""Driver-based scenario planning for go-to-market bookings.

:class:`Scenario` is a thin, functional what-if wrapper over a flat dictionary
of named business drivers (top-of-funnel volume, conversion rates, average
contract value, headcount, ...). It does not forecast; it *composes* a bookings
number from drivers the caller already believes, reusing
:func:`forecast_os.gtm.funnel.propagate` for the funnel chain, so a revenue
leader can ask "what if win-rate slips five points?" and read the delta.

Projection model
----------------
The default projection (:meth:`Scenario.project` with no ``stages`` chain) is
the canonical *leads x rate x deal-size* identity::

    bookings = top_of_funnel * prod(rate drivers) * acv

where the *rate drivers* are every driver whose name ends in ``_rate`` (so
``win_rate``, ``sql_rate``, ... all multiply in, and each must be non-negative —
a driver bumped below zero raises, but one above 1.0 is allowed and multiplies
in as-is, since ``expansion_rate``/``attach_rate``-style drivers are genuine
multipliers rather than probabilities), ``acv`` (average contract
value) turns a deal count into currency and defaults to ``1.0`` when absent,
and any other driver (e.g. ``rep_count``) is carried for the caller's own use
but does not enter the default bookings number. ``top_of_funnel`` is required.

Supplying a ``stages`` chain instead runs the volume through
:func:`propagate` stage by stage — the per-stage deal volumes are exactly what
:func:`propagate` produces — and applies ``acv`` to the final stage.

Scenarios are immutable: :meth:`Scenario.with_` (absolute set) and
:meth:`Scenario.bump` (relative delta) both return a *new* Scenario, leaving
the original untouched, so a baseline can spawn a best/base/worst family.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import pandas as pd

from ..core.exceptions import DataContractError, ForecastOSError
from ..core.types import ID_COL, TIME_COL
from .funnel import propagate

__all__ = ["Scenario", "compare_scenarios"]

#: Driver key holding the volume entering the top of the funnel.
TOP_OF_FUNNEL = "top_of_funnel"
#: Driver key holding the average contract value (deal size).
ACV = "acv"
#: Suffix that marks a driver as a multiplicative conversion rate.
RATE_SUFFIX = "_rate"


def _is_real_number(v: object) -> bool:
    # bool is a subclass of int but is never a valid driver value.
    return isinstance(v, (int, float)) and not isinstance(v, bool)


class Scenario:
    """A driver-based what-if over the GTM funnel/quota primitives.

    Parameters
    ----------
    baseline_drivers : mapping of ``name -> number``
        Named business drivers, e.g. ``{"top_of_funnel": 1000, "win_rate":
        0.30, "acv": 25000, "rep_count": 10}``. Names must be strings and
        values finite real numbers (``bool`` is rejected). Stored on the
        same-named attribute ``baseline_drivers``; read a defensive copy via
        the :attr:`drivers` property.
    """

    def __init__(self, baseline_drivers: Mapping[str, float]) -> None:
        if not isinstance(baseline_drivers, Mapping):
            raise DataContractError(
                "baseline_drivers must be a mapping of name -> number, got "
                f"{type(baseline_drivers).__name__}"
            )
        validated: dict[str, float] = {}
        for key, value in baseline_drivers.items():
            if not isinstance(key, str):
                raise DataContractError(f"driver names must be strings, got {key!r}")
            if not _is_real_number(value):
                raise DataContractError(
                    f"driver {key!r} must be a real number, got {type(value).__name__}"
                )
            fv = float(value)
            if not math.isfinite(fv):
                raise DataContractError(f"driver {key!r} must be finite, got {value!r}")
            validated[key] = fv
        self.baseline_drivers: dict[str, float] = validated

    # -- accessors -----------------------------------------------------------

    @property
    def drivers(self) -> dict[str, float]:
        """A defensive copy of the driver dict (mutating it is a no-op)."""
        return dict(self.baseline_drivers)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Scenario({self.baseline_drivers!r})"

    # -- what-if constructors ------------------------------------------------

    def _check_known(self, deltas: Mapping[str, float]) -> None:
        unknown = [k for k in deltas if k not in self.baseline_drivers]
        if unknown:
            raise ForecastOSError(
                f"unknown driver(s) {unknown}; with_/bump only modify existing "
                f"drivers (guards against typos). Known drivers: "
                f"{sorted(self.baseline_drivers)}"
            )

    def with_(self, **deltas: float) -> Scenario:
        """A new Scenario with the named drivers *set* to new absolute values.

        ``baseline.with_(win_rate=0.25)`` returns a copy whose ``win_rate`` is
        ``0.25`` regardless of its previous value. Only existing drivers may be
        set — an unknown name raises :class:`ForecastOSError` to catch typos.
        """
        self._check_known(deltas)
        updated = dict(self.baseline_drivers)
        updated.update(deltas)
        return Scenario(updated)

    def bump(self, **deltas: float) -> Scenario:
        """A new Scenario with the named drivers shifted by *additive* deltas.

        ``baseline.bump(win_rate=-0.05)`` subtracts 0.05 from ``win_rate``;
        ``baseline.bump(top_of_funnel=200)`` adds 200 leads. Deltas are added
        (not multiplied); use :meth:`with_` for an absolute set. Only existing
        drivers may be bumped — an unknown name raises :class:`ForecastOSError`.
        """
        self._check_known(deltas)
        updated = dict(self.baseline_drivers)
        for key, delta in deltas.items():
            if not _is_real_number(delta):
                raise DataContractError(
                    f"bump delta for {key!r} must be a real number, got "
                    f"{type(delta).__name__}"
                )
            updated[key] = self.baseline_drivers[key] + float(delta)
        return Scenario(updated)

    # -- projection ----------------------------------------------------------

    def _require_top(self) -> float:
        if TOP_OF_FUNNEL not in self.baseline_drivers:
            raise ForecastOSError(
                f"projection requires a {TOP_OF_FUNNEL!r} driver; drivers are "
                f"{sorted(self.baseline_drivers)}"
            )
        top = self.baseline_drivers[TOP_OF_FUNNEL]
        if top < 0:
            raise ForecastOSError(
                f"{TOP_OF_FUNNEL!r} must be non-negative, got {top}"
            )
        return top

    def _acv(self) -> float:
        acv = self.baseline_drivers.get(ACV, 1.0)
        if acv < 0:
            raise ForecastOSError(f"{ACV!r} must be non-negative, got {acv}")
        return acv

    def _rate_product(self) -> float:
        """Product of the ``*_rate`` drivers, each validated to be non-negative.

        The stages path validates every rate inside :func:`propagate`, so
        ``project({"won": "win_rate"})`` has always refused a win_rate of -0.1.
        The default path multiplied the same driver in unchecked, which turned
        one ``bump(win_rate=-0.20)`` past zero into a silently NEGATIVE bookings
        number (and a -200% pct_delta in :func:`compare_scenarios`). A negative
        multiplier has no meaning in the identity, so it is rejected here too.

        Only the lower bound is enforced. ``_rate`` is a *name* suffix, and it
        cannot distinguish a conversion probability from a multiplier:
        ``expansion_rate=1.15``, ``attach_rate=1.8`` and ``growth_rate=1.4`` are
        ordinary GTM drivers for which the identity is perfectly well defined.
        Bounding them at 1.0 would also be unescapable, since the ``_rate``
        suffix is the only thing that makes a driver multiply in at all — a
        rename to ``expansion_factor`` drops the factor from the projection
        rather than preserving it. Values above 1 therefore multiply in as-is.
        """
        product = 1.0
        for key, value in self.baseline_drivers.items():
            if not key.endswith(RATE_SUFFIX):
                continue
            if value < 0.0:
                raise ForecastOSError(
                    f"rate driver {key!r} must be non-negative, got {value}"
                )
            product *= value
        return product

    def stage_volumes(self, stages: Mapping[str, float | str]) -> pd.Series:
        """Per-stage deal volumes down a funnel chain, via :func:`propagate`.

        ``stages`` is an ordered mapping ``stage_name -> rate`` where each rate
        is either a literal float or the *name of a driver* holding the rate
        (so ``{"mql": "mql_rate"}`` reads ``mql_rate`` from the drivers and a
        :meth:`with_`/:meth:`bump` on that driver flows through). The starting
        volume is the ``top_of_funnel`` driver; the returned Series is indexed
        by stage name in chain order and holds implied deal counts identical to
        :func:`propagate`'s output (``acv`` is *not* applied here).
        """
        if not isinstance(stages, Mapping):
            raise ForecastOSError(
                "stages must be an ordered mapping of stage_name -> rate "
                f"(float or driver name), got {type(stages).__name__}"
            )
        if not stages:
            raise ValueError("stages is empty; provide at least one stage")
        top = self._require_top()
        rates: dict[str, float] = {}
        for stage, spec in stages.items():
            if isinstance(spec, str):
                if spec not in self.baseline_drivers:
                    raise ForecastOSError(
                        f"stage {stage!r} references unknown driver {spec!r}; "
                        f"known drivers: {sorted(self.baseline_drivers)}"
                    )
                rate = self.baseline_drivers[spec]
            elif _is_real_number(spec):
                rate = float(spec)
            else:
                raise ForecastOSError(
                    f"rate for stage {stage!r} must be a number or a driver "
                    f"name, got {type(spec).__name__}"
                )
            rates[str(stage)] = rate
        top_frame = pd.DataFrame({ID_COL: ["scenario"], TIME_COL: [0], "yhat": [top]})
        # propagate validates each rate is in [0, 1] and chains within-period.
        out = propagate(top_frame, rates)
        return pd.Series(
            {stage: float(out[stage].iloc[0]) for stage in rates}, dtype=float
        )

    def project(self, stages: Mapping[str, float | str] | None = None) -> float:
        """The projected bookings/pipeline number for this scenario.

        With ``stages=None`` (default) applies the *leads x rate x deal-size*
        model documented on the module: ``top_of_funnel * prod(rate drivers) *
        acv``. With a ``stages`` chain, runs the volume through
        :meth:`stage_volumes` and multiplies the final stage's deal count by
        ``acv``. Returns a plain float.

        Raises :class:`ForecastOSError` when ``top_of_funnel``, ``acv`` or a
        ``*_rate`` driver is negative; both projection paths reject negatives
        alike (the stages path via :func:`propagate`). The two paths differ on
        the upper end: a ``*_rate`` driver above 1.0 multiplies in normally on
        the default path (it may be a multiplier such as ``expansion_rate``),
        while the ``stages`` path caps every rate at 1.0 because a funnel chain
        cannot grow volume from one stage to the next.
        """
        top = self._require_top()
        acv = self._acv()
        if stages is None:
            return top * self._rate_product() * acv
        volumes = self.stage_volumes(stages)
        final = float(volumes.iloc[-1])
        return final * acv


def compare_scenarios(
    base: Scenario,
    *others: Scenario,
    labels: list[str] | None = None,
    stages: Mapping[str, float | str] | None = None,
) -> pd.DataFrame:
    """Baseline-vs-alternatives comparison table — the CRO's best/base/worst view.

    Projects ``base`` and each alternative in ``others`` (all via
    :meth:`Scenario.project`, optionally through a shared ``stages`` chain) and
    reports, one row per scenario, the projected number plus its ``delta`` and
    ``pct_delta`` against the baseline.

    Parameters
    ----------
    base, *others : Scenario
        The baseline scenario followed by any number of alternatives.
    labels : list of str, optional
        One label per scenario (``base`` first). Defaults to ``["baseline",
        "scenario_1", ...]``. Must match ``1 + len(others)`` when supplied.
    stages : mapping, optional
        A funnel chain forwarded to every :meth:`Scenario.project` call.

    Returns
    -------
    DataFrame with columns ``label, projection, delta, pct_delta``. ``delta``
    is ``projection - base_projection`` (0.0 on the baseline row); ``pct_delta``
    is that delta as a fraction of the baseline projection, ``NaN`` when the
    baseline projects ~0.
    """
    scenarios = [base, *others]
    for scenario in scenarios:
        if not isinstance(scenario, Scenario):
            raise ForecastOSError(
                f"compare_scenarios expects Scenario instances, got "
                f"{type(scenario).__name__}"
            )
    if labels is None:
        labels = ["baseline"] + [f"scenario_{i}" for i in range(1, len(others) + 1)]
    else:
        labels = list(labels)
        if len(labels) != len(scenarios):
            raise ValueError(
                f"labels must have {len(scenarios)} entries (baseline + "
                f"{len(others)} alternative(s)), got {len(labels)}"
            )

    base_proj = base.project(stages=stages)
    rows = []
    for label, scenario in zip(labels, scenarios):
        proj = scenario.project(stages=stages)
        delta = proj - base_proj
        pct = float("nan") if abs(base_proj) < 1e-12 else delta / base_proj
        rows.append(
            {"label": label, "projection": proj, "delta": delta, "pct_delta": pct}
        )
    return pd.DataFrame(rows, columns=["label", "projection", "delta", "pct_delta"])

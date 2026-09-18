"""Stage 0 arithmetic: required sample size and the written gate.

The proposal pre-registers a strict gate (Sec. 5.3): run the full C1 study only
if the sample size needed for 80% power is at most 1,000 documents *and* the
dispersion of position-specific noise is large enough for the question to have
content.  Otherwise C1 is restricted to the 110M model and compute shifts to C2.

The effect the study must be powered for is not the pilot point estimate alone.
Sec. 7 declares an effect only if the interval excludes zero *and* the estimate
exceeds twice the trace-level non-determinism floor, so the sample size is
computed against whichever bar is higher.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class PowerResult:
    n_required: int
    target_effect: float
    se_at_pilot: float
    n_pilot: int
    power: float
    alpha: float
    binding_bar: str      # "pilot_estimate" or "noise_floor"

    def __str__(self) -> str:
        return (f"n_required={self.n_required} docs for {self.power:.0%} power at "
                f"alpha={self.alpha} against effect {self.target_effect:+.4f} "
                f"(bar: {self.binding_bar})")


def required_documents(beta_hat: float, se_pilot: float, n_pilot: int,
                       floor_effect: float = 0.0, power: float = 0.80,
                       alpha: float = 0.05) -> PowerResult:
    """Documents needed for ``power`` against the binding effect bar.

    The clustered standard error scales as ``1/sqrt(n_docs)`` because documents
    are the independent unit, so the pilot SE extrapolates by that factor.
    """
    if se_pilot <= 0 or not np.isfinite(se_pilot):
        raise ValueError(f"pilot SE must be positive and finite, got {se_pilot}")
    if n_pilot <= 0:
        raise ValueError(f"n_pilot must be positive, got {n_pilot}")

    bar = 2.0 * abs(floor_effect)
    target = max(abs(beta_hat), bar)
    binding = "noise_floor" if bar > abs(beta_hat) else "pilot_estimate"
    if target <= 0:
        raise ValueError("no effect bar to power against: both the pilot estimate "
                         "and the noise floor are zero")

    z_a = norm.ppf(1 - alpha / 2)
    z_b = norm.ppf(power)
    n_req = n_pilot * ((z_a + z_b) * se_pilot / target) ** 2
    return PowerResult(int(np.ceil(n_req)), float(target), float(se_pilot),
                       int(n_pilot), float(power), float(alpha), binding)


@dataclass
class GateDecision:
    proceed_full: bool
    n_required: int
    dispersion: float
    dispersion_threshold: float
    max_documents: int
    reasons: list[str]

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        head = "PROCEED (full study)" if self.proceed_full else "RESTRICT (110M only, shift compute to C2)"
        return head + "\n" + "\n".join(f"  - {r}" for r in self.reasons)


def stage0_gate(n_required: int, dispersion: float, max_documents: int = 1000,
                dispersion_threshold: float = 0.25) -> GateDecision:
    """The pre-registered Stage 0 decision.

    ``dispersion`` is the estimated spread of position-specific noise scales, on
    the log scale (see :mod:`c3.dispersion`).  The threshold is a judgement call
    and must be fixed *before* the pilot is read; it is recorded here so the
    decision is auditable rather than retrofitted.
    """
    reasons, ok = [], True
    if n_required <= max_documents:
        reasons.append(f"required sample {n_required} <= {max_documents} documents")
    else:
        ok = False
        reasons.append(f"required sample {n_required} EXCEEDS {max_documents} documents")

    if dispersion >= dispersion_threshold:
        reasons.append(f"noise dispersion {dispersion:.3f} >= threshold {dispersion_threshold}")
    else:
        ok = False
        reasons.append(
            f"noise dispersion {dispersion:.3f} BELOW threshold {dispersion_threshold}; "
            "under near-homoskedastic noise the selection question has no content "
            "and a nonzero beta_z would indicate misspecification, not selection")
    return GateDecision(ok, int(n_required), float(dispersion),
                        float(dispersion_threshold), int(max_documents), reasons)

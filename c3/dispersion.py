"""Estimating the dispersion of position-specific approximation error.

Sec. 5.1 makes dispersion the precondition for the whole audit: if the noise
scale is uniform across positions, conditioning on reported confidence makes the
selection event ancillary and the null and alternative predict the same thing.
Stage 0 therefore has to estimate it, and the gate turns on the answer.

The estimator here uses attenuation.  The proposal defines ``ell_i = ell*_i +
eps_i`` with ``ell*`` the *true* log-odds, so correctness satisfies
``logit P(agree) = alpha_state + ell*`` with slope exactly one by construction.
Regressing agreement on the *observed* ``ell`` instead attenuates that slope by
the classical measurement-error factor, so within a stratum

    b  ~=  Var(ell*) / Var(ell)      =>      sigma^2  ~=  Var(ell) * (1 - b)

and the spread of ``log sigma`` across strata estimates the dispersion.

Identification caveat, and it is not a small one: dispersion is only estimable
when the noise scale *tracks something observable* to stratify on.  If the scale
varies idiosyncratically from position to position with no observable
correlate, every stratum averages over the same mixture and the estimate
collapses toward zero regardless of the true dispersion.  What this function
returns is therefore a lower bound -- the dispersion visible through the chosen
stratifier -- and the default stratifier is position in the document, which is
where the closest prior work (PC-Sampler's position-aware weight) locates it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frame import AnalysisFrame
from .logit import fit_fe_logit


@dataclass
class DispersionEstimate:
    dispersion: float            # sd of log sigma across strata
    sigma_by_stratum: np.ndarray
    strata_used: int
    strata_dropped: int
    mean_sigma: float
    stratifier: str
    slopes: np.ndarray = None            # attenuation slope per stratum
    n_no_attenuation: int = 0            # strata with b_hat >= 1

    def __str__(self) -> str:
        if self.n_no_attenuation and self.strata_used == 0:
            return (f"dispersion = 0.000 -- no attenuation detected in any of "
                    f"{self.n_no_attenuation} strata by {self.stratifier} "
                    f"(slope >= 1 throughout), so there is no position-specific "
                    f"approximation error for selection to act on")
        return (f"dispersion(log sigma) = {self.dispersion:.3f} over "
                f"{self.strata_used} strata by {self.stratifier} "
                f"(mean sigma {self.mean_sigma:.3f}, {self.strata_dropped} dropped, "
                f"{self.n_no_attenuation} with no attenuation)")


def _unique_positions(frame: AnalysisFrame):
    """Collapse pool rows back to one row per (state, position).

    Pool membership duplicates each position many times.  Attenuation must be
    estimated on the underlying positions, otherwise the duplication is mistaken
    for information.
    """
    key = frame.state_id.astype(np.int64) * (frame.position.max() + 1) + frame.position
    _, first = np.unique(key, return_index=True)
    return frame.select(np.sort(first))


def estimate_dispersion(frame: AnalysisFrame, n_strata: int = 10,
                        stratifier: str = "position", min_per_stratum: int = 200
                        ) -> DispersionEstimate:
    """Dispersion of position-specific noise, via stratified attenuation."""
    f = _unique_positions(frame)
    if stratifier == "position":
        key = f.position.astype(np.float64) / max(1.0, float(f.position.max()))
    elif stratifier == "ell":
        key = f.ell
    else:
        raise ValueError(f"unknown stratifier {stratifier!r}")

    edges = np.quantile(key, np.linspace(0, 1, n_strata + 1))
    edges[-1] += 1e-9
    bucket = np.clip(np.searchsorted(edges, key, side="right") - 1, 0, n_strata - 1)

    sigmas, dropped, no_atten, slopes = [], 0, 0, []
    groups, n_groups = f.state_index
    for b in range(n_strata):
        sel = bucket == b
        if sel.sum() < min_per_stratum or len(np.unique(f.agree[sel])) < 2:
            dropped += 1
            continue
        V = float(np.var(f.ell[sel]))
        if V <= 0:
            dropped += 1
            continue
        g = np.unique(groups[sel], return_inverse=True)[1]
        try:
            fit = fit_fe_logit(f.agree[sel], f.ell[sel][:, None], g,
                               n_groups=int(g.max() + 1), names=["slope"], ridge_fe=1e-4)
        except (FloatingPointError, np.linalg.LinAlgError):
            dropped += 1
            continue
        b_hat = fit.coef("slope")
        slopes.append(b_hat)
        # b >= 1 is not a failure to estimate: it says reported log-odds move at
        # least one-for-one with correctness, so the attenuation model implies no
        # detectable approximation error at that stratum (sigma^2 <= 0). Counting
        # those as "no attenuation" rather than dropping them is the difference
        # between reporting nan and reporting the finding, which is that there is
        # nothing here for selection to bite on.
        if b_hat >= 1.0:
            no_atten += 1
            continue
        if b_hat <= 0.0:
            dropped += 1
            continue
        sigmas.append(np.sqrt(V * (1.0 - b_hat)))

    slopes = np.asarray(slopes)
    if len(sigmas) < 2:
        # Every stratum that could be fitted showed no attenuation -> dispersion
        # is zero, not unknown.
        disp = 0.0 if no_atten > 0 else float("nan")
        return DispersionEstimate(disp, np.array(sigmas), len(sigmas), dropped,
                                  float("nan"), stratifier, slopes, no_atten)
    sig = np.asarray(sigmas)
    return DispersionEstimate(float(np.std(np.log(sig), ddof=1)), sig, len(sig),
                              dropped, float(sig.mean()), stratifier, slopes, no_atten)

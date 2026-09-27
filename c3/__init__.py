"""C3: Committed Confidence Calibration.

Selection-conditional calibration audit for diffusion language models.
The specification is the detailed plan v2 (documents/C3_Detailed_Plan_v2.pdf),
which supersedes the proposal (documents/DuckDuckDuckGoose-Proposal.pdf).

The constants below are the analysis choices the plan fixes in advance.  Every
script takes its defaults from here so a pre-registered value is stated once.
"""

__version__ = "0.2.0"

POOL_SIZES = (1, 2, 4, 8, 16, 32)
"""Field sizes for E1/E2 (plan v2 T2, T5).  m=1 is the no-selection baseline."""

N_POOLS = 20
"""Random pools drawn per field size per state (plan v2 T5)."""

RETAIN = 4
"""Top ranks recorded per pool, for the rank profile gamma_k (plan v2 T5)."""

N_BOOT = 10_000
"""Document-bootstrap resamples for every reported interval (plan v2 Sec. 5)."""

DEFAULT_KNOTS = 9
"""Restricted-cubic-spline knots for f(ell).  5 and 13 are the sensitivity panel."""

DELTA_MIN = 0.02
"""Smallest effect worth reporting on Delta(32) - Delta(1), fixed before looking (T2)."""

LEGACY_POOL_SIZES = (2, 4, 8, 16, 32)
"""The proposal-era grid.  Only the scripts that produced the committed
proposal-era artifacts use it, so re-running them reproduces those records."""

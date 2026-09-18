"""C3: Committed Confidence Calibration.

Selection-conditional calibration audit for diffusion language models.
See documents/DuckDuckDuckGoose-Proposal.pdf for the specification this implements.
"""

__version__ = "0.1.0"

POOL_SIZES = (2, 4, 8, 16, 32)

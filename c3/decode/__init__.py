"""Decoding: per-position scoring, commit policies and the shared decode loop.

``scoring``   per-position statistics from one forward pass (plan v2 Sec. 3.4)
``policies``  the T12 commit rules (top-k confidence, Fast-dLLM threshold,
              EB-Sampler, margin, KLASS, PC-Sampler)
``loop``      free-running, sampled or teacher-forced decoding, block-wise or not
"""

"""T9: the synthetic JSON probe, where C2 can be computed exactly (plan v2 T9).

Committing k masked tokens in one step samples them from the product of their
per-position marginals, so the decoder's error against the true joint splits as

    KL(p*(x_S | x_obs) || prod_i p_theta,i) = TC(x_S | x_obs) + sum_i KL(p*_i || p_theta,i)
                                              (A) dependency    (B) estimation

On real text p* is unknown.  Here every instance is drawn from a known
distribution over single-token field values, so p*, both terms and their sum
are exact, in nats.

* ``c3.probe.spec``      -- fields from SchemaStore schemas, value lists, marginals, pairs
* ``c3.probe.instances`` -- seeded instance generation and compact-JSON serialisation
* ``c3.probe.truth``     -- exact p* by enumeration and the (A)/(B) decomposition
* ``c3.probe.tokens``    -- the single-token-in-context check under any tokenizer
* ``c3.probe.grid``      -- the rho x k x selection-rule runner on a diffusion LM
"""

RHOS = (0.0, 0.25, 0.5, 0.75, 1.0)
"""Dependency knob: probability that a follower copies its leader (plan v2 T9 step 3)."""

KS = (1, 2, 4, 8)
"""Tokens committed per decoding step."""

RULES = ("random", "confidence")
"""Selection rules: uniformly random masked positions, or confidence top-k."""

PROBE_SEED = 20261012
"""Seed for the spec (marginals, pairs, string value lists) and every instance set."""

N_SCHEMAS = 200
N_EVAL_PER_RHO = 200
N_TRAIN = 200_000
N_VAL = 2_000
MAX_TOKENS = 256

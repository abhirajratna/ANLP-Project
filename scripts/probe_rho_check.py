#!/usr/bin/env python3
"""Does a model condition on the probe's rho marker?  (T9 diagnostic)

    python scripts/probe_rho_check.py --model-path final.pt --tag ft110m-v3

For every pair (leader a, follower b) of the rho = 0 and rho = 1 evaluation
instances, only the leader is observed and the model's q_b is read twice: with
the instance's own marker and with the marker swapped to the other extreme.
A model that uses rho puts nearly all of q_b on f(a) at rho = 1 and about
p_b(f(a)) at rho = 0; a model that ignores the marker gives the same q_b(f(a))
either way and predicts a mixture over rho, which inflates (B) at high rho.

Writes ``<out-dir>/rho_check_<tag>.json``: per true rho and marker, the mean
follower KL(p*(b | a) || q_b), the mean q_b(f(a)), and the leader's marginal KL
with nothing observed (how well the marginals were learned).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c3.probe.grid import prepare, score
from c3.probe.instances import Instance, ProbeSpec, eval_instances
from c3.probe.truth import MASKED, kl, marginals
from c3.provenance import collect as provenance, sha256_file
from c3.util import hard_exit


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default="artifacts/probe/spec.json")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default="artifacts/probe/results")
    ap.add_argument("--n-schemas", type=int, default=200)
    a = ap.parse_args()

    from c3.traces.models import build_model
    spec = ProbeSpec.load(a.spec)
    model = build_model("local-110m", path=a.model_path, device=a.device)
    ev = [i for i in eval_instances(spec, per_rho=200)
          if i.rho in (0.0, 1.0) and i.schema < a.n_schemas]
    acc = defaultdict(list)
    for inst in ev:
        sch = spec.schemas[inst.schema]
        pairs = [g for g in sch.groups if len(g.members) == 2]
        for marker in ("own", "swapped"):
            use = inst if marker == "own" else Instance(inst.instance_id, inst.schema,
                                                        1.0 - inst.rho, inst.values)
            (p,) = prepare(spec, [use], model.tokenizer, "gpt2")
            obs_list = []
            for g in pairs:
                o = np.full(len(sch.fields), MASKED)
                o[g.members[0]] = inst.values[g.members[0]]
                obs_list.append(o)
            if marker == "own":
                obs_list.append(np.full(len(sch.fields), MASKED))
            scored = score(model, [p] * len(obs_list), obs_list)
            for g, o, sc in zip(pairs, obs_list, scored):
                a_, b = g.members
                logq = sc[b][0]
                acc[(inst.rho, marker, "follower_kl")].append(
                    kl(marginals(sch, inst.rho, o, [b])[b], logq))
                acc[(inst.rho, marker, "q_copy")].append(float(np.exp(logq[g.fmap[inst.values[a_]]])))
            if marker == "own":
                empty, sc0 = obs_list[-1], scored[-1]
                for g in pairs:
                    lead = g.members[0]
                    acc[(inst.rho, marker, "leader_marginal_kl")].append(
                        kl(marginals(sch, inst.rho, empty, [lead])[lead], sc0[lead][0]))
    table = {f"rho={k[0]:g}/marker={k[1]}/{k[2]}": {"mean": float(np.mean(v)), "n": len(v)}
             for k, v in sorted(acc.items())}
    shift = (table["rho=1/marker=own/q_copy"]["mean"] - table["rho=1/marker=swapped/q_copy"]["mean"])
    record = {"tag": a.tag, "checkpoint_sha256": sha256_file(a.model_path), "spec_sha256":
              sha256_file(a.spec), "n_instances": len(ev), "results": table,
              "marker_effect_on_q_copy_at_rho1": shift, "provenance": provenance()}
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"rho_check_{a.tag}.json").write_text(json.dumps(record, indent=1))
    for k, v in table.items():
        print(f"[rho-check] {a.tag} {k}: {v['mean']:.4f} (n={v['n']})")
    print(f"[rho-check] {a.tag}: the marker moves q(f(a)) at rho=1 by {shift:+.4f}")
    return 0


if __name__ == "__main__":
    hard_exit(main())

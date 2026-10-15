#!/usr/bin/env python3
"""Rescore T9 fine-tunes on the probe validation set under one protocol.  (T9)

    python scripts/rescore_probe_val.py --index probe-shards/index.json \\
        --run ft110m-v3=/path/final.pt --run ft110m-v3-visible=/path/final.pt

ft110m-v3-visible was trained with the rho marker never masked, but the validation
during training masked and scored it, so its logged NELBO is not comparable with
ft110m-v3's.  This scores every run's raw and EMA weights on the same validation
batches the fine-tunes used (split "val", prefix windows, the run's batch size,
batch count and seed) twice:

  marker_masked    the protocol of the training logs (a check: it reproduces the
                   last validation row of history_<run>.jsonl)
  marker_visible   the first ``visible_prefix_tokens`` of each window (the marker,
                   from the shard index) are context, never masked and not scored

Writes ``<out>``: per run and weights, both NELBOs, with the history's last row.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def last_val_row(history: Path) -> dict | None:
    if not history.exists():
        return None
    rows = [json.loads(line) for line in history.read_text().splitlines() if '"val_nelbo"' in line]
    return rows[-1] if rows else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="probe shard index.json (scripts/build_probe_shards.py)")
    ap.add_argument("--run", action="append", required=True, help="tag=final.pt, repeatable")
    ap.add_argument("--history-dir", default="artifacts/probe/results")
    ap.add_argument("--out", default="artifacts/probe/results/val_rescored.json")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    import torch

    from c3.provenance import collect as provenance
    from c3.training.data import DocWindowDataset
    from c3.training.model import MaskedDiffusionLM, ModelConfig
    from c3.training.train import evaluate

    n_visible = int(json.loads(Path(a.index).read_text())["visible_prefix_tokens"])
    out = {"index": a.index, "visible_prefix_tokens": n_visible, "runs": {}}
    for spec in a.run:
        tag, path = spec.split("=", 1)
        state = torch.load(path, map_location="cpu", weights_only=False)
        cfg, mcfg = state["cfg"], state["model_cfg"]
        val = DocWindowDataset(a.index, cfg["seq_len"], mode="prefix", split="val", pad_id=50256)
        batches = list(val.fixed_batches(cfg["eval_batches"], cfg["batch_size"], seed=cfg["seed"]))
        model = MaskedDiffusionLM(ModelConfig(**mcfg)).to(a.device)
        res = {"checkpoint": path, "trained_visible_prefix": cfg.get("visible_prefix", 0),
               "eval_batches": cfg["eval_batches"], "batch_size": cfg["batch_size"],
               "seed": cfg["seed"], "history_last_val": last_val_row(Path(a.history_dir) / f"history_{tag}.jsonl")}
        for which in ("model", "model_ema"):
            if not state.get(which):
                continue
            model.load_state_dict({k: v.to(torch.float32) for k, v in state[which].items()})
            res["ema" if which == "model_ema" else "raw"] = {
                "marker_masked": evaluate(model, batches, mcfg["mask_token_id"], a.device),
                "marker_visible": evaluate(model, batches, mcfg["mask_token_id"], a.device,
                                           visible_prefix=n_visible)}
            r = res["ema" if which == "model_ema" else "raw"]
            print(f"[rescore] {tag} {which}: marker masked {r['marker_masked']['val_nelbo']:.4f}, "
                  f"marker visible {r['marker_visible']['val_nelbo']:.4f} nats/token", flush=True)
        out["runs"][tag] = res
    out["provenance"] = provenance(with_gpu=a.device.startswith("cuda"))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"[rescore] -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

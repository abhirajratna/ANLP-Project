#!/usr/bin/env python3
"""Publish the trained 110M model to the HuggingFace Hub (course README requirement).

Uploads, to a model repo (private by default):
  model.safetensors      the EMA weights (or raw with --raw-weights), mask logit SUBS applies at load
  config.json            ModelConfig + training config + config hash + git commit
  train_summary.json     from the run directory, if present
  history.jsonl          the training log from the run directory, if present
  eval_110m.json         from scripts/eval_110m.py, if given
  README.md              a model card generated from the above (numbers only)

Needs ``huggingface_hub`` and a token with write access (``huggingface-cli login``
or HF_TOKEN).  Loading back: ``c3.traces.models.LocalMDLM`` reads ``final.pt``;
``load_from_hub`` below rebuilds that from the uploaded files.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="final.pt from the run")
    ap.add_argument("--repo", required=True, help="e.g. ANLP-project/c3-mdlm-110m")
    ap.add_argument("--eval", default=None, help="eval_110m.json to include")
    ap.add_argument("--summary", default=None, help="train_summary.json (default: next to --ckpt)")
    ap.add_argument("--note", default=None, help="one paragraph for the model card (what the run is for)")
    ap.add_argument("--raw-weights", action="store_true")
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="write the files locally, upload nothing")
    a = ap.parse_args()

    import torch
    from safetensors.torch import save_file

    from c3.provenance import git_commit

    state = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    weights = state["model"] if a.raw_weights or not state.get("model_ema") else state["model_ema"]
    which = "raw" if weights is state["model"] else "ema"
    # Tied embeddings: safetensors refuses shared storage, so store one copy.
    tensors = {k: v.float().contiguous().clone() for k, v in weights.items() if k != "head.weight"}
    cfg = {"model_config": state.get("model_cfg"), "train_config": state.get("cfg"),
           "config_hash": state.get("config_hash"), "steps": state.get("step"),
           "tokens_seen": state.get("tokens_seen"), "weights": which,
           "git_commit": git_commit()[0], "tied": {"head.weight": "tok.weight"},
           "tokenizer": "gpt2", "mask_token_id": (state.get("model_cfg") or {}).get("mask_token_id")}
    summary = Path(a.summary) if a.summary else Path(a.ckpt).with_name("train_summary.json")
    history = Path(a.ckpt).with_name("history.jsonl")
    init_from = (cfg["train_config"] or {}).get("init_from")
    ev = json.loads(Path(a.eval).read_text()) if a.eval else None

    origin = "fine-tuned" if init_from else "trained from scratch"
    card = [f"# {a.repo.split('/')[-1]}", "",
            f"Masked diffusion language model (MDLM objective, absorbing mask, SUBS), {origin} "
            "for the C3 project (committed-confidence calibration in diffusion LMs).", "",
            *([a.note, ""] if a.note else []),
            "| | |", "|---|---|",
            f"| parameters | {sum(t.numel() for t in tensors.values())/1e6:.1f}M (tied embeddings) |",
            f"| training tokens | {cfg['tokens_seen']:,} |" if cfg["tokens_seen"] else None,
            f"| steps | {cfg['steps']} |", f"| weights | {which} |",
            f"| config hash | `{cfg['config_hash']}` |", f"| code commit | `{cfg['git_commit']}` |",
            None if init_from else
            "| data | FineWeb-Edu sample-10BT, document-aligned 256-token windows (no packing), "
            "evaluation documents excluded by id / text hash / first-256-token hash |",
            "| tokenizer | GPT-2 BPE + one [MASK] token (id 50257) |"]
    if ev:
        card += [f"| held-out NELBO | {ev['nelbo_per_token']:.4f} nats/token "
                 f"(ppl bound {ev['ppl_bound']:.2f}, {ev['documents']} documents) |"]
        if ev.get("gpt2_reference"):
            card += [f"| GPT-2 small, same documents | ppl {ev['gpt2_reference']['ppl']:.2f} |"]
    card = "\n".join(line for line in card if line is not None) + "\n"

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) if not a.dry_run else Path(a.ckpt).with_name("hf_upload")
        d.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(d / "model.safetensors"))
        (d / "config.json").write_text(json.dumps(cfg, indent=2))
        (d / "README.md").write_text(card)
        if summary.exists():
            (d / "train_summary.json").write_text(summary.read_text())
        if history.exists():
            (d / "history.jsonl").write_text(history.read_text())
        if ev:
            (d / "eval_110m.json").write_text(json.dumps(ev, indent=2))
        if a.dry_run:
            print(f"[hf] dry run: files written to {d}")
            return 0
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(a.repo, private=not a.public, exist_ok=True, repo_type="model")
        api.upload_folder(folder_path=str(d), repo_id=a.repo, repo_type="model",
                          commit_message=f"110M MDLM ({which} weights, config {cfg['config_hash']})")
    print(f"[hf] uploaded -> https://huggingface.co/{a.repo}  (put this link in README.md)")
    return 0


def load_from_hub(repo: str, out_path: str | Path) -> Path:
    """Rebuild a ``final.pt`` that ``LocalMDLM`` can load, from the uploaded files."""
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    w = load_file(hf_hub_download(repo, "model.safetensors"))
    cfg = json.loads(Path(hf_hub_download(repo, "config.json")).read_text())
    w["head.weight"] = w["tok.weight"]
    torch.save({"model": w, "model_cfg": cfg["model_config"], "step": cfg["steps"],
                "tokens_seen": cfg["tokens_seen"], "config_hash": cfg["config_hash"]}, out_path)
    return Path(out_path)


if __name__ == "__main__":
    raise SystemExit(main())

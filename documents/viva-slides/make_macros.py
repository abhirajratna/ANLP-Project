"""Write macros_setup.tex: the setup facts the viva deck states, read from committed artifacts.

    python3 documents/viva-slides/make_macros.py      # from the repository root

The C2 numbers come from artifacts/probe/macros_t9.tex and artifacts/c2/macros_t10.tex, and
the C1 and C3 results will come from the macro files their analyses write. This script
covers what no other macro file holds: the C1 arms (documents, rows, passes, GPU-hours),
the non-determinism floors, the design constants fixed in c3/__init__.py, the split, the
contamination screen, the 110M model and the probe's size. Standard library only.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "macros_setup.tex"

# label used in the macro name -> artifacts/c1/<run>.json; the labels follow
# prereg/analysis_commands.sh (ARMS and the time-split rows).
ARMS = {
    "LladaBase": "c1-llada-base",
    "DreamBase": "c1-dream-base",
    "LladaInstructGsm": "c1-llada-instruct-gsm8k",
    "LladaInstructMbpp": "c1-llada-instruct-mbpp-v2",
    "DreamInstructGsm": "c1-dream-instruct-gsm8k",
    "DreamInstructMbpp": "c1-dream-instruct-mbpp",
    "Mdlm": "main-110m-v3",
    "LladaBaseTimeSplit": "c1ts-llada-base",
    "DreamBaseTimeSplit": "c1ts-dream-base",
    "MdlmTimeSplit": "timesplit-110m-v3",
    "LladaBaseFourBit": "c1t8-llada-base-4bit",
}
FLOORS = {
    "Llada": "artifacts/t2_pilot/floor_llada_base_fp16.json",
    "Dream": "artifacts/t2_pilot/floor_dream-7b-base_fp16.json",
    "Mdlm": "artifacts/t2_pilot/floor_local-110m-v3_fp32.json",
}

sources: list[str] = []


def load(rel: str) -> dict:
    p = ROOT / rel
    sources.append(f"{rel}  {hashlib.sha256(p.read_bytes()).hexdigest()}")
    return json.loads(p.read_text())


def thousands(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def fmt(x: float, nd: int) -> str:
    return thousands(int(round(x))) if nd == 0 else f"{x:.{nd}f}"


def constants() -> dict:
    rel = "c3/__init__.py"
    p = ROOT / rel
    sources.append(f"{rel}  {hashlib.sha256(p.read_bytes()).hexdigest()}")
    out = {}
    for node in ast.parse(p.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], "id", None)
            if name:
                try:
                    out[name] = ast.literal_eval(node.value)
                except ValueError:
                    pass
    return out


def main() -> None:
    m: list[tuple[str, str, str]] = []  # (name, value, comment)

    def add(name: str, value: str, comment: str = "") -> None:
        m.append((name, value, comment))

    # ---- C1 arms
    tot_rows = tot_gpuh = 0.0
    for label, run in ARMS.items():
        d = load(f"artifacts/c1/{run}.json")
        assert d["all_ok"], run
        s = d["summary"]
        add(f"Viva{label}Docs", thousands(s["documents"]), f"{run}: summary.documents")
        add(f"Viva{label}RowsM", fmt(s["rows"] / 1e6, 1), "summary.rows / 1e6")
        add(f"Viva{label}Passes", thousands(s["forward_passes"]), "summary.forward_passes")
        add(f"Viva{label}GpuH", fmt(s["gpu_hours"], 0), "summary.gpu_hours")
        add(f"Viva{label}Precision", s["precision"], "summary.precision")
        tot_rows += s["rows"]
        tot_gpuh += s["gpu_hours"]
        if label == "LladaBase":
            des = s["design"]
            add("VivaDecoderK", ", ".join(str(k) for k in des["decoder_k"]), "design.decoder_k")
            rates = des["mask_rates"]
            add("VivaMaskRateMin", f"{min(rates):g}", "design.mask_rates")
            add("VivaMaskRateMax", f"{max(rates):g}", "design.mask_rates")
            add("VivaMasksPerRate", str(des["masks_per_rate"]), "design.masks_per_rate")
            add("VivaPassesPerDoc", fmt(d["checks"]["forward_passes"]["per_doc"], 0),
                "checks.forward_passes.per_doc")
            add("VivaCorpusDocs", thousands(d["checks"]["every_document_present"]["n_corpus"]),
                "checks.every_document_present.n_corpus")
            add("VivaTopK", str(des["topk"]), "design.topk")
    add("VivaCOneArms", str(len(ARMS)), "number of C1 runs listed here")
    add("VivaCOneRowsB", fmt(tot_rows / 1e9, 2), "sum of summary.rows / 1e9")
    add("VivaCOneGpuH", fmt(tot_gpuh, 0), "sum of summary.gpu_hours")

    # ---- floors
    for fam, rel in FLOORS.items():
        add(f"Viva{fam}Floor", f"{load(rel)['plan_v2']['floor_delta']:g}", f"{rel}: plan_v2.floor_delta")

    # ---- design constants
    c = constants()
    add("VivaPoolSizes", ", ".join(str(x) for x in c["POOL_SIZES"]), "c3.POOL_SIZES")
    add("VivaPoolMax", str(max(c["POOL_SIZES"])), "c3.POOL_SIZES")
    add("VivaNPools", str(c["N_POOLS"]), "c3.N_POOLS")
    add("VivaNBoot", thousands(c["N_BOOT"]), "c3.N_BOOT")
    add("VivaKnots", str(c["DEFAULT_KNOTS"]), "c3.DEFAULT_KNOTS")
    add("VivaDeltaMin", f"{c['DELTA_MIN']:g}", "c3.DELTA_MIN")
    add("VivaContamThreshold", str(c["CONTAM_THRESHOLD"]), "c3.CONTAM_THRESHOLD")

    # ---- split and contamination
    sm = load("artifacts/split_manifest.meta.json")
    add("VivaNFit", thousands(sm["n_fit"]), "split_manifest.meta.json: n_fit")
    add("VivaNReport", thousands(sm["n_report"]), "n_report")
    for fam, model in (("Llada", "llada-8b-base"), ("Dream", "dream-7b-base")):
        cd = load(f"artifacts/contamination_{model}.json")
        add(f"Viva{fam}ContamFlagged", str(cd["n_flagged"]), "n_flagged")
        add(f"Viva{fam}ContamDocs", thousands(cd["n_docs"]), "n_docs")
        add(f"Viva{fam}ContamPrefix", str(cd["prefix_tokens"]), "prefix_tokens")

    # ---- the 110M model
    ts = load("artifacts/train_summary_train110m-v3.json")
    mc = ts["model_config"]
    add("VivaMdlmLayers", str(mc["n_layers"]), "model_config.n_layers")
    add("VivaMdlmWidth", str(mc["d_model"]), "model_config.d_model")
    add("VivaMdlmHeads", str(mc["n_heads"]), "model_config.n_heads")
    add("VivaMdlmSeqLen", str(mc["max_len"]), "model_config.max_len")
    add("VivaMdlmTokensB", fmt(ts["tokens_seen"] / 1e9, 1), "tokens_seen / 1e9")
    add("VivaMdlmSteps", thousands(ts["steps"]), "steps")
    ev = load("artifacts/eval_110m_train110m-v3.json")
    add("VivaMdlmPplBound", fmt(ev["ppl_bound"], 1), "ppl_bound (NELBO)")
    add("VivaMdlmPplBoundLo", fmt(ev["ppl_bound_ci"][0], 1), "ppl_bound_ci")
    add("VivaMdlmPplBoundHi", fmt(ev["ppl_bound_ci"][1], 1), "ppl_bound_ci")
    add("VivaGptTwoPpl", fmt(ev["gpt2_reference"]["ppl"], 1), "gpt2_reference.ppl")
    add("VivaGptTwoPplLo", fmt(ev["gpt2_reference"]["ppl_ci"][0], 1), "gpt2_reference.ppl_ci")
    add("VivaGptTwoPplHi", fmt(ev["gpt2_reference"]["ppl_ci"][1], 1), "gpt2_reference.ppl_ci")
    add("VivaMdlmEvalDocs", thousands(ev["documents"]), "documents")

    # ---- the probe
    pm = load("artifacts/probe/manifest.json")
    add("VivaProbeSchemas", str(pm["n_schemas"]), "n_schemas")
    add("VivaProbeFieldsMin", str(pm["fields_per_schema"]["min"]), "fields_per_schema.min")
    add("VivaProbeFieldsMax", str(pm["fields_per_schema"]["max"]), "fields_per_schema.max")
    add("VivaProbeTrainK", fmt(pm["train"]["n"] / 1e3, 0), "train.n / 1e3")

    for name, _, _ in m:
        assert name.isalpha(), name  # TeX macro names: letters only
    lines = [
        "% Generated by documents/viva-slides/make_macros.py -- do not edit by hand.",
        "% Sources (sha256):",
        *[f"%   {s}" for s in sources],
        "",
    ]
    lines += [f"\\newcommand{{\\{n}}}{{{v}}}" + (f"  % {c}" if c else "") for n, v, c in m]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(m)} macros)")


if __name__ == "__main__":
    main()

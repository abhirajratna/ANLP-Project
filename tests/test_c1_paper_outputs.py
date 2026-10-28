"""scripts/c1_paper_outputs.py on the synthetic fixtures of tests/fixtures/c1_results.

The fixtures are the output of the real analysis scripts on a toy model (see
make_fixtures.py there): a schema check, not results.  The report-half results do not
exist yet; these tests are what lets the tables and figures be ready when they do.
"""
import json
import re
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "c1_results"
sys.path.insert(0, str(ROOT / "scripts"))

import c1_paper_outputs as P  # noqa: E402

NEWCOMMAND = re.compile(r"^\\newcommand\{\\([^}]*)\}\{(.*)\}$")
TABLES = ("c1_main", "c1_main_col", "c1_beta", "c1_gamma", "c1_shared", "c1_sensitivity", "c1_e3",
          "c1_t8",
          "c1_t8_precision")


@pytest.fixture
def results(tmp_path):
    """A writable copy of the fixture results root."""
    dst = tmp_path / "results"
    shutil.copytree(FIXTURES, dst, ignore=shutil.ignore_patterns("make_fixtures.py", "__pycache__"))
    return dst


def run(results, tmp_path, *extra):
    out = tmp_path / "paper"
    rc = P.main(["--results", str(results), "--out-tables", str(out / "tables" / "c1"),
                 "--out-figures", str(out / "figures" / "c1"), *extra])
    assert rc == 0
    return out


def table(out, name):
    return (out / "tables" / "c1" / f"{name}.tex").read_text()


def rows(tex):
    """The data rows of a tabular: lines ending in ``\\\\`` between midrule and bottomrule."""
    body = tex.split("\\midrule", 1)[1].split("\\bottomrule", 1)[0]
    return [ln.rstrip()[:-2].rstrip().split(" & ") for ln in body.splitlines() if ln.rstrip().endswith("\\\\")]


def macro_defs(path):
    out = {}
    for ln in Path(path).read_text().splitlines():
        if ln.startswith("%") or not ln.strip() or ln.startswith("\\input"):
            continue
        m = NEWCOMMAND.match(ln)
        assert m, f"{path}: not a well-formed \\newcommand: {ln!r}"
        out[m.group(1)] = m.group(2)
    return out


def test_the_fixture_is_labelled_synthetic():
    res = json.loads((FIXTURES / "c1" / "LladaBase" / "results.json").read_text())
    assert "SYNTHETIC" in res["label"] and res["input"]["source"].startswith("synthetic:")


def test_every_table_is_compile_ready(results, tmp_path):
    out = run(results, tmp_path, "--no-figures")
    for name in TABLES:
        tex = table(out, name)
        assert tex.count("\\begin{tabular}") == tex.count("\\end{tabular}") == 1
        assert all(r in tex for r in ("\\toprule", "\\midrule", "\\bottomrule"))
        assert tex.count("\\begingroup") == tex.count("\\endgroup") == 1
        assert tex.count("{") == tex.count("}")                    # braces balance
        assert tex.count("$") % 2 == 0                              # math mode closes
        for bad in ("nan", "inf", "None", "Traceback"):
            assert bad not in tex.replace("\\scriptsize", ""), (name, bad)
        spec = re.search(r"\\begin\{tabular\}\{(\w+)\}", tex).group(1)
        head = tex.split("\\midrule", 1)[0].splitlines()[-1][:-2].split(" & ")
        assert len(head) == len(spec)
        assert all(len(r) == len(spec) for r in rows(tex)), name   # every row has every cell


def test_main_table_has_one_row_pair_per_arm_and_the_summary_decisions(results, tmp_path):
    out = run(results, tmp_path, "--no-figures")
    tex = table(out, "c1_main")
    rs = rows(tex)
    arms = list(P.MAIN_ARMS) + ["LladaBaseTimeSplit"]
    assert len(rs) == 2 * len(arms)
    assert [r[0] for r in rs[::2]] == [P.display(a) for a in arms]
    assert [r[1] for r in rs[:2]] == ["uniform", "dec. $k{=}4$"]
    summ = json.loads((results / "c1" / "summary" / "c1_summary.json").read_text())
    for label in P.MAIN_ARMS:
        for i, mask in enumerate(P.PRIMARY_MASKS):
            row = next(r for r in summ["rows"] if r["model"] == label and r["arm"] == mask)
            r = rs[2 * arms.index(label) + i]
            assert r[3] == ("yes" if row["contrast"]["holm_reject"] else "no")
            assert r[5] == ("yes" if row["field_contrast"]["holm_reject"] else "no")
            res = json.loads((results / "c1" / label / "results.json").read_text())
            c = res["by_mask_kind"][mask]["contrast"]
            assert f"{abs(c['value']):.3f}" in r[2]
            assert ("\\dagger" in r[2]) == bool(c["declared"])
    # the time-split arm is not in the Holm summary: no decision
    assert rs[-1][3] == rs[-1][5] == P.DASH
    # markers share one superscript
    assert "$^{*}$$^{" not in tex and "$^{\\dagger}$$^{" not in tex


def test_sensitivity_shows_the_variants_that_exist_and_dashes_the_rest(results, tmp_path):
    out = run(results, tmp_path, "--no-figures")
    rs = rows(table(out, "c1_sensitivity"))
    llada, dream = rs[0], rs[2]
    assert P.DASH not in llada[2:]                      # knots 5 and 13 and clean all there
    assert dream[4] != P.DASH and dream[6] == P.DASH    # DreamBase has knots5 only
    assert dream[3] == dream[7] == P.DASH               # no clean variant


def test_a_missing_arm_is_a_row_of_dashes_and_the_run_does_not_crash(results, tmp_path, capsys):
    shutil.rmtree(results / "c1" / "Mdlm")
    shutil.rmtree(results / "e3" / "Mdlm")
    shutil.rmtree(results / "t8" / "Mdlm")
    out = run(results, tmp_path, "--no-figures")
    rs = rows(table(out, "c1_main"))
    mdlm = [r for r in rs if r[0] == P.display("Mdlm")]
    assert len(mdlm) == 1 and mdlm[0][2:] == [P.DASH] * 6
    assert rows(table(out, "c1_t8"))[12][2:] == [P.DASH] * 8
    macros = (out / "tables" / "c1" / "macros_c1.tex").read_text()
    assert "arms without results: Mdlm" in macros
    assert "c1-Mdlm" not in macros and "COneMdlm" not in macros.replace("COneHolm", "")
    assert "no results for: Mdlm" in capsys.readouterr().out


def test_missing_summary_means_no_holm_decisions_and_no_holm_macros(results, tmp_path):
    shutil.rmtree(results / "c1" / "summary")
    out = run(results, tmp_path, "--no-figures")
    rs = rows(table(out, "c1_main"))
    assert all(r[3] == r[5] == P.DASH for r in rs)
    names = macro_defs(out / "tables" / "c1" / "macros_c1.tex")
    assert not any(n.startswith("COneHolm") for n in names)
    assert "COneFieldContrastLladaBaseUni" in names           # still there: not a summary number


def test_missing_mask_arm_and_failed_regression_degrade(results, tmp_path):
    p = results / "c1" / "DreamBase" / "results.json"
    res = json.loads(p.read_text())
    del res["by_mask_kind"]["decoder:topk_conf:k4"]
    res["by_mask_kind"]["uniform"]["regression"] = {"eq2_nofe": {"error": "singular"}}
    p.write_text(json.dumps(res))
    out = run(results, tmp_path, "--no-figures")
    rs = rows(table(out, "c1_main"))
    dream = [r for r in rs if r[1] in ("uniform", "dec. $k{=}4$")][2:4]
    assert dream[1][2:] == [P.DASH] * 6
    assert dream[0][6] == dream[0][7] == P.DASH and dream[0][2] != P.DASH


def test_nothing_at_all_still_writes_every_file(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    out = run(empty, tmp_path, "--no-figures")
    for name in TABLES:
        assert P.DASH in table(out, name)
    assert macro_defs(out / "tables" / "c1" / "macros_c1.tex") == {}
    man = json.loads((out / "tables" / "c1" / "c1_sources.json").read_text())
    assert man["arms_present"] == [] and len(man["arms_missing"]) == 7


def test_an_unreadable_results_file_is_skipped_with_a_warning(results, tmp_path, capsys):
    (results / "e3" / "LladaBase" / "results.json").write_text("{not json")
    out = run(results, tmp_path, "--no-figures")
    assert "unreadable" in capsys.readouterr().err
    assert rows(table(out, "c1_e3"))[0][2] == P.DASH


def test_macros_are_well_formed_unique_and_every_input_exists(results, tmp_path):
    out = run(results, tmp_path, "--no-figures")
    main = out / "tables" / "c1" / "macros_c1.tex"
    text = main.read_text()
    inputs = re.findall(r"^\\input\{([^}]*)\}$", text, re.M)
    assert len(inputs) >= 12
    seen, files = {}, [main]
    for rel in inputs:
        f = out / (rel + ".tex")                      # relative to the paper directory
        assert f.exists(), rel
        files.append(f)
    for f in files:
        for name, body in macro_defs(f).items():
            assert name.isascii() and name.isalpha(), name
            assert name not in seen, f"{name} defined in {f} and {seen[name]}"   # \newcommand would fail
            assert "nan" not in body and "inf" not in body
            seen[name] = f
    mine = macro_defs(main)
    # decisions and counts agree with the summary
    summ = json.loads((results / "c1" / "summary" / "c1_summary.json").read_text())
    rows_ = summ["rows"]
    n_rej = sum(r["contrast"]["holm_reject"] for r in rows_ if r["contrast"]["holm_family"] == "E1")
    assert mine["COneHolmEOneReject"] == str(n_rej)
    assert mine["COneHolmEOneFamily"] == str(sum(r["contrast"]["holm_family"] == "E1" for r in rows_))
    assert mine["COneHolmDecisionEOneLladaBaseDecTopkConfKFour"] == "yes"
    assert mine["COneHolmDecisionEOneDreamInstructMbppUni"] == "no"
    assert mine["COneArmsPresent"] == "7" and mine["COneArmsExpected"] == "7"
    assert mine["COneTimeSplitPresent"] == "1"
    # no duplicate of what the analysis already writes: per-arm names come only from the copies
    assert not any(n.startswith("COneLladaBaseGap") for n in mine)
    assert "COneLladaBaseGapOneUni" in seen and "COneLladaBaseContrastDecTopkConfKFour" in seen
    assert "COneLladaBaseKnotsFiveContrastUni" in seen and "EThreeLladaBaseTypeEffectTopkConfKFourMatched" in seen


def test_macro_inputs_follow_tex_root(results, tmp_path):
    out = tmp_path / "x" / "tables" / "c1"
    P.main(["--results", str(results), "--out-tables", str(out), "--no-figures",
            "--tex-root", str(tmp_path / "x")])
    assert "\\input{tables/c1/macros/c1-LladaBase}" in (out / "macros_c1.tex").read_text()


def test_arm_tag_is_the_one_analyze_traces_uses():
    import analyze_traces
    for arm in ("uniform", "decoder:topk_conf:k4", "decoder:topk_conf:k8", "uniform_matched:topk_conf:k2",
                "decoder"):
        assert P.arm_tag(arm) == analyze_traces.arm_tag(arm)


def test_number_formatting():
    assert P.num(-0.0004) == "$0.000$" and P.num(0.0512) == "$0.051$" and P.num(-1.2, 2) == "$-1.20$"
    assert P.num(float("nan")) == P.num(None) == P.num(float("inf")) == P.DASH
    assert P.num(1.8e6) == "$1.8{\\times}10^{6}$"
    assert P.est_ci({"value": 0.1, "ci": [None, 0.2]}) == "$0.100$"      # no half-interval
    assert P.est_ci(None) == P.DASH
    # rounding that would mislead takes more decimals: an end of an interval excluding 0, or [x, x]
    assert P.est_ci({"value": 0.0015, "ci": [0.00003, 0.003]}) == "$0.00150$ {\\scriptsize[$0.00003$, $0.00300$]}"
    assert P.est_ci({"value": 0.0010, "ci": [0.00098, 0.00106]}) == "$0.0010$ {\\scriptsize[$0.0010$, $0.0011$]}"
    assert P.est_ci({"value": 0.05, "ci": [-0.0002, 0.1]}) == "$0.050$ {\\scriptsize[$0.000$, $0.100$]}"
    assert P.sup("*", "", "\\dagger") == "$^{*,\\dagger}$" and P.sup("") == ""
    assert P.pval(0.0004) == "$<0.001$" and P.pval(0.0321) == "$0.032$"


def test_sources_manifest_carries_hashes_and_provenance(results, tmp_path):
    out = run(results, tmp_path, "--no-figures")
    man = json.loads((out / "tables" / "c1" / "c1_sources.json").read_text())
    assert man["provenance"]["git_commit"] is not None
    src = man["sources"]["c1/LladaBase/results.json"]
    from c3.provenance import sha256_file
    assert src["sha256"] == sha256_file(results / "c1" / "LladaBase" / "results.json")
    assert "c1_main.tex" in man["outputs"]


def test_figures_are_written_as_pdf_and_png(results, tmp_path):
    pytest.importorskip("matplotlib")
    out = run(results, tmp_path)
    fig = out / "figures" / "c1"
    for stem in ("c1_overview", "c1_mechanisms", "c1_delta_m_all", "c1_field_m_all", "c1_delta_m_LladaBase",
                 "c1_reliability_LladaBase", "c1_e3_LladaBase"):
        for ext in ("pdf", "png"):
            assert (fig / f"{stem}.{ext}").stat().st_size > 5_000, stem


def test_figures_degrade_with_missing_arms_and_e3(results, tmp_path):
    pytest.importorskip("matplotlib")
    shutil.rmtree(results / "e3")
    shutil.rmtree(results / "c1" / "LladaBase")
    out = run(results, tmp_path, "--main-arm", "DreamBase")
    fig = out / "figures" / "c1"
    assert (fig / "c1_delta_m_all.pdf").exists() and (fig / "c1_delta_m_DreamBase.pdf").exists()
    assert (fig / "c1_reliability_DreamBase.png").exists()
    assert (fig / "c1_overview.pdf").exists() and (fig / "c1_mechanisms.pdf").exists()
    assert not list(fig.glob("c1_e3_*"))                  # no E3 results, no E3 figure
    shutil.rmtree(results / "c1")
    out2 = run(results, tmp_path / "again")                # nothing: no figure, no crash
    assert not list((out2 / "figures" / "c1").glob("*.pdf"))


def test_native_panel_skips_thin_groups_and_contrast_is_32_minus_1():
    r = {"native": {"rows": [{"level": 32, "delta": 0.1, "n_docs": 40},
                             {"level": 128, "delta": 0.3, "n_docs": P.NATIVE_MIN_DOCS - 1}]},
         "decomposition": [{"m": 1, "field": 0.01}, {"m": 8, "field": 0.5}, {"m": 32, "field": 0.04}]}
    assert P._native_at(r, 32)["delta"] == 0.1
    assert P._native_at(r, 128) is None and P._native_at(None, 32) is None
    assert P._contrast(r, "field") == pytest.approx(0.03)
    assert P._contrast(r, "residual") is None


def test_native_macros_per_level_with_doc_counts():
    rows = [{"level": 32, "delta": 0.0791, "ci_low": 0.07, "ci_high": 0.088, "n_docs": 651},
            {"level": 256, "delta": 0.4, "ci_low": 0.1, "ci_high": 0.7, "n_docs": 2}]
    R = type("R", (), {"labels": ["LladaInstructGsm"],
                       "kind": lambda self, l, k, v=None: {"native": {"rows": rows}} if k.startswith("decoder") else None})()
    m = P.Macros()
    P.native_macros(R, m)
    body = "\n".join(m.lines)
    assert "\\newcommand{\\COneLladaInstructGsmNativeThirtyTwoDecTopkConfKFour}{0.079}" in body
    assert "{\\COneLladaInstructGsmNativeDocsThirtyTwoDecTopkConfKFour}{651}" in body
    assert "TwoFiftySix" not in body                     # 2 documents: below NATIVE_MIN_DOCS



# ---------------------------------------------------------------- claimed, family footnote
def _summary(results):
    return json.loads((results / "c1" / "summary" / "c1_summary.json").read_text())


def _write_summary(results, summ):
    (results / "c1" / "summary" / "c1_summary.json").write_text(json.dumps(summ))


def test_stars_mark_claimed_not_just_holm_rejected(results, tmp_path):
    out = run(results, tmp_path, "--no-figures")
    summ = _summary(results)
    arms = list(P.MAIN_ARMS)
    # the fixture has cells that Holm rejects but that are not declared: the two must differ
    assert any(r["contrast"].get("holm_reject") and not r["contrast"]["claimed"] for r in summ["rows"])
    assert any(r["contrast"].get("claimed") for r in summ["rows"])
    main, col = rows(table(out, "c1_main")), rows(table(out, "c1_main_col"))
    n_star = 0
    for label in P.MAIN_ARMS:
        for i, mask in enumerate(P.PRIMARY_MASKS):
            row = next(r for r in summ["rows"] if r["model"] == label and r["arm"] == mask)
            for key, mcell, ccell in (("contrast", 2, 2), ("field_contrast", 4, 3)):
                want = row[key]["claimed"]
                n_star += want
                assert ("*" in main[2 * arms.index(label) + i][mcell]) == want, (label, mask, key)
                assert ("*" in col[2 * arms.index(label) + i][ccell]) == want, (label, mask, key)
    assert n_star == 4
    # the time-split arm is outside the families: never starred
    assert "*" not in main[-1][2] + main[-2][2]


def test_claimed_is_rebuilt_from_an_older_summary_without_the_field(results, tmp_path):
    summ = _summary(results)
    for row in summ["rows"]:
        for key in ("contrast", "field_contrast"):
            row[key].pop("claimed", None)
    _write_summary(results, summ)
    out = run(results, tmp_path, "--no-figures")
    mine = macro_defs(out / "tables" / "c1" / "macros_c1.tex")
    assert mine["COneClaimedEOneLladaBaseDecTopkConfKFour"] == "yes"
    assert mine["COneClaimedEOneLladaBaseUni"] == "no"          # Holm-rejected but not declared
    assert mine["COneHolmDecisionEOneLladaBaseUni"] == "yes"


def test_claimed_macros_and_the_holm_footnote(results, tmp_path):
    out = run(results, tmp_path, "--no-figures")
    mine = macro_defs(out / "tables" / "c1" / "macros_c1.tex")
    summ = _summary(results)
    prim = [r for r in summ["rows"] if r["contrast"]["holm_family"] == "E1"]
    assert mine["COneConfirmedEOne"] == str(sum(r["contrast"]["claimed"] for r in prim))
    assert mine["COneConfirmedETwo"] == str(sum(r["field_contrast"]["claimed"] for r in prim))
    assert mine["COneClaimedEOneLladaBaseDecTopkConfKFour"] == "yes"
    assert mine["COneClaimedEOneLladaBaseUni"] == "no"
    assert mine["COneHolmM"] == str(prim[0]["contrast"]["holm_m"]) == "14"
    assert mine["COneHolmMissingCount"] == "0" and mine["COneHolmMissing"] == "none"
    note = mine["COneHolmFootnote"]
    assert "$14$" in note and "Missing: none" in note and "0.05" in note


def test_the_footnote_names_the_missing_family_members(results, tmp_path):
    summ = _summary(results)
    summ["holm_missing"] = ["Mdlm:uniform", "DreamBase:decoder:topk_conf:k4"]
    _write_summary(results, summ)
    out = run(results, tmp_path, "--no-figures")
    mine = macro_defs(out / "tables" / "c1" / "macros_c1.tex")
    assert mine["COneHolmMissingCount"] == "2"
    assert mine["COneHolmMissing"] == "MDLM-110M (uniform), Dream Base (dec. $k{=}4$)"
    assert "Missing: MDLM-110M (uniform)" in mine["COneHolmFootnote"]


def test_a_summary_without_the_family_fields_gives_no_footnote_numbers(results, tmp_path):
    summ = _summary(results)
    for k in ("holm_members", "holm_missing"):
        summ.pop(k)
    for row in summ["rows"]:
        for key in ("contrast", "field_contrast"):
            row[key].pop("holm_m", None)
    _write_summary(results, summ)
    mine = macro_defs(run(results, tmp_path, "--no-figures") / "tables" / "c1" / "macros_c1.tex")
    assert "COneHolmM" not in mine and "COneHolmMissing" not in mine
    assert "COneHolmFootnote" in mine and "$" in mine["COneHolmFootnote"] and "cells per contrast" not in mine["COneHolmFootnote"]


# ---------------------------------------------------------------- gamma_2..4
def test_gamma_table_and_macros_follow_the_summary(results, tmp_path):
    out = run(results, tmp_path, "--no-figures")
    summ = _summary(results)
    rs = rows(table(out, "c1_gamma"))
    arms = list(P.MAIN_ARMS) + ["LladaBaseTimeSplit"]
    assert len(rs) == 2 * len(arms)
    mine = macro_defs(out / "tables" / "c1" / "macros_c1.tex")
    fam = rej = rows_rej = 0
    for label in P.MAIN_ARMS:
        for i, mask in enumerate(P.PRIMARY_MASKS):
            row = next(r for r in summ["rows"] if r["model"] == label and r["arm"] == mask)
            cells = rs[2 * arms.index(label) + i]
            tag = label + P.arm_tag(mask)
            any_rej = False
            for j, (g, word) in enumerate(P.GAMMAS.items()):
                c = row["gamma"][g]
                est, ph = cells[2 + 2 * j], cells[3 + 2 * j]
                assert ("*" in est) == c["holm_reject"], (label, mask, g)
                assert (ph == "$<0.001$") if c["p_holm"] < 0.001 else (f"{c['p_holm']:.3f}" in ph)
                assert mine[f"COneHolmDecisionGamma{word}{tag[:len(label)]}{P.arm_tag(mask)}"] == (
                    "yes" if c["holm_reject"] else "no")
                assert mine[f"COneHolmPGamma{word}{label}{P.arm_tag(mask)}"] == f"{c['p_holm']:.4f}"
                assert mine[f"COne{label}Gamma{word}{P.arm_tag(mask)}"] == f"{c['value']:.3f}"
                assert mine[f"COne{label}Gamma{word}{P.arm_tag(mask)}Lo"] == f"{c['ci'][0]:.3f}"
                fam += 1
                rej += c["holm_reject"]
                any_rej = any_rej or c["holm_reject"]
            rows_rej += any_rej
    assert 0 < rej < fam                                    # the fixture has both decisions
    assert (mine["COneHolmGammaFamily"], mine["COneHolmGammaReject"], mine["COneHolmGammaRows"]) == (
        str(fam), str(rej), str(rows_rej))
    assert fam == 3 * 14
    assert rs[-1][2:] == rs[-2][2:] == [P.DASH] * 6        # time-split: no regression, no row


def test_gamma_counts_leave_out_exploratory_arms(results, tmp_path):
    summ = _summary(results)
    expl = [r for r in summ["rows"] if r["contrast"]["holm_family"] == "exploratory"]
    assert expl                                            # decoder k = 2, 8 and matched uniform
    for r in expl:
        r["gamma"]["gamma_2"]["holm_reject"] = True
    _write_summary(results, summ)
    mine = macro_defs(run(results, tmp_path, "--no-figures") / "tables" / "c1" / "macros_c1.tex")
    assert mine["COneHolmGammaFamily"] == str(3 * 14)
    assert not any("DecTopkConfKTwo" in n for n in mine if "Gamma" in n)


def test_gamma_without_a_summary_is_dashes_and_no_macros(results, tmp_path):
    shutil.rmtree(results / "c1" / "summary")
    out = run(results, tmp_path, "--no-figures")
    assert all(r[2:] == [P.DASH] * 6 for r in rows(table(out, "c1_gamma")))
    assert not any("Gamma" in n for n in macro_defs(out / "tables" / "c1" / "macros_c1.tex"))


def test_gamma_counts_undefined_when_the_summary_has_no_gamma(results, tmp_path):
    summ = _summary(results)                               # E1/E2-only runs: gamma is None
    for r in summ["rows"]:
        r["gamma"] = {g: None for g in P.GAMMAS}
    _write_summary(results, summ)
    mine = macro_defs(run(results, tmp_path, "--no-figures") / "tables" / "c1" / "macros_c1.tex")
    assert not any(n.startswith("COneHolmGamma") for n in mine)


# ---------------------------------------------------------------- shared-error term and ICC
def test_shared_error_term_and_icc_macros_and_table(results, tmp_path):
    out = run(results, tmp_path, "--no-figures")
    mine = macro_defs(out / "tables" / "c1" / "macros_c1.tex")
    rs = rows(table(out, "c1_shared"))
    arms = list(P.MAIN_ARMS) + ["LladaBaseTimeSplit"]
    assert len(rs) == 2 * len(arms) and len(rs[0]) == 5
    for label in arms:
        res = json.loads((results / "c1" / label / "results.json").read_text())
        for i, mask in enumerate(P.PRIMARY_MASKS):
            r = res["by_mask_kind"][mask]
            tag = P.arm_tag(mask)
            cells = rs[2 * arms.index(label) + i]
            for j, m in enumerate(P.SHARED_MS):
                row = next(x for x in r["decomposition"] if x["m"] == m)
                assert f"{abs(row['residual']):.3f}" in cells[2 + j]
                nd = P.ci_decimals({"value": row["residual"], "ci": row["residual_ci"]})
                assert mine[f"COne{label}Residual{P._WORDS[m]}{tag}"] == f"{row['residual']:.{nd}f}"
                assert mine[f"COne{label}Residual{P._WORDS[m]}{tag}Lo"] == f"{row['residual_ci'][0]:.{nd}f}"
                assert mine[f"COne{label}Residual{P._WORDS[m]}{tag}Hi"] == f"{row['residual_ci'][1]:.{nd}f}"
            icc = r["icc_residuals"]
            assert f"{abs(icc['icc']):.3f}" in cells[4]
            assert mine[f"COne{label}ICC{tag}"] == f"{icc['icc']:.3f}"
            assert mine[f"COne{label}ICC{tag}Lo"] == f"{icc['ci'][0]:.3f}"
            assert mine[f"COne{label}ICC{tag}Hi"] == f"{icc['ci'][1]:.3f}"
    # one name pattern with the other per-arm macros, and none of these is written by the analysis
    seen = {n for f in (out / "tables" / "c1" / "macros").glob("*.tex") for n in macro_defs(f)}
    assert not any("Residual" in n or "ICC" in n for n in seen)


def test_shared_error_cells_degrade_without_icc_or_residual(results, tmp_path):
    p = results / "c1" / "DreamBase" / "results.json"
    res = json.loads(p.read_text())
    del res["by_mask_kind"]["uniform"]["icc_residuals"]
    for row in res["by_mask_kind"]["uniform"]["decomposition"]:
        row.pop("residual")
    p.write_text(json.dumps(res))
    out = run(results, tmp_path, "--no-figures")
    cells = rows(table(out, "c1_shared"))[2]
    assert cells[2:] == [P.DASH] * 3
    mine = macro_defs(out / "tables" / "c1" / "macros_c1.tex")
    assert not any(n.startswith("COneDreamBaseICCUni") or n.startswith("COneDreamBaseResidual") and n.endswith("Uni")
                   for n in mine)
    assert "COneDreamBaseICCDecTopkConfKFour" in mine


# ---------------------------------------------------------------- additive-calibrator sensitivity
def test_additive_column_and_macros_come_from_the_additive_run(results, tmp_path):
    out = run(results, tmp_path, "--no-figures")
    rs = rows(table(out, "c1_sensitivity"))
    assert len(rs[0]) == 9
    mine = macro_defs(out / "tables" / "c1" / "macros_c1.tex")
    for label, base_row in (("LladaBase", 0), ("LladaInstructGsm", 4)):
        add = json.loads((results / "c1" / f"{label}_additive" / "results.json").read_text())
        prim = json.loads((results / "c1" / label / "results.json").read_text())
        for i, mask in enumerate(P.PRIMARY_MASKS):
            a = add["by_mask_kind"][mask]["field_contrast"]
            b = prim["by_mask_kind"][mask]["field_contrast"]
            assert f"{abs(a['value']):.3f}" in rs[base_row + i][8]
            if i == 1:                                  # uniform E2 is ~0 at 3 digits in both runs
                assert rs[base_row + i][5] != rs[base_row + i][8]   # not the primary run's cell
            tag = P.arm_tag(mask)
            assert mine[f"COneFieldContrast{label}Additive{tag}"] == f"{a['value']:.3f}"
            assert mine[f"COneFieldContrast{label}Additive{tag}Hi"] == f"{a['ci'][1]:.3f}"
            assert mine[f"COneFieldDeclared{label}Additive{tag}"] == ("yes" if a["declared"] else "no")
            assert mine[f"COneFieldAdditiveShift{label}{tag}"] == f"{a['value'] - b['value']:.3f}"
    # the other arms have no additive run: dashes
    assert all(r[8] == P.DASH for r in rs[2:4] + rs[6:])
    assert (mine["COneAdditiveRuns"], mine["COneAdditiveCells"]) == ("2", "4")
    # E1 is the analysis' own macro of the additive run (copied), not a number of this script
    assert "COneLladaBaseAdditiveContrastUni" in {n for f in (out / "tables" / "c1" / "macros").glob("*additive*.tex")
                                                  for n in macro_defs(f)}
    man = json.loads((out / "tables" / "c1" / "c1_sources.json").read_text())
    assert "c1/LladaBase_additive/results.json" in man["sources"]


def test_additive_agreement_flags_a_changed_sign_or_decision(results, tmp_path):
    p = results / "c1" / "LladaBase_additive" / "results.json"
    res = json.loads(p.read_text())
    res["by_mask_kind"]["decoder:topk_conf:k4"]["field_contrast"]["value"] *= -1
    p.write_text(json.dumps(res))
    out = run(results, tmp_path, "--no-figures")
    mine = macro_defs(out / "tables" / "c1" / "macros_c1.tex")
    assert mine["COneFieldAdditiveAgreeLladaBaseDecTopkConfKFour"] == "no"
    assert mine["COneFieldAdditiveAgreeLladaBaseUni"] == "yes"
    assert mine["COneAdditiveAgree"] == "3" and mine["COneAdditiveCells"] == "4"


def test_a_missing_additive_dir_is_a_dash_and_a_stderr_note(results, tmp_path, capsys):
    shutil.rmtree(results / "c1" / "LladaBase_additive")
    out = run(results, tmp_path, "--no-figures")
    err = capsys.readouterr().err
    assert "no additive-calibrator run" in err and "LladaBase," in err
    rs = rows(table(out, "c1_sensitivity"))
    assert rs[0][8] == rs[1][8] == P.DASH and P.DASH not in rs[0][2:8]      # the rest of the row stays
    mine = macro_defs(out / "tables" / "c1" / "macros_c1.tex")
    assert not any("Additive" in n and "LladaBase" in n and "InstructGsm" not in n for n in mine)
    assert mine["COneAdditiveRuns"] == "1"
    shutil.rmtree(results / "c1" / "LladaInstructGsm_additive")
    mine = macro_defs(run(results, tmp_path / "again", "--no-figures") / "tables" / "c1" / "macros_c1.tex")
    assert not any(n.startswith("COneAdditive") or "Additive" in n for n in mine)


def test_knots_agreement_macros_count_cells_with_both_knot_runs(results, tmp_path):
    out = run(results, tmp_path, "--no-figures")
    mine = macro_defs(out / "tables" / "c1" / "macros_c1.tex")

    def fc(run_dir, mask):
        r = json.loads((results / "c1" / run_dir / "results.json").read_text())
        return r["by_mask_kind"][mask]["field_contrast"]

    cells = agree = 0
    for label in sorted(p.name for p in (results / "c1").iterdir() if p.name.endswith("_knots5")):
        label = label[:-len("_knots5")]
        if not (results / "c1" / f"{label}_knots13").exists():
            continue                                    # one knot run only: not a cell
        for mask in P.PRIMARY_MASKS:
            prim = fc(label, mask)
            runs = [fc(f"{label}_knots{k}", mask) for k in (5, 13)]
            cells += 1
            agree += all((x["value"] > 0) == (prim["value"] > 0)
                         and bool(x.get("declared")) == bool(prim.get("declared")) for x in runs)
    assert cells > 0
    assert mine["COneKnotsCells"] == str(cells)
    assert mine["COneKnotsAgree"] == str(agree)
    shutil.rmtree(results / "c1" / "LladaBase_knots13")
    again = macro_defs(run(results, tmp_path / "again", "--no-figures") / "tables" / "c1" / "macros_c1.tex")
    assert "COneKnotsCells" not in again and "COneKnotsAgree" not in again


def test_an_unreadable_additive_file_is_skipped_with_a_warning(results, tmp_path, capsys):
    (results / "c1" / "LladaBase_additive" / "results.json").write_text("{not json")
    out = run(results, tmp_path, "--no-figures")
    assert "unreadable" in capsys.readouterr().err
    assert rows(table(out, "c1_sensitivity"))[0][8] == P.DASH


def test_the_script_does_not_touch_the_frozen_analysis():
    """analyze_traces.py is frozen by the pre-registration; this feature only reads its output."""
    src = (ROOT / "scripts" / "c1_paper_outputs.py").read_text()
    assert "import analyze_traces" not in src and "from analyze_traces" not in src


def test_ci_macros_get_decimals_where_rounding_misleads():
    # [0.0005, 0.0014] would print as [0.001, 0.001]; an interval excluding 0 must not print a 0.000 end
    m = P.Macros()
    P._est_macros(m, "Narrow", {"value": 0.00093, "ci": [0.0005, 0.0014]})
    P._est_macros(m, "NearZero", {"value": -0.0012, "ci": [-0.0021, -0.0003]})
    P._est_macros(m, "Plain", {"value": 0.025, "ci": [0.011, 0.040]})
    got = dict(l[len("\\newcommand{\\"):-1].split("}{") for l in m.lines)
    assert (got["Narrow"], got["NarrowLo"], got["NarrowHi"]) == ("0.0009", "0.0005", "0.0014")
    assert (got["NearZero"], got["NearZeroLo"], got["NearZeroHi"]) == ("-0.0012", "-0.0021", "-0.0003")
    assert (got["Plain"], got["PlainLo"], got["PlainHi"]) == ("0.025", "0.011", "0.040")

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
TABLES = ("c1_main", "c1_main_col", "c1_beta", "c1_sensitivity", "c1_e3", "c1_t8",
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
    for stem in ("c1_delta_m_all", "c1_field_m_all", "c1_delta_m_LladaBase",
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
    assert not list(fig.glob("c1_e3_*"))                  # no E3 results, no E3 figure
    shutil.rmtree(results / "c1")
    out2 = run(results, tmp_path / "again")                # nothing: no figure, no crash
    assert not list((out2 / "figures" / "c1").glob("*.pdf"))


def test_the_script_does_not_touch_the_frozen_analysis():
    """analyze_traces.py is frozen by the pre-registration; this feature only reads its output."""
    src = (ROOT / "scripts" / "c1_paper_outputs.py").read_text()
    assert "import analyze_traces" not in src and "from analyze_traces" not in src

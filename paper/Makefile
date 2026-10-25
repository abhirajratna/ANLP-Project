# One build command: `make` (needs tectonic; fetches TeX packages on first run).
main.pdf: main.tex sections/*.tex refs.bib ../artifacts/probe/macros_t9.tex ../artifacts/c2/macros_t10.tex ../documents/viva-slides/macros_setup.tex
	tectonic main.tex

# C3 outputs (parth, scripts/c3_paper_outputs.py): prerequisites only once they exist.
main.pdf: $(wildcard ../artifacts/c3/macros_c3.tex ../artifacts/t12/t12_matched_speed.tex ../artifacts/t12/t12_sanity_cost.tex ../artifacts/t12/figures/t12_matched_speed.pdf)

.PHONY: clean
clean:
	rm -f main.pdf main.aux main.bbl main.blg main.log main.out

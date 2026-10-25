# One build command: `make` (needs tectonic; fetches TeX packages on first run).
main.pdf: main.tex sections/*.tex refs.bib ../artifacts/probe/macros_t9.tex ../artifacts/c2/macros_t10.tex ../documents/viva-slides/macros_setup.tex
	tectonic main.tex

.PHONY: clean
clean:
	rm -f main.pdf main.aux main.bbl main.blg main.log main.out

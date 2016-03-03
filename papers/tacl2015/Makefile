all:
	touch paper.bbl
	rm paper.bbl
	latex paper.tex
	dvips -Ppdf -o paper-pics.ps paper.dvi
	ps2pdf -dAutoRotatePages=/None paper-pics.ps paper-pics.pdf

	pdflatex paper
	bibtex paper
	pdflatex paper
	pdflatex paper


response: all
	pdflatex our-cover-letter.tex
	pdflatex decision.tex
	pdflatex resubmission-pkg.tex
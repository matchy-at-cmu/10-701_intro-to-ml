# 10-701 Introduction to Machine Learning

Course repo of 2026 Fall10-701 Introduction to Machine Learning.

## Note beforehand

The course materials and hw handouts should all be available on the [course website](https://10-701.github.io/). If in the future any materials are missing, they might be added to this repository for sharing (upon instructor approval, of course).

What will be shared:
- Notes
- HW handout per se
- Aesthetic latex code for writing beautiful hw
- "Recreational" code that verifies some concepts taught in class

What will **not** be shared:
- Solutions to HW
- Exam materials

## Setup

This project uses [uv](https://docs.astral.sh/uv/) to manage Python version and dependencies. To set up the environment, run:

```bash
git clone git@github.com:matchy-at-cmu/10-701_intro-to-ml.git
cd 10-701_intro-to-ml
uv sync
```

Run commands inside the environment with `uv run`. For example, start Jupyter
Lab with:

```bash
uv run jupyter lab
```

## Compile HW handout

CMU uses Gradescope to collect HW submissions. The submitted PDF must retain
the page layout and answer-box positions from the course handout. Compile the
provided template with pdfLaTeX:

```bash
cd hw1/hw1_latex
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The checked-in LaTeX Workshop recipe runs the same pdfLaTeX workflow. Keep the
template's default OT1 text encoding and Computer Modern fonts. The template's
`\usepackage[utf8]{inputenc}` only controls how pdfLaTeX reads the `.tex` source;
it does not change the output font encoding.

Do not compile homework templates with XeLaTeX or LuaLaTeX, and do not load
global font or math-font packages such as `fontspec`, `fontenc`, `libertine`, or
`mathastext`. Their font metrics can change line heights, line breaks, float
placement, or pagination even when the nominal font size is unchanged. Put
personal typography inside the grouped `answer` environment or `\ans{...}`
command defined in `latex/preamble.tex`.

Before submitting or publishing template changes, run the layout regression
test from the repository root:

```bash
uv run pytest tests/test_pdf_layout.py -q
```

The test treats `hw1/hw1.pdf` as the reference image. It compiles the empty
`hw1/hw1_latex/main.tex` with pdfLaTeX in a temporary directory, renders every
page at 144 DPI, and compares the rendered pages. A small per-channel tolerance
accounts for antialiasing differences between independently generated PDFs;
font, color, geometry, page-count, and other visible changes fail the test.

This is a regression test for the public empty template. Run it before adding
answers, or from the public branch after rebasing it onto the latest template
changes.

## Useful links

- [Course Website](https://10-701.github.io/)

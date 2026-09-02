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
the page layout and answer-box positions from the course handout (or risk losing
points).

To ensure a consistent layout, adhere to the following compilation guidelines:

1. Compile only with pdfLaTeX:

    ```bash
    cd hw1/latex
    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
    ```

   Do not use XeLaTeX or LuaLaTeX for the homework templates. They use a
   different Unicode/OpenType font pipeline, which can change font selection,
   metrics, line breaks, and pagination.

2. Keep the course template's default OT1 output encoding and Computer Modern
   fonts. Do not load global font, math-font, or font-encoding packages such as
   `libertine`, `fontspec`, `fontenc`, or `mathastext`.

`latex-commons` contains reusable settings and macros that apply my preferred
answer typography locally without changing the course template layout. From a
homework's `latex/main.tex`, include them with:

```latex
\input{../../latex-commons/preamble.tex}
```

The test suite discovers each `hw*` directory. It expects the course handout at
`hwN/hwN.pdf` and the empty template at `hwN/latex/main.tex`. It compiles the
template with pdfLaTeX, renders every page at 144 DPI, and compares it with the
handout using an approach similar to [Typst's visual regression
tests](https://github.com/typst/typst/tree/main/tests#testing-strategies).
Tiny per-channel differences caused by PDF edge antialiasing are tolerated;
other visible changes fail the test.

Run the tests from the repository root:

```bash
uv run pytest tests/test_pdf_layout.py -q
```

## Useful links

- [Course Website](https://10-701.github.io/)

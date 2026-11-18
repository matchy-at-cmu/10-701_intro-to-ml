Fira Code Regular and Bold are unmodified TrueType fonts from the [Fira Code 6.2 release](https://github.com/tonsky/FiraCode/releases/tag/6.2). See `OFL.txt` for the license.

Generate the pdfLaTeX metrics from this directory with TeX Live's `ttf2tfm`:

```sh
ttf2tfm FiraCode-Regular.ttf -q -p 8r.enc -u FiraCode-Regular.tfm
ttf2tfm FiraCode-Bold.ttf -q -p 8r.enc -u FiraCode-Bold.tfm
ttf2tfm FiraCode-Regular.ttf -q -N -p FiraCode-operators.enc -u FiraCode-operators.tfm
```

The shared answer style uses the stock `8r.enc` encoding with pdfLaTeX's TrueType embedding. Fira Code is selected by `codeblock` and by `\texttt` inside `answer`. No system font installation is needed.

`codeblock` uses a second encoding to select the same Fira Code glyphs that OpenType shaping selects for common operators such as `==`, `!=`, `<=`, `>=`, and `->`. Each shaped operator retains its ASCII text in the PDF for copying and searching.

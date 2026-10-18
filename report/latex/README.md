# LaTeX figures and tables

Ready to `\input{}` into the course template. `preview.tex` compiles them standalone so they can be checked.

| File | Contents | Packages it needs |
|---|---|---|
| `fig_pipeline.tex` | Pipeline overview (TikZ), `\label{fig:pipeline}` | `tikz` with `\usetikzlibrary{positioning,arrows.meta}` |
| `tab_segments.tex` | Customer segments, `\label{tab:segments}` | `booktabs`, `makecell` |
| `tab_product_rules.tex` | Product rules with bank sources, `\label{tab:product-rules}` | `booktabs`, `tabularx`, `array` |
| `tab_scam_flags.tex` | Scam types and red-flag probabilities, `\label{tab:scam-flags}` | `booktabs`, `tabularx`, `array`, `makecell` |
| `tab_features.tex` | All 34 features, may break across pages, `\label{tab:features}` | `booktabs`, `xltabular`, `array` |
| `bank_sources.bib` | BibTeX entries for the bank product pages cited in the product rules table | `url` |

Values come from `config/generator.yaml` and `src/features.py`; update the tables if those change.

```bash
cd report/latex && pdflatex preview && bibtex preview && pdflatex preview && pdflatex preview
```

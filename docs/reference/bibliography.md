# Shared bibliography

The repository-root `references.bib` is the single source of bibliographic
metadata. It is also included in installed distributions under `share/eitlab`.
Lookups work offline and do not import NGSolve. Install the updated dependencies
with `python -m pip install -e .`.

The registry contains five software references. Use component names,
reference keys, or the class and instance lookup methods:

```python
from eitlab import references, main_article, get_reference
from eit_inverse import SciPySolver, CVXPYSolver

print(main_article("scipy").citation())
print(SciPySolver.main_article().doi)
print(SciPySolver().main_article().bibtex())

for article in CVXPYSolver.references():
    print(article.key, article.title, article.url)

all_articles = references()
article = get_reference("virtanen2020scipy")
metadata = dict(article.fields)
```

To add a method, add its entries to `references.bib`, tagging
`eitlab_components = {method_name}` and assigning exactly one primary entry
with `eitlab_primary = {method_name}`. Both fields accept comma-separated
component names. A class can inherit `BibliographyMixin` and set
`bibliography_component = "method_name"`; no duplicate metadata is needed.
Standard BibTeX exports omit these internal tags. Restart Python after editing
the bibliography, because parsed entries are cached per process. Unknown keys
and component names raise `KeyError`.

## Software references

The shared bibliography also includes the software references below. Metadata
and project citation guidance were checked on 16 October 2026. CVXPY recommends
both of its listed papers, so use `references("cvxpy")` to retrieve both.

| Component | Main reference | Component access |
| --- | --- | --- |
| `netgen` | Schöberl (1997), NETGEN; DOI 10.1007/s007910050004 | `ElectrodeDisk.main_article()`, `CylinderModel.main_article()` |
| `ngsolve` | Schöberl (2014), C++11 Implementation of Finite Elements in NGSolve; ASC Report 30/2014 | `ElectricalModel.main_article()` |
| `cvxpy` | Diamond & Boyd (2016), plus Agrawal et al. (2018) | `CVXPYSolver.references()` |
| `scipy` | Virtanen et al. (2020), SciPy 1.0; DOI 10.1038/s41592-019-0686-2 | `SciPySolver.main_article()` |

```python
from eitlab import main_article, references
from eit_inverse import CVXPYSolver, SciPySolver

print(main_article("ngsolve").citation())
print(main_article("netgen").bibtex())
print(SciPySolver.main_article().doi)
for article in CVXPYSolver.references():
    print(article.bibtex())

# All five software references, once each; excludes reconstruction papers.
software_bibtex = "\n".join(article.bibtex() for article in references("software"))
```

`software` is a collection, with no single primary article. Query individual
package names with `main_article`. NGSolve's entry is a technical report with
its institutional URL; the CVXPY JMLR entry uses the journal URL. No DOI is
invented for either entry. Lookup by name works without importing these
optional packages. Class access on FEM classes imports their normal FEM modules.

Sources:

- [NGSolve selected publications](https://ngsolve.org/documentation.html#selected-publications)
- [Netgen publisher record](https://doi.org/10.1007/s007910050004)
- [NGSolve report, TU Wien](https://repositum.tuwien.at/handle/20.500.12708/28346)
- [CVXPY citation guidance](https://www.cvxpy.org/resources/citing/index.html)
- [CVXPY JMLR paper](https://jmlr.org/papers/v17/15-408.html)
- [CVXPY rewriting paper](https://doi.org/10.1080/23307706.2017.1397554)
- [SciPy citation guidance](https://scipy.org/citing-scipy/)

These cite the general software. Record installed versions separately for
reproducibility, and cite additional algorithms or numerical solvers when they
are significant to the work.

[Documentation index](../README.md)

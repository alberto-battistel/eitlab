"""Shared, offline bibliography backed by the root references.bib file."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Mapping


def bibliography_path():
    """Locate the canonical file in a source tree or installed distribution."""
    root = Path(__file__).resolve().parent.parent
    for path in (root / "references.bib", root / "share/eitlab/references.bib",
                 Path(sys.prefix) / "share/eitlab/references.bib"):
        if path.is_file():
            return path
    raise FileNotFoundError("EITLab references.bib is missing; reinstall eitlab")


@dataclass(frozen=True)
class Reference:
    key: str
    fields: Mapping[str, str]

    @property
    def title(self):
        return self.fields["title"].replace("{", "").replace("}", "")

    @property
    def doi(self):
        return self.fields.get("doi", "")

    @property
    def url(self):
        return self.fields.get("url", "")

    def citation(self):
        """Human-readable citation, retaining the complete author list."""
        f = self.fields
        publication = (f.get("journal") or f.get("booktitle") or
                       f.get("institution") or f.get("publisher", ""))
        if f.get("ENTRYTYPE") == "techreport":
            details = " ".join(v for v in (f.get("type", "Report"), f.get("number", "")) if v)
        else:
            issue = f"({f['number']})" if f.get("number") else ""
            details = f.get("volume", "") + issue
        pages = f", {f['pages'].replace('--', '–')}" if f.get("pages") else ""
        return (f"{f.get('author', '')} ({f.get('year', '')}). {self.title}. "
                f"{publication} {details}{pages}. {self.url}").strip()

    def bibtex(self):
        """Standard BibTeX, omitting EITLab's internal component tags."""
        import bibtexparser
        from bibtexparser.bibdatabase import BibDatabase
        database = BibDatabase()
        database.entries = [{"ID": self.key, **{
            k: v for k, v in self.fields.items() if not k.startswith("eitlab_")}}]
        return bibtexparser.dumps(database)


@lru_cache(maxsize=1)
def _registry():
    import bibtexparser
    with bibliography_path().open(encoding="utf-8") as stream:
        entries = bibtexparser.load(stream).entries
    registry = {}
    for entry in entries:
        fields = dict(entry)
        key = fields.pop("ID")
        if key in registry:
            raise ValueError(f"Duplicate bibliography key: {key}")
        registry[key] = Reference(key, MappingProxyType(fields))
    return registry


def _tags(reference, field):
    return {tag.strip().casefold() for tag in reference.fields.get(field, "").split(",")
            if tag.strip()}


def get_reference(key):
    """Get metadata by BibTeX key; unknown keys raise KeyError."""
    return _registry()[key]


def references(component=None):
    """All articles, or articles tagged for a component (case insensitive).

    Unknown component names raise KeyError to expose misspellings.
    """
    articles = tuple(_registry().values())
    if component is None:
        return articles
    selected = tuple(r for r in articles if component.casefold() in _tags(r, "eitlab_components"))
    if not selected:
        raise KeyError(f"Unknown bibliography component: {component}")
    return selected


def main_article(component):
    """Return the single primary article tagged for a component."""
    selected = [r for r in references(component) if component.casefold() in _tags(r, "eitlab_primary")]
    if len(selected) != 1:
        raise ValueError(f"Expected one primary article for {component!r}; found {len(selected)}")
    return selected[0]


class BibliographyMixin:
    """Components opt in by setting bibliography_component to their BibTeX tag."""
    bibliography_component = None

    @classmethod
    def references(cls):
        if cls.bibliography_component is None:
            raise ValueError("Set bibliography_component on the component class")
        return references(cls.bibliography_component)

    @classmethod
    def main_article(cls):
        if cls.bibliography_component is None:
            raise ValueError("Set bibliography_component on the component class")
        return main_article(cls.bibliography_component)

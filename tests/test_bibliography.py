"""Bibliographic metadata, component lookup and standard BibTeX export."""
import subprocess
import sys

import bibtexparser
import pytest

from eitlab import get_reference, main_article, references
from eit_inverse import SciPySolver


def test_method_metadata_and_class_instance_lookup():
    original = main_article("SCIPY")
    assert original.key == "virtanen2020scipy"
    assert original.doi == "10.1038/s41592-019-0686-2"
    assert len(original.fields["author"].split(" and ")) == 35
    assert original.fields["pages"] == "261--272"
    assert "Schöberl" in main_article("netgen").citation()
    assert {"schoeberl1997netgen", "virtanen2020scipy"} <= {r.key for r in references()}
    assert SciPySolver.main_article() is original
    assert SciPySolver.references() == references("scipy")
    assert SciPySolver().main_article() is original
    with pytest.raises(TypeError):
        original.fields["doi"] = "wrong"


def test_bibtex_export_round_trip_and_unknown_keys():
    for reference in references():
        entry, = bibtexparser.loads(reference.bibtex()).entries
        assert entry["ID"] == reference.key
        assert entry.get("doi", "") == reference.doi
        assert not any(key.startswith("eitlab_") for key in entry)
        assert get_reference(reference.key) is reference
    with pytest.raises(KeyError):
        references("typo")
    with pytest.raises(KeyError):
        get_reference("unknown")


def test_bibliography_top_level_without_fem_or_inverse_imports():
    subprocess.run([sys.executable, "-c", '''
import sys
from eitlab import main_article
assert main_article("scipy").doi
assert "ngsolve" not in sys.modules
assert "eit_inverse" not in sys.modules
'''], check=True)


def test_software_metadata_export_and_backend_access():
    from eit_inverse import SciPySolver, CVXPYSolver
    assert main_article("netgen").doi == "10.1007/s007910050004"
    report = main_article("ngsolve")
    assert report.fields["ENTRYTYPE"] == "techreport"
    assert "ASC Report 30/2014" in report.citation()
    assert "TU Wien" in report.citation()
    assert report.doi == ""
    assert report.url.startswith("https://repositum.tuwien.at/")
    assert SciPySolver.main_article().doi == "10.1038/s41592-019-0686-2"
    assert len(SciPySolver.main_article().fields["author"].split(" and ")) == 35
    assert CVXPYSolver().main_article().key == "diamond2016cvxpy"
    assert {r.key for r in CVXPYSolver.references()} == {"diamond2016cvxpy", "agrawal2018rewriting"}
    software = references("software")
    assert len({r.key for r in software}) == 5
    assert len(bibtexparser.loads("\n".join(r.bibtex() for r in software)).entries) == 5
    with pytest.raises(ValueError, match="primary"):
        main_article("software")


@pytest.mark.fem
def test_fem_and_geometry_reference_access():
    pytest.importorskip("ngsolve")
    from eitlab import ElectricalModel, ElectrodeDisk, CylinderModel
    assert ElectricalModel.main_article() is main_article("ngsolve")
    assert ElectrodeDisk.main_article() is main_article("netgen")
    assert CylinderModel.main_article() is main_article("netgen")

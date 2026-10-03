from bdopt import IDF
from conftest import MODELS


def test_roundtrip_real_model(tmp_path):
    idf = IDF.from_file(MODELS / "Office_IDF.idf")
    assert len(idf) > 400
    assert str(idf.version) == "9.6.0"
    out = idf.save(tmp_path / "x.idf")
    again = IDF.from_file(out)
    assert [(o.class_name, o.fields) for o in idf.objects] == [(o.class_name, o.fields) for o in again.objects]


def test_find_and_edit():
    idf = IDF.from_file(MODELS / "Office_IDF.idf")
    mats = idf.find("material", "_InsulationEX_.01")
    assert len(mats) == 1
    mats[0].set(2, 0.123)
    assert idf.find("Material", "_InsulationEX_.01")[0].fields[2] == "0.123"
    assert len(idf.find("Lights", "*")) == len(idf.by_class("LIGHTS"))
    assert idf.find("Lights", "*", where=lambda o: False) == []


def test_text_parsing_comments_and_missing_semicolon():
    idf = IDF.from_text("Version, 9.6;  ! comment\nZone,\n  Z1,  !- name\n  0;\nBuilding, B1")
    assert [o.class_name for o in idf.objects] == ["Version", "Zone", "Building"]
    assert idf.objects[1].fields == ["Z1", "0"]
    assert idf.objects[2].fields == ["B1"]


def test_eppy_style_access():
    idf = IDF.from_file(MODELS / "lab_building.idf")
    win = [w for w in idf.idfobjects["FenestrationSurface:Detailed"] if w.obj[2] == "Window"]
    assert win and win[0].Name == win[0].fields[0]


def test_add_and_remove():
    idf = IDF.from_text("Version,9.6;")
    o = idf.add("Output:SQLite", "SimpleAndTabular")
    assert idf.by_class("output:sqlite") == [o]
    idf.remove([o])
    assert not idf.by_class("Output:SQLite")

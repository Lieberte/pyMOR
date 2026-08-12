import importlib.util
import json
import zipfile
from pathlib import Path


SCRIPT = Path(__file__).parents[3] / "scripts" / "rename_motor_dataset_cases.py"
SPEC = importlib.util.spec_from_file_location("rename_motor_dataset_cases", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_transform_json_preserves_provenance_and_renames_references():
    mapping = {"2500": "tutorial-a"}
    source = {
        "parentCaseId": "2500",
        "caseId": "2500-stator",
        "path": "../../../cases/2500-stator",
        "metadata": {
            "parentCaseIds": ["2500"],
            "sourceCase": "2500",
            "resultFile": r"F:\source\2500\2500.h5",
        },
        "interfaces": [{
            "componentCaseId": "2500-shaft",
            "oppositeComponentCaseId": "2500-disk2",
        }],
    }

    updated = MODULE.transform_json(source, mapping)

    assert updated["parentCaseId"] == "tutorial-a"
    assert updated["caseId"] == "tutorial-a-stator"
    assert updated["path"] == "../../../cases/tutorial-a-stator"
    assert updated["metadata"]["parentCaseIds"] == ["tutorial-a"]
    assert updated["metadata"]["sourceCase"] == "2500"
    assert updated["metadata"]["resultFile"] == r"F:\source\2500\2500.h5"
    assert updated["interfaces"][0]["componentCaseId"] == "tutorial-a-shaft"
    assert updated["interfaces"][0]["oppositeComponentCaseId"] == "tutorial-a-disk2"


def test_rename_dataset_moves_directories_without_rewriting_npz(tmp_path):
    root = tmp_path / "dataset"
    mapping = {"2500": "tutorial-a"}
    components = ("stator", "magnets", "shaft", "rotor_core", "disk1", "housing", "winding", "disk2")
    for view in ("point", "graph"):
        component_root = root / view / "motors" / "motor-htc-prototype" / "cases"
        whole_root = root / view / "whole_motor" / "cases"
        for component in components:
            case_dir = component_root / f"2500-{component}"
            case_dir.mkdir(parents=True)
            with zipfile.ZipFile(case_dir / "data.npz", "w") as archive:
                archive.writestr("array.npy", b"unchanged-array-data")
            (case_dir / "manifest.json").write_text(json.dumps({
                "caseId": f"2500-{component}",
                "parentCaseId": "2500",
                "metadata": {"sourceCase": "2500"},
            }), encoding="utf-8")
        whole_dir = whole_root / "2500"
        whole_dir.mkdir(parents=True)
        component_entries = [{
            "caseId": f"2500-{component}",
            "path": f"../../../motors/motor-htc-prototype/cases/2500-{component}",
        } for component in components]
        (whole_dir / "manifest.json").write_text(json.dumps({
            "parentCaseId": "2500",
            "components": component_entries,
        }), encoding="utf-8")
        index_entries = [{
            "caseId": f"2500-{component}",
            "parentCaseId": "2500",
            "path": f"motors/motor-htc-prototype/cases/2500-{component}",
        } for component in components]
        (root / view / "dataset.json").write_text(json.dumps({"cases": index_entries}), encoding="utf-8")
        (root / view / "whole_motor" / "dataset.json").write_text(json.dumps({"cases": [{
            "parentCaseId": "2500", "path": "cases/2500",
        }]}), encoding="utf-8")
    (root / "views.json").write_text(json.dumps({"parentCaseIds": ["2500"]}), encoding="utf-8")

    before = (root / "point" / "motors" / "motor-htc-prototype" / "cases" / "2500-stator" / "data.npz").read_bytes()
    result = MODULE.rename_dataset(root, mapping, apply=True)
    renamed = root / "point" / "motors" / "motor-htc-prototype" / "cases" / "tutorial-a-stator"

    assert result["directoryRenames"] == 18
    assert renamed.is_dir()
    assert (renamed / "data.npz").read_bytes() == before
    manifest = json.loads((renamed / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["caseId"] == "tutorial-a-stator"
    assert manifest["parentCaseId"] == "tutorial-a"
    assert manifest["metadata"]["sourceCase"] == "2500"

    validation = MODULE.validate_dataset(root, mapping)
    assert validation == {
        "jsonFiles": 23,
        "componentCases": 16,
        "wholeMotorCases": 2,
        "npzArchives": 16,
        "assetsDirectories": 0,
    }

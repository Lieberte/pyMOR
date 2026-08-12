import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[3] / 'scripts' / 'merge_motor_disk_component_type.py'
SPEC = importlib.util.spec_from_file_location('merge_motor_disk_component_type', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_transform_merges_type_but_preserves_instance_identity():
    source = {
        'caseId': 'tutorial-a-disk1',
        'componentType': 'disk1',
        'geometryId': 'disk1-mesh',
        'metadata': {'regionName': 'disk1_mm'},
        'interface': {
            'componentCaseId': 'tutorial-a-disk2',
            'componentType': 'disk2',
        },
    }

    updated = MODULE.transform(source)

    assert updated['caseId'] == 'tutorial-a-disk1'
    assert updated['geometryId'] == 'disk1-mesh'
    assert updated['componentType'] == 'magnetic_isolation_plate'
    assert updated['componentInstance'] == 'disk1'
    assert updated['metadata']['componentInstance'] == 'disk1'
    assert updated['interface']['componentType'] == 'magnetic_isolation_plate'
    assert updated['interface']['componentInstance'] == 'disk2'


def test_transform_upgrades_existing_disk_semantic_type():
    updated = MODULE.transform({
        'componentType': 'disk',
        'componentInstance': 'disk2',
    })

    assert updated == {
        'componentType': 'magnetic_isolation_plate',
        'componentInstance': 'disk2',
    }

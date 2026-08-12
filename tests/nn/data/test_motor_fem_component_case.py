import json
import shutil
from dataclasses import replace
from zipfile import ZIP_STORED, ZipFile

import numpy as np
import pytest

from mor.nn.data import (
    buildMotorDatasetFromSta,
    buildMotorDatasetFromDirectories,
    buildFullResolutionMotorDatasetFromDirectories,
    buildMotorFemComponentCase,
    loadThermalComponentDataset,
    saveThermalComponentDataset,
    saveWholeMotorDataset,
)
from mor.nn.data.motorFemComponentCase import (
    _componentTypeName,
    _fixedMaterialConductivityTensor,
    _sampleByGroupWithConservedWeight,
)


h5py = pytest.importorskip('h5py')


@pytest.mark.parametrize(
    ('instance', 'componentType'),
    [
        ('disk1', 'magnetic_isolation_plate'),
        ('disk2', 'magnetic_isolation_plate'),
        ('disk_3', 'magnetic_isolation_plate'),
        ('shaft', 'shaft'),
    ],
)
def test_component_type_groups_disk_instances(instance, componentType):
    assert _componentTypeName(instance) == componentType


def test_fixed_anisotropic_conductivity_is_kept_as_a_tensor(tmp_path):
    path = tmp_path / 'materials.h5'
    expected = np.array(
        [
            [12.0, 1.0, 0.5],
            [1.0, 8.0, 0.25],
            [0.5, 0.25, 5.0],
        ]
    )
    names = ('XX', 'XY', 'XZ', 'YX', 'YY', 'YZ', 'ZX', 'ZY', 'ZZ')
    with h5py.File(path, 'w') as file:
        anisotropic = file.create_group('material/thermalConductivity/anisotropic')
        for name, value in zip(names, expected.reshape(-1)):
            anisotropic.create_group(name).create_dataset('fixedValue', data=[value])

    with h5py.File(path, 'r') as file:
        conductivity = _fixedMaterialConductivityTensor(file, 'material')

    np.testing.assert_allclose(conductivity, expected)


def test_grouped_subsampling_preserves_heat_source_region_volumes():
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    heatSources = np.array([0.0, 0.0, 5.0, 5.0])
    _, groupIds = np.unique(heatSources, return_inverse=True)

    selection, selectedWeights = _sampleByGroupWithConservedWeight(
        weights, groupIds, maximumCount=2, randomSeed=3
    )

    np.testing.assert_allclose(selectedWeights.sum(), weights.sum())
    np.testing.assert_allclose(
        np.dot(heatSources[selection], selectedWeights),
        np.dot(heatSources, weights),
    )


def _writeScalar(group, name, value):
    propertyGroup = group.create_group(name)
    isotropic = propertyGroup.create_group('isotropic')
    isotropic.create_dataset('fixedValue', data=[value])


def _makeSolvedCase(root):
    case = root / '2500'
    meshDirectory = case / 'mesh'
    resultDirectory = case / 'results' / 'system'
    meshDirectory.mkdir(parents=True)
    resultDirectory.mkdir(parents=True)

    control = {
        'rootSystem': {
            'thermalComponents': {
                'fem3DThermals': {
                    'system': {
                        'femRegions': {
                            'region': {
                                'name': 'part_mm',
                                'meshUuid': 'mesh',
                                'materialUuid': 'material',
                                'initialConditions': {'temperature': 293.15},
                                'patches': {
                                    'surface': {
                                        'name': 'surface',
                                        'children': {
                                            'surface_a': {
                                                'uuid': 'surface_a',
                                                'name': 'surface_a',
                                                'normalBoundaryEnabled': True,
                                                'networkBoundaryEnabled': False,
                                                'tccEnabled': False,
                                            },
                                            'surface_b': {
                                                'uuid': 'surface_b',
                                                'name': 'surface_b',
                                                'normalBoundaryEnabled': False,
                                                'networkBoundaryEnabled': False,
                                                'tccEnabled': True,
                                            },
                                        },
                                    }
                                },
                                'volumePatches': {
                                    'volumePatch': {
                                        'children': {
                                            'body': {
                                                'name': 'part_body',
                                                'volumetricGroups': {
                                                    'heatSource': {
                                                        'inputVariable': 'power',
                                                        'heatSource1D': {
                                                            'type': 'fixedValue',
                                                            'fixedValue': 12.0,
                                                        },
                                                    }
                                                },
                                            }
                                        }
                                    }
                                },
                            }
                        }
                    }
                }
            }
        }
    }
    (case / 'controlData.json').write_text(json.dumps(control), encoding='utf-8')

    with h5py.File(case / 'controlData.h5', 'w') as file:
        material = file.create_group('material')
        _writeScalar(material, 'density', 7800.0)
        _writeScalar(material, 'specificHeatCapacity', 500.0)
        _writeScalar(material, 'thermalConductivity', 40.0)

    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    surfaces = np.array(
        [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64
    )
    with h5py.File(meshDirectory / 'mesh.samesh', 'w') as file:
        file.create_dataset('nodes', data=nodes.reshape(-1))
        file.create_dataset('connectivityVolume', data=np.array([0, 1, 2, 3]))
        file.create_dataset('connectivitySurface', data=surfaces.reshape(-1))
        patches = file.create_group('patches').create_group('surface')
        patches.create_dataset('surface_a', data=[0, 1])
        patches.create_dataset('surface_b', data=[2, 3])
        volumes = file.create_group('volumes')
        volumes.create_dataset('body', data=[0])

    resultBase = 'thermalComponents/thermal3dFem/system/region/3dResults/results'
    with h5py.File(resultDirectory / '2500.h5', 'w') as file:
        group = file.create_group(resultBase)
        group.create_dataset('temperature', data=[300.0, 301.0, 302.0, 303.0])
        group.create_dataset('heatFlowRate', data=[-0.5, -1.0, -1.5, -3.0])
        overview = file.create_group(
            'thermalComponents/thermal3dFem/system/region/overviewResults/results'
        )
        overview.create_dataset('heatFlowRate', data=[5.0, 7.0])
        locations = file.create_group(
            'thermalComponents/thermal3dFem/system/region/overviewResults/locationIDs'
        )
        locations.create_dataset('heatFlowRate', data=[b'surface_a', b'surface_b'])
    return case


def test_solved_motor_fem_region_becomes_closed_component_case(tmp_path):
    case = buildMotorFemComponentCase(_makeSolvedCase(tmp_path), 'part')

    assert case.caseId == '2500rpm-part'
    assert case.geometryId == 'part-mesh'
    assert case.componentType == 'part'
    assert case.metadata['componentInstance'] == 'part'
    assert case.volume.coordinatesMeters.shape == (1, 3)
    assert case.boundary.coordinatesMeters.shape == (8, 3)
    assert case.temperature.coordinatesMeters.shape == (4, 3)
    np.testing.assert_allclose(case.volume.weightsM3.sum(), 1.0 / 6.0)
    np.testing.assert_allclose(
        np.dot(case.volume.heatSourceWPerM3, case.volume.weightsM3), 12.0
    )
    np.testing.assert_allclose(case.volume.totalHeatSourcePowerW, 12.0)
    np.testing.assert_allclose(
        np.dot(case.boundary.heatFluxWPerM2, case.boundary.weightsM2), -12.0
    )
    np.testing.assert_allclose(case.boundary.totalHeatFlowRateW, -12.0)
    np.testing.assert_allclose(
        list(case.boundary.heatFlowRateWByPatch().values()), [-5.0, -7.0]
    )
    assert case.metadata['surfacePatchCount'] == 2
    np.testing.assert_allclose(
        [patch['sampledHeatFlowErrorW'] for patch in case.metadata['surfacePatches']],
        0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(case.metadata['boundaryHeatFlowScale'], 2.0)
    np.testing.assert_allclose(case.temperature.temperatureK, [300.0, 301.0, 302.0, 303.0])
    np.testing.assert_allclose(case.boundary.ambientTemperatureK, 0.0)
    assert case.metadata['absoluteTemperatureAnchorAvailable'] is False
    assert 'neutralBoundaryTemperatureK' not in case.metadata


def test_volume_and_query_subsampling_preserve_integration_weights(tmp_path):
    case = buildMotorFemComponentCase(
        _makeSolvedCase(tmp_path),
        'part',
        maximumVolumeSamples=1,
        maximumBoundarySamples=3,
        maximumQuerySamples=2,
        randomSeed=7,
    )

    assert case.temperature.coordinatesMeters.shape == (2, 3)
    assert case.boundary.coordinatesMeters.shape == (3, 3)
    assert set(case.boundary.patchIds) == {0, 1}
    np.testing.assert_allclose(case.volume.weightsM3.sum(), 1.0 / 6.0)
    np.testing.assert_allclose(
        np.dot(case.boundary.heatFluxWPerM2, case.boundary.weightsM2), -12.0
    )
    np.testing.assert_allclose(
        list(case.boundary.heatFlowRateWByPatch().values()), [-5.0, -7.0]
    )
    np.testing.assert_allclose(case.temperature.weightsM3.sum(), 1.0 / 6.0)


def test_sta_archive_builds_one_dataset_with_independent_component_cases(tmp_path):
    solvedCase = _makeSolvedCase(tmp_path / 'source')
    archivePath = tmp_path / 'single_motor.sta'
    with ZipFile(archivePath, 'w', compression=ZIP_STORED, allowZip64=True) as archive:
        for path in solvedCase.rglob('*'):
            if path.is_file():
                archive.write(path, path.relative_to(solvedCase).as_posix())

    build = buildMotorDatasetFromSta(
        archivePath,
        tmp_path / 'dataset',
        motorId='motor-a',
        parentCaseId='baseline',
        maximumVolumeSamples=1,
        maximumBoundarySamples=4,
        maximumQuerySamples=2,
    )
    cases = loadThermalComponentDataset(tmp_path / 'dataset')
    index = json.loads(build.indexPath.read_text(encoding='utf-8'))
    wholeIndex = json.loads(build.wholeMotorIndexPath.read_text(encoding='utf-8'))
    wholeManifest = json.loads(
        (
            tmp_path
            / 'dataset'
            / 'whole_motor'
            / 'cases'
            / 'baseline'
            / 'manifest.json'
        ).read_text(encoding='utf-8')
    )
    views = json.loads(build.viewsPath.read_text(encoding='utf-8'))

    assert build.motorId == 'motor-a'
    assert build.parentCaseId == 'baseline'
    assert build.resultMember == 'results/system/2500.h5'
    assert build.componentCaseIds == ('baseline-part',)
    assert build.reusedExtractedDirectory is None
    assert len(cases) == 1
    assert cases[0].caseId == 'baseline-part'
    assert 'speedRpm' not in cases[0].operatingFeatures
    assert cases[0].metadata['motorId'] == 'motor-a'
    assert cases[0].metadata['parentCaseId'] == 'baseline'
    assert cases[0].metadata['resultFile'] == 'results/system/2500.h5'
    assert index['metadata']['datasetType'] == 'singleMotorFemComponents'
    assert index['metadata']['selectedComponents'] == ['part']
    assert index['featureSchema']['boundaryTypeNames'] == {
        '0': 'resultHeatFlux',
        '1': 'convection',
        '2': 'fixedTemperature',
    }
    assert wholeIndex['viewType'] == 'wholeMotor'
    assert wholeManifest['components'][0]['caseId'] == 'baseline-part'
    assert wholeManifest['components'][0]['path'] == '../../../cases/baseline-part'
    assert wholeManifest['inputPolicy']['internalInterfaceHeatFlowAndTemperature'] == (
        'supervision targets only'
    )
    assert views['views'] == {
        'component': 'dataset.json',
        'wholeMotor': 'whole_motor/dataset.json',
    }


def test_whole_motor_view_pairs_component_interface_patches(tmp_path):
    base = buildMotorFemComponentCase(_makeSolvedCase(tmp_path / 'source'), 'part')

    def interfaceCase(caseId, componentType, regionUuid, oppositeRegionUuid, heatFlow):
        patches = [dict(patch) for patch in base.metadata['surfacePatches']]
        patches[0].update(
            {
                'interfaceUuid': 'interface-ab',
                'interfaceName': 'a/b',
                'oppositeRegionUuid': oppositeRegionUuid,
                'oppositePatchUuid': 'patch-opposite',
                'contactConductanceWPerM2K': 1000.0,
                'expectedHeatFlowRateW': heatFlow,
                'sampledReconstructedHeatFlowRateW': heatFlow,
            }
        )
        return replace(
            base,
            caseId=caseId,
            componentType=componentType,
            geometryId=f'{componentType}-geometry',
            metadata={
                **base.metadata,
                'regionUuid': regionUuid,
                'surfacePatches': patches,
            },
        )

    cases = (
        interfaceCase('baseline-a', 'a', 'region-a', 'region-b', -5.0),
        interfaceCase('baseline-b', 'b', 'region-b', 'region-a', 5.0),
    )
    datasetRoot = tmp_path / 'dataset'
    saveThermalComponentDataset(cases, datasetRoot)
    wholeIndex = saveWholeMotorDataset(
        cases,
        datasetRoot / 'whole_motor',
        componentDatasetDirectory=datasetRoot,
        motorId='motor-a',
        parentCaseId='baseline',
    )
    manifest = json.loads(
        (wholeIndex.parent / 'cases' / 'baseline' / 'manifest.json').read_text(
            encoding='utf-8'
        )
    )

    assert manifest['summary']['componentCount'] == 2
    assert manifest['summary']['interfaceCount'] == 1
    assert manifest['interfaces'][0]['isComplete'] is True
    assert len(manifest['interfaces'][0]['sides']) == 2
    np.testing.assert_allclose(
        manifest['interfaces'][0]['resultTargetHeatFlowBalanceW'], 0.0
    )


def test_numeric_motor_directories_build_multiple_parent_cases(tmp_path):
    sourceRoot = tmp_path / 'source'
    firstCase = _makeSolvedCase(sourceRoot)
    shutil.copytree(firstCase, sourceRoot / '5000')

    build = buildMotorDatasetFromDirectories(
        sourceRoot,
        tmp_path / 'dataset',
        motorId='motor-a',
        maximumVolumeSamples=1,
        maximumBoundarySamples=4,
        maximumQuerySamples=2,
    )
    componentCases = loadThermalComponentDataset(tmp_path / 'dataset')
    wholeIndex = json.loads(build.wholeMotorIndexPath.read_text(encoding='utf-8'))

    assert build.parentCaseIds == ('2500', '5000')
    assert build.componentCaseIds == ('2500-part', '5000-part')
    assert len(componentCases) == 2
    assert {case.caseId for case in componentCases} == {'2500-part', '5000-part'}
    assert [entry['parentCaseId'] for entry in wholeIndex['cases']] == ['2500', '5000']


def test_full_resolution_dataset_keeps_point_and_graph_views(tmp_path):
    sourceRoot = tmp_path / 'source'
    _makeSolvedCase(sourceRoot)
    outputRoot = tmp_path / 'full'

    viewsPath = buildFullResolutionMotorDatasetFromDirectories(
        sourceRoot,
        outputRoot,
        motorId='motor-a',
    )
    views = json.loads(viewsPath.read_text(encoding='utf-8'))
    pointIndex = json.loads((outputRoot / views['views']['point']).read_text(encoding='utf-8'))
    graphIndex = json.loads((outputRoot / views['views']['graph']).read_text(encoding='utf-8'))
    pointCase = outputRoot / 'point' / pointIndex['cases'][0]['path']
    graphCase = outputRoot / 'graph' / graphIndex['cases'][0]['path']
    with np.load(pointCase / 'data.npz') as arrays:
        assert arrays['volumeCoordinatesMeters'].shape == (1, 3)
        assert arrays['queryCoordinatesMeters'].shape == (4, 3)
        assert 'tetrahedra' not in arrays
        np.testing.assert_allclose(arrays['nodeTemperatureK'], [300, 301, 302, 303])
    with np.load(graphCase / 'data.npz') as arrays:
        np.testing.assert_array_equal(arrays['tetrahedra'], [[0, 1, 2, 3]])
        assert arrays['surfaceTriangles'].shape == (4, 3)
        assert arrays['nodeHeatSourceWPerM3'].shape == (4,)
    assert pointIndex['metadata']['randomSubsampling'] is False
    assert graphIndex['connectivityStored'] is True
    assert not (outputRoot / 'point' / 'assets').exists()
    assert not (outputRoot / 'graph' / 'assets').exists()

import json
import shutil
from dataclasses import replace
from zipfile import ZIP_STORED, ZipFile

import numpy as np
import pytest

from mor.nn.data import (
    buildMotorDatasetFromSta,
    buildMotorTransientSeriesFromSta,
    buildMotorDatasetFromDirectories,
    buildFullResolutionMotorDatasetFromDirectories,
    buildMotorFemComponentCase,
    loadThermalComponentDataset,
    loadMotorTransientSeriesRecords,
    saveThermalComponentDataset,
    saveWholeMotorDataset,
    writeMotorThermalAnalysisIndexes,
)
from mor.nn.data.motorFemComponentCase import (
    PRESCRIBED_HEAT_FLUX_BOUNDARY_TYPE,
    _boundaryNodeGeometry,
    _componentTypeName,
    _fixedMaterialConductivityTensor,
    _normalizedComponentName,
    _sampleByGroupWithConservedWeight,
    _thermalAnalysisMetadata,
)
from mor.nn.data.motorStaDataset import _selectedResultMember


h5py = pytest.importorskip('h5py')


def test_boundary_normal_falls_back_to_largest_face_when_nodal_vectors_cancel():
    nodes = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    triangles = np.asarray([[0, 1, 2], [0, 2, 1]], dtype=np.int64)

    boundaryNodes, normals, weights = _boundaryNodeGeometry(
        nodes, triangles, np.asarray([0.0, 0.0, 0.0])
    )

    np.testing.assert_array_equal(boundaryNodes, [0, 1, 2])
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0)
    assert np.all(weights > 0.0)


def test_iteration_info_h5_files_are_not_mistaken_for_solved_result_fields(tmp_path):
    archivePath = tmp_path / 'iteration-results.sta'
    with ZipFile(archivePath, 'w', compression=ZIP_STORED) as archive:
        for time, iterations in ((0.1, (1, 2, 20)), (0.2, (1, 7)), (10.0, (1, 3))):
            for iteration in iterations:
                archive.writestr(
                    f'results/iterationInfo/{time:.6f}/{iteration:.6f}.h5', b'result'
                )

    with ZipFile(archivePath) as archive, pytest.raises(
        ValueError, match='iteration metadata only'
    ):
        _selectedResultMember(archive, resultTime=None, resultMember=None)


def test_result_selection_prefers_the_resolved_fem_child_system(tmp_path):
    archivePath = tmp_path / 'child-system-results.sta'
    with ZipFile(archivePath, 'w', compression=ZIP_STORED) as archive:
        archive.writestr('results/root-system/10.000000.h5', b'root')
        archive.writestr('results/child-system/2.000000.h5', b'child-2')
        archive.writestr('results/child-system/5.000000.h5', b'child-5')

    with ZipFile(archivePath) as archive:
        selected, selectedTime, times, members = _selectedResultMember(
            archive,
            resultTime=None,
            resultMember=None,
            systemUuid='child-system',
        )

    assert selected == 'results/child-system/5.000000.h5'
    assert selectedTime == 5.0
    assert times == (2.0, 5.0)
    assert members == (
        'results/child-system/2.000000.h5',
        'results/child-system/5.000000.h5',
    )


@pytest.mark.parametrize(
    ('instance', 'componentType'),
    [
        ('disk1', 'magnetic_isolation_plate'),
        ('disk2', 'magnetic_isolation_plate'),
        ('disk_3', 'magnetic_isolation_plate'),
        ('rotor_shaft', 'shaft'),
        ('cap2', 'end_cap'),
        ('insulating_paper2', 'insulation'),
        ('bearing_ball_v2', 'bearing'),
        ('magnet', 'magnets'),
        ('stator_fe3', 'stator'),
        ('shell', 'housing'),
        ('coil_coarse200_final3', 'winding'),
        ('rotor_fe', 'rotor_core'),
        ('rt_steel_sleeve', 'sleeve'),
        ('shaft', 'shaft'),
    ],
)
def test_component_type_groups_disk_instances(instance, componentType):
    assert _componentTypeName(instance) == componentType


@pytest.mark.parametrize(
    ('regionName', 'componentInstance', 'componentType'),
    (
        ('转子铁芯', 'rotor_core', 'rotor_core'),
        ('端板2', 'end_plate_2', 'end_plate'),
        ('绕组皇冠端', 'winding_crown_end', 'winding'),
        ('隔磁板', 'magnetic_isolation_plate', 'magnetic_isolation_plate'),
        ('支架轴承', 'carrier_bearing', 'bearing'),
        ('支架轴承2', 'carrier_bearing_2', 'bearing'),
        ('轴承支架', 'bearing_support', 'bearing'),
        ('轴承内圈', 'bearing_inner_ring', 'bearing'),
        ('转轴', 'shaft', 'shaft'),
        ('球', 'bearing_ball', 'bearing'),
        ('行星轮', 'planet_gear', 'planet_gear'),
        ('行星轮3', 'planet_gear_3', 'planet_gear'),
        ('齿圈', 'ring_gear', 'ring_gear'),
        ('g3', 'g3', 'gear'),
        ('g0_2', 'g0_2', 'gear'),
        ('zhou1', 'zhou1', 'shaft'),
    ),
)
def test_chinese_and_reducer_component_names_are_semantically_normalized(
    regionName, componentInstance, componentType
):
    assert _normalizedComponentName(regionName) == componentInstance
    assert _componentTypeName(regionName) == componentType


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
        'solverSetting': {
            'timeDomain': {
                'isTransient': False,
                'deltaTime': 0.1,
                'endTime': 1.0,
            }
        },
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


def _useDirectFemRegionSchema(case):
    controlPath = case / 'controlData.json'
    control = json.loads(controlPath.read_text(encoding='utf-8'))
    root = control['rootSystem']
    femSystems = root['thermalComponents']['fem3DThermals']
    femSystem = next(iter(femSystems.values()))
    root['uuid'] = 'direct-root'
    root['thermalComponents'] = {
        'femRegions': femSystem['femRegions'],
        'femFemTccInterfaces': femSystem.get('tccInterfaces', {}),
    }
    controlPath.write_text(json.dumps(control), encoding='utf-8')
    resultPath = case / 'results' / 'system' / '2500.h5'
    with h5py.File(resultPath, 'a') as file:
        source = file['thermalComponents/thermal3dFem/system/region/3dResults/results']
        target = file.create_group('thermalComponents/femRegion/region/results')
        target.create_dataset('temperature', data=source['temperature'][:])
        target.create_dataset('heatFlowRate', data=source['heatFlowRate'][:])
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


def test_direct_fem_region_schema_is_supported(tmp_path):
    case = buildMotorFemComponentCase(
        _useDirectFemRegionSchema(_makeSolvedCase(tmp_path)),
        'part',
    )

    assert case.metadata['femSchema'] == 'directFemRegions'
    assert case.metadata['femSystemUuid'] == 'direct-root'
    np.testing.assert_allclose(case.temperature.temperatureK, [300.0, 301.0, 302.0, 303.0])


def _addNetworkSurfacePower(source, power=100.0):
    controlPath = source / 'controlData.json'
    control = json.loads(controlPath.read_text(encoding='utf-8'))
    control['rootSystem']['thermalComponents']['femNetworkInterfaces'] = {
        'network-interface': {
            'isActive': True,
            'fem': {
                'femUuid': 'region',
                'patchUuid': 'surface_a',
                'isHeatFluxInput': True,
                'heatFlux': {
                    'inputVariable': 'power',
                    'heatFlux1D': {
                        'type': 'fixedValue',
                        'fixedValue': power,
                    },
                },
            },
        }
    }
    controlPath.write_text(json.dumps(control), encoding='utf-8')
    return source


def test_fem_network_surface_power_is_distributed_to_boundary_nodes(tmp_path):
    source = _addNetworkSurfacePower(_makeSolvedCase(tmp_path))

    case = buildMotorFemComponentCase(source, 'part')
    patch = case.metadata['surfacePatches'][0]
    mask = case.boundary.patchIds == patch['patchId']

    assert patch['boundaryInputType'] == 'prescribedHeatFlux'
    assert patch['boundaryTypeId'] == PRESCRIBED_HEAT_FLUX_BOUNDARY_TYPE
    assert patch['networkInterfaceUuid'] == 'network-interface'
    assert patch['resultHeatFlowSource'] == 'prescribedInput'
    np.testing.assert_allclose(patch['prescribedHeatFlowRateW'], 100.0)
    np.testing.assert_allclose(patch['prescribedSurfaceHeatSourcePowerW'], 100.0)
    np.testing.assert_allclose(patch['sampledBoundaryInputHeatFlowRateW'], 100.0)
    np.testing.assert_allclose(
        np.dot(case.boundary.heatFluxWPerM2[mask], case.boundary.weightsM2[mask]),
        100.0,
    )
    np.testing.assert_allclose(case.metadata['prescribedBoundaryHeatFlowRateW'], 100.0)
    np.testing.assert_allclose(case.metadata['prescribedSurfaceHeatSourcePowerW'], 100.0)
    np.testing.assert_allclose(
        case.metadata['hybridBoundaryInputHeatFlowRateW'],
        case.boundary.totalHeatFlowRateW,
    )


def test_solver_connection_heat_flow_closes_component_energy_balance(tmp_path):
    source = _makeSolvedCase(tmp_path)
    resultPath = source / 'results' / 'system' / '2500.h5'
    with h5py.File(resultPath, 'a') as file:
        connections = file.create_group('connections')
        connections.create_dataset(
            'heatFlowRate',
            data=np.asarray(
                [
                    b'region@surface_a&undefined',
                    b'external-a@null&A',
                    b'3.0',
                    b'region@surface_b&undefined',
                    b'external-b@null&A',
                    b'9.0',
                ],
                dtype='S64',
            ),
        )

    case = buildMotorFemComponentCase(source, 'part')

    np.testing.assert_allclose(
        list(case.boundary.heatFlowRateWByPatch().values()), [-3.0, -9.0]
    )
    np.testing.assert_allclose(case.metadata['connectionBoundaryHeatFlowRateW'], -12.0)
    np.testing.assert_allclose(case.metadata['hybridBoundaryInputHeatFlowRateW'], -12.0)
    np.testing.assert_allclose(case.metadata['steadyEnergyResidualW'], 0.0)
    assert case.metadata['connectionBoundaryPatchCount'] == 2
    assert [
        patch['resultHeatFlowSource'] for patch in case.metadata['surfacePatches']
    ] == ['connectionTable', 'connectionTable']


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


def test_transient_analysis_metadata_keeps_snapshot_time_separate():
    metadata = _thermalAnalysisMetadata(
        {
            'solverSetting': {
                'timeDomain': {
                    'isTransient': True,
                    'deltaTime': 0.5,
                    'endTime': 60.0,
                }
            }
        },
        snapshotTimeS=60.0,
    )

    assert metadata == {
        'analysisType': 'transient',
        'isTransient': True,
        'temporalSampleType': 'transientSnapshot',
        'timeSeriesStored': False,
        'sourceTimeStepS': 0.5,
        'sourceEndTimeS': 60.0,
        'snapshotTimeS': 60.0,
    }


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
    assert cases[0].metadata['analysisType'] == 'steady'
    assert cases[0].metadata['temporalSampleType'] == 'steadySolution'
    assert 'snapshotTimeS' not in cases[0].operatingFeatures
    assert index['metadata']['datasetType'] == 'singleMotorFemComponents'
    assert index['metadata']['analysisType'] == 'steady'
    assert index['metadata']['selectedComponents'] == ['part']
    assert index['featureSchema']['boundaryTypeNames'] == {
        '0': 'resultHeatFlux',
        '1': 'convection',
        '2': 'fixedTemperature',
        '3': 'prescribedHeatFlux',
    }
    assert wholeIndex['viewType'] == 'wholeMotor'
    assert wholeManifest['components'][0]['caseId'] == 'baseline-part'
    assert wholeManifest['analysisType'] == 'steady'
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
    _addNetworkSurfacePower(_makeSolvedCase(sourceRoot))
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
        mask = arrays['boundaryTypeIds'] == PRESCRIBED_HEAT_FLUX_BOUNDARY_TYPE
        np.testing.assert_allclose(
            np.dot(arrays['boundaryHeatFluxWPerM2'][mask], arrays['boundaryWeightsM2'][mask]),
            100.0,
        )
    with np.load(graphCase / 'data.npz') as arrays:
        np.testing.assert_array_equal(arrays['tetrahedra'], [[0, 1, 2, 3]])
        assert arrays['surfaceTriangles'].shape == (4, 3)
        assert arrays['nodeHeatSourceWPerM3'].shape == (4,)
        mask = arrays['boundaryTypeIds'] == PRESCRIBED_HEAT_FLUX_BOUNDARY_TYPE
        np.testing.assert_allclose(
            np.dot(arrays['boundaryHeatFluxWPerM2'][mask], arrays['boundaryWeightsM2'][mask]),
            100.0,
        )
    assert pointIndex['metadata']['randomSubsampling'] is False
    assert graphIndex['connectivityStored'] is True
    assert graphIndex['cases'][0]['systemType'] == 'electric_machine'
    assert graphIndex['cases'][0]['assemblyType'] == 'motor'
    assert graphIndex['cases'][0]['assemblyId'] == 'motor-a'
    hierarchy = json.loads((outputRoot / 'hierarchy.json').read_text(encoding='utf-8'))
    motorHierarchy = hierarchy['systems']['electric_machine']['assemblyTypes']['motor']
    assert motorHierarchy['assemblyIds'] == ['motor-a']
    assert (
        outputRoot
        / motorHierarchy['views']['graph']['steady']['components']
    ).is_file()
    assert (outputRoot / 'graph' / 'whole_assembly' / 'dataset.json').is_file()
    assert not (outputRoot / 'point' / 'assets').exists()
    assert not (outputRoot / 'graph' / 'assets').exists()


def test_transient_sta_stores_one_fixed_mesh_with_all_result_times(tmp_path):
    sourceRoot = tmp_path / 'source'
    solvedCase = _addNetworkSurfacePower(_makeSolvedCase(sourceRoot))
    outputRoot = tmp_path / 'full'
    buildFullResolutionMotorDatasetFromDirectories(
        sourceRoot,
        outputRoot,
        motorId='motor-a',
    )

    controlPath = solvedCase / 'controlData.json'
    control = json.loads(controlPath.read_text(encoding='utf-8'))
    control['solverSetting']['timeDomain'] = {
        'isTransient': True,
        'deltaTime': 1.0,
        'endTime': 1.0,
    }
    controlPath.write_text(json.dumps(control), encoding='utf-8')
    resultDirectory = solvedCase / 'results' / 'system'
    finalResult = resultDirectory / '2500.h5'
    firstResult = resultDirectory / '0.000000.h5'
    secondResult = resultDirectory / '1.000000.h5'
    shutil.copy2(finalResult, firstResult)
    shutil.copy2(finalResult, secondResult)
    finalResult.unlink()
    resultBase = 'thermalComponents/thermal3dFem/system/region/3dResults/results'
    with h5py.File(firstResult, 'r+') as file:
        file[f'{resultBase}/temperature'][:] = [290.0, 291.0, 292.0, 293.0]

    archivePath = tmp_path / 'transient.sta'
    with ZipFile(archivePath, 'w', compression=ZIP_STORED, allowZip64=True) as archive:
        for path in solvedCase.rglob('*'):
            if path.is_file():
                archive.write(path, path.relative_to(solvedCase).as_posix())

    pointIndexPath = outputRoot / 'point' / 'dataset.json'
    graphIndexPath = outputRoot / 'graph' / 'dataset.json'
    pointIndex = json.loads(pointIndexPath.read_text(encoding='utf-8'))
    graphIndex = json.loads(graphIndexPath.read_text(encoding='utf-8'))
    summary = buildMotorTransientSeriesFromSta(
        archivePath,
        outputRoot / 'graph',
        outputRoot / 'point',
        graphIndex['cases'],
        pointIndex['cases'],
    )
    graphIndexPath.write_text(json.dumps(graphIndex, indent=2), encoding='utf-8')
    pointIndexPath.write_text(json.dumps(pointIndex, indent=2), encoding='utf-8')
    writeMotorThermalAnalysisIndexes(outputRoot)

    records = loadMotorTransientSeriesRecords(outputRoot / 'graph' / 'transient')
    assert summary['timeStepCount'] == 2
    assert len(records) == 1
    assert records[0].systemType == 'electric_machine'
    assert records[0].assemblyType == 'motor'
    assert records[0].assemblyId == 'motor-a'
    np.testing.assert_allclose(records[0].times(), [0.0, 1.0])
    np.testing.assert_allclose(
        records[0].loadFrame(0).nodeTemperatureK,
        [290.0, 291.0, 292.0, 293.0],
    )
    np.testing.assert_allclose(
        records[0].loadFrame(1).nodeTemperatureK,
        [300.0, 301.0, 302.0, 303.0],
    )
    with np.load(records[0].geometryPath) as geometry:
        prescribed = (
            geometry['boundaryTypeIds'] == PRESCRIBED_HEAT_FLUX_BOUNDARY_TYPE
        )
        weights = geometry['boundaryWeightsM2'][prescribed]
    for frameIndex in range(2):
        np.testing.assert_allclose(
            np.dot(
                records[0].loadFrame(frameIndex).boundaryHeatFluxWPerM2[prescribed],
                weights,
            ),
            100.0,
        )
    transitions = tuple(records[0].iterTransitions())
    assert len(transitions) == 1
    assert transitions[0].deltaTimeS == 1.0
    np.testing.assert_allclose(
        transitions[0].targetTemperatureK,
        [300.0, 301.0, 302.0, 303.0],
    )
    assert (outputRoot / 'point' / pointIndex['cases'][0]['path'] / 'series.h5').is_file()
    assert not (outputRoot / 'graph' / 'transient' / 'assets').exists()

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import re
import shutil
from zipfile import ZipFile

from mor.nn.data.motorFemComponentCase import (
    buildMotorFemComponentCase,
)
from mor.nn.data.motorFemMetadata import (
    normalizedComponentName,
    resolveFemSystem,
)
from mor.nn.data.motorFemIO import (
    loadH5py,
)
from mor.nn.data.motorFullResolutionDataset import (
    FULL_DATASET_FORMAT_VERSION,
    GRAPH_FORMAT_VERSION,
    POINT_FORMAT_VERSION,
    _geometryClassificationTask,
    _meshData,
    _saveSelfContainedCase,
    _validateAlignment,
    _writeJson,
)
from mor.nn.data.motorStaDataset import (
    _prepareExtractedDirectory,
    _staTemporalMetadata,
    inspectMotorStaArchive,
)
from mor.nn.data.motorTransientDataset import (
    buildMotorTransientSeriesFromSta,
    writeMotorThermalAnalysisIndexes,
)
from mor.nn.data.motorWholeDataset import saveWholeMotorDataset


def _identifier(value: str) -> str:
    normalized = re.sub(r'[^A-Za-z0-9_-]+', '-', value.strip()).strip('-_').lower()
    if not normalized:
        raise ValueError(f'cannot normalize identifier: {value!r}')
    return normalized


def _tutorialNumber(value: str) -> int | None:
    match = re.fullmatch(r'tutorial-([a-z]+)', value)
    if match is None:
        return None
    result = 0
    for character in match.group(1):
        result = result * 26 + ord(character) - ord('a') + 1
    return result


def _tutorialName(number: int) -> str:
    if number <= 0:
        raise ValueError('tutorial number must be positive')
    letters = []
    while number:
        number, remainder = divmod(number - 1, 26)
        letters.append(chr(ord('a') + remainder))
    return f'tutorial-{"".join(reversed(letters))}'


def _motorId(path: Path) -> str:
    stem = path.stem.lower()
    if stem.startswith('df'):
        return 'motor-df-prototype'
    if stem.startswith('hk'):
        return 'motor-hk-prototype'
    nameWithoutDigits = re.sub(r'\d+', '', stem) or 'new'
    return f'motor-{_identifier(nameWithoutDigits)}-prototype'


def _classification(path: Path) -> dict[str, str]:
    value = str(path).lower()
    stem = path.stem.lower()
    if 'catl' in value:
        return {
            'systemType': 'integrated_drivetrain',
            'assemblyType': 'electric_drive_unit',
            'assemblyId': 'catl-electric-drive-unit-prototype',
        }
    if '南高齿' in value:
        return {
            'systemType': 'transmission_system',
            'assemblyType': 'reducer',
            'assemblyId': 'nangaochi-reducer-prototype',
        }
    if '增加换热器工况' in value or '增加工况2.4' in value:
        return {
            'systemType': 'transmission_system',
            'assemblyType': 'reducer',
            'assemblyId': 'nangaochi-reducer-prototype',
        }
    if stem.startswith('project370'):
        return {
            'systemType': 'transmission_system',
            'assemblyType': 'bearing_assembly',
            'assemblyId': 'project370-bearing-prototype',
        }
    if stem in {'gearboxffff', 'gearsamplecomplete'}:
        return {
            'systemType': 'transmission_system',
            'assemblyType': 'reducer',
            'assemblyId': 'gearbox-reducer-prototype',
        }
    if stem == 'case_2_5':
        return {
            'systemType': 'transmission_system',
            'assemblyType': 'reducer',
            'assemblyId': 'planetary-reducer-prototype',
        }
    if '空间分布热源的立方体' in value:
        return {
            'systemType': 'generic_thermal',
            'assemblyType': 'benchmark',
            'assemblyId': 'distributed-heat-source-cube',
        }
    if '电池冷却系统' in value:
        return {
            'systemType': 'energy_storage',
            'assemblyType': 'battery_cooling_system',
            'assemblyId': 'battery-cooling-prototype',
        }
    if '一维电机网络' in value:
        return {
            'systemType': 'electric_machine',
            'assemblyType': 'motor_network',
            'assemblyId': 'motor-1d-network-prototype',
        }
    if '扁线永磁同步电机' in value:
        return {
            'systemType': 'electric_machine',
            'assemblyType': 'motor',
            'assemblyId': 'flat-wire-pmsm-prototype',
        }
    if stem == 'xan':
        return {
            'systemType': 'generic_thermal',
            'assemblyType': 'thermal_network',
            'assemblyId': 'xan-network-prototype',
        }
    if '1d-3d 电机网络' in value:
        return {
            'systemType': 'electric_machine',
            'assemblyType': 'motor',
            'assemblyId': 'motor-network-prototype',
        }
    if '结果文件' in value:
        return {
            'systemType': 'electric_machine',
            'assemblyType': 'motor',
            'assemblyId': 'motor-project-prototype',
        }
    motorId = _motorId(path)
    return {
        'systemType': 'electric_machine',
        'assemblyType': 'motor',
        'assemblyId': motorId,
    }


def _loadJson(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _existingSourceArchives(datasetRoot: Path) -> set[str]:
    views = _loadJson(datasetRoot / 'views.json')
    return {str(Path(value).resolve()).lower() for value in views.get('sourceArchives', ())}


def _existingExtractedDuplicate(source: Path, info, datasetRoot: Path) -> str | None:
    if source.stem.lower() != '7500htc':
        return None
    index = _loadJson(datasetRoot / 'graph' / 'dataset.json')
    sourceMeshes = {Path(member).stem.lower() for member in info.meshMembers}
    for entry in index.get('cases', ()):
        if str(entry.get('meshUuid', '')).lower() not in sourceMeshes:
            continue
        manifest = _loadJson(datasetRoot / 'graph' / entry['path'] / 'manifest.json')
        if str(manifest.get('metadata', {}).get('sourceCase', '')).strip() == '7500':
            return 'same HTC 7500 result is already present from the extracted case directory'
    return None


def _unsupportedReason(error: Exception) -> str:
    message = f'{type(error).__name__}: {error}'
    if 'neither fem3DThermals nor direct femRegions' in message:
        return 'no compatible 3D FEM regions'
    if 'result' in message.lower() and (
        'multiple non-numeric' in message or 'does not contain' in message
    ):
        return 'no compatible solved 3D FEM result'
    return message


def _records(sourceDirectory: Path, datasetRoot: Path) -> tuple[list[dict], list[dict]]:
    views = _loadJson(datasetRoot / 'views.json')
    usedNumbers = [
        number
        for value in views.get('parentCaseIds', ())
        if (number := _tutorialNumber(str(value))) is not None
    ]
    nextNumber = max(usedNumbers, default=0) + 1
    existingSources = _existingSourceArchives(datasetRoot)
    records = []
    inventory = []
    for source in sorted(sourceDirectory.rglob('*.sta')):
        labels = _classification(source)
        if str(source.resolve()).lower() in existingSources:
            inventory.append({
                'source': str(source.resolve()),
                'status': 'duplicate',
                **labels,
                'reason': 'source archive already listed in the dataset',
            })
            continue
        try:
            info = inspectMotorStaArchive(source)
        except Exception as error:
            inventory.append({
                'source': str(source.resolve()),
                'status': 'unsupported',
                **labels,
                'reason': _unsupportedReason(error),
                'detail': f'{type(error).__name__}: {error}',
            })
            continue
        duplicateReason = _existingExtractedDuplicate(source, info, datasetRoot)
        if duplicateReason is not None:
            inventory.append({
                'source': str(source.resolve()),
                'status': 'duplicate',
                **labels,
                'reason': duplicateReason,
            })
            continue
        if labels['assemblyId'] == 'distributed-heat-source-cube':
            inventory.append({
                'source': str(source.resolve()),
                'status': 'unsupported',
                **labels,
                'reason': (
                    'non-tetrahedral volume connectivity is not supported by the '
                    'current full-resolution graph schema'
                ),
            })
            continue
        record = {
            'source': source,
            'motorId': labels['assemblyId'],
            'parentCaseId': _tutorialName(nextNumber),
            'info': info,
            **labels,
        }
        records.append(record)
        inventory.append({
            'source': str(source.resolve()),
            'status': 'accepted',
            'parentCaseId': record['parentCaseId'],
            **labels,
            'analysisType': info.analysisType,
            'componentCount': len(info.componentNames),
            'resultFrameCount': len(info.availableResultMembers),
        })
        nextNumber += 1
    return records, inventory


def _caseTargets(root: Path, motorId: str, parentCaseId: str, components: tuple[str, ...]) -> list[Path]:
    targets = []
    for view in ('point', 'graph'):
        caseRoot = root / view / 'motors' / motorId / 'cases'
        targets.extend(caseRoot / f'{parentCaseId}-{component}' for component in components)
        targets.append(root / view / 'whole_motor' / 'cases' / parentCaseId)
    return targets


def _solvedComponents(extractedRoot: Path, info) -> tuple[tuple[str, ...], tuple[str, ...]]:
    control = _loadJson(extractedRoot / 'controlData.json')
    femSystemUuid, femSystem, femSchema = resolveFemSystem(control)
    solved = []
    skipped = []
    h5py = loadH5py()
    with h5py.File(extractedRoot / info.resultMember, 'r') as result:
        for regionUuid, region in femSystem.get('femRegions', {}).items():
            name = normalizedComponentName(region.get('name', ''))
            if femSchema == 'nestedFem3DThermal':
                base = (
                    f'thermalComponents/thermal3dFem/{femSystemUuid}/'
                    f'{regionUuid}/3dResults/results'
                )
            else:
                base = f'thermalComponents/femRegion/{regionUuid}/results'
            (solved if f'{base}/temperature' in result else skipped).append(name)
    if not solved:
        raise ValueError('selected result does not contain temperature for any 3D FEM region')
    return tuple(solved), tuple(skipped)


def _buildCase(record: dict, stagingRoot: Path) -> dict:
    source = record['source']
    motorId = record['motorId']
    parentCaseId = record['parentCaseId']
    info = record['info']
    hierarchy = {
        key: record[key]
        for key in ('systemType', 'assemblyType', 'assemblyId')
    }
    temporalMetadata = _staTemporalMetadata(info)
    targets = _caseTargets(stagingRoot, motorId, parentCaseId, info.componentNames)
    occupied = [path for path in targets if path.exists()]
    if occupied:
        raise FileExistsError(f'staging target already exists: {occupied[0]}')

    pointRoot = stagingRoot / 'point'
    graphRoot = stagingRoot / 'graph'
    pointEntries = []
    graphEntries = []
    componentCases = []
    skippedComponents = ()
    with ZipFile(source) as archive:
        extractedRoot, temporary, _ = _prepareExtractedDirectory(
            archive,
            info,
            None,
            stagingRoot / '.extracted',
        )
        try:
            solvedComponents, skippedComponents = _solvedComponents(extractedRoot, info)
            for component in skippedComponents:
                print(json.dumps({
                    'event': 'component-skipped',
                    'caseId': parentCaseId,
                    'component': component,
                    'reason': 'selected result has no temperature field for this FEM region',
                }), flush=True)
            for index, component in enumerate(solvedComponents, start=1):
                print(json.dumps({
                    'event': 'component-start',
                    'caseId': parentCaseId,
                    'component': component,
                    'position': index,
                    'count': len(solvedComponents),
                }), flush=True)
                case = buildMotorFemComponentCase(
                    extractedRoot,
                    component,
                    resultFile=info.resultMember,
                    motorId=motorId,
                    parentCaseId=parentCaseId,
                )
                case = replace(
                    case,
                    operatingFeatures={
                        **case.operatingFeatures,
                        **(
                            {'snapshotTimeS': temporalMetadata['snapshotTimeS']}
                            if temporalMetadata.get('snapshotTimeS') is not None
                            else {}
                        ),
                    },
                    metadata={
                        **case.metadata,
                        **temporalMetadata,
                        'sourceCase': source.name,
                        'sourceSta': str(source.resolve()),
                        'resultFile': info.resultMember,
                        'resultTime': info.resultTime,
                        **hierarchy,
                    },
                )
                meshData = _meshData(extractedRoot, component)
                _validateAlignment(case, meshData)
                pointEntries.append(_saveSelfContainedCase(
                    pointRoot, case, meshData, formatVersion=POINT_FORMAT_VERSION
                ))
                graphEntries.append(_saveSelfContainedCase(
                    graphRoot, case, meshData, formatVersion=GRAPH_FORMAT_VERSION
                ))
                componentCases.append(case)
                print(json.dumps({
                    'event': 'component-complete',
                    'caseId': parentCaseId,
                    'component': component,
                    'nodes': int(meshData['nodesMeters'].shape[0]),
                    'tetrahedra': int(meshData['tetrahedra'].shape[0]),
                }), flush=True)
        finally:
            if temporary is not None:
                temporary.cleanup()

    sequenceSummary = None
    if info.isTransient is True:
        sequenceSummary = buildMotorTransientSeriesFromSta(
            source,
            graphRoot,
            pointRoot,
            graphEntries,
            pointEntries,
        )
        temporalMetadata = sequenceSummary['temporalMetadata']

    metadata = {
        'motorId': motorId,
        'parentCaseId': parentCaseId,
        'sourceSta': str(source.resolve()),
        'selectedResultMember': info.resultMember,
        'selectedResultTime': info.resultTime,
        **hierarchy,
        **temporalMetadata,
        'fullResolution': True,
        'randomSubsampling': False,
        'selfContainedCases': True,
        'geometryAssetsSharedByMeshUuid': False,
    }
    for viewRoot, entries in ((pointRoot, pointEntries), (graphRoot, graphEntries)):
        paths = {entry['caseId']: entry['path'] for entry in entries}
        saveWholeMotorDataset(
            componentCases,
            viewRoot / 'whole_motor',
            componentDatasetDirectory=viewRoot,
            componentCasePaths=paths,
            motorId=motorId,
            parentCaseId=parentCaseId,
            metadata=metadata,
            append=(viewRoot / 'whole_motor' / 'dataset.json').exists(),
        )
    marker = stagingRoot / '.completed' / f'{parentCaseId}.json'
    _writeJson(marker, {
        'source': str(source.resolve()),
        'motorId': motorId,
        'parentCaseId': parentCaseId,
        'pointEntries': pointEntries,
        'graphEntries': graphEntries,
        'analysisType': info.analysisType,
        **hierarchy,
        'skippedComponentsWithoutTemperature': list(skippedComponents),
        'sequenceSummary': sequenceSummary,
    })
    print(json.dumps({'event': 'case-complete', 'caseId': parentCaseId, 'source': source.name}), flush=True)
    return _loadJson(marker)


def _writeStagingIndexes(stagingRoot: Path, records: list[dict]) -> None:
    markers = [_loadJson(stagingRoot / '.completed' / f'{record["parentCaseId"]}.json') for record in records]
    motorIds = list(dict.fromkeys(marker['motorId'] for marker in markers))
    parentCaseIds = [marker['parentCaseId'] for marker in markers]
    sourceArchives = [marker['source'] for marker in markers]
    metadata = {
        'motorIds': motorIds,
        'systemTypes': sorted({marker['systemType'] for marker in markers}),
        'assemblyTypes': sorted({marker['assemblyType'] for marker in markers}),
        'assemblyIds': sorted({marker['assemblyId'] for marker in markers}),
        'parentCaseIds': parentCaseIds,
        'sourceArchives': sourceArchives,
        'fullResolution': True,
        'randomSubsampling': False,
        'selfContainedCases': True,
        'geometryAssetsSharedByMeshUuid': False,
        'analysisTypes': sorted({marker['analysisType'] for marker in markers}),
    }
    pointEntries = [entry for marker in markers for entry in marker['pointEntries']]
    graphEntries = [entry for marker in markers for entry in marker['graphEntries']]
    _writeJson(stagingRoot / 'point' / 'dataset.json', {
        'formatVersion': POINT_FORMAT_VERSION,
        'viewType': 'fullPoint',
        'connectivityStored': False,
        'selfContainedCases': True,
        'metadata': metadata,
        'tasks': {
            'geometryComponentClassification': _geometryClassificationTask(pointEntries),
        },
        'cases': pointEntries,
    })
    _writeJson(stagingRoot / 'graph' / 'dataset.json', {
        'formatVersion': GRAPH_FORMAT_VERSION,
        'viewType': 'fullGraph',
        'connectivityStored': True,
        'selfContainedCases': True,
        'metadata': metadata,
        'tasks': {
            'geometryComponentClassification': _geometryClassificationTask(graphEntries),
        },
        'cases': graphEntries,
    })
    _writeJson(stagingRoot / 'views.json', {
        'formatVersion': FULL_DATASET_FORMAT_VERSION,
        **metadata,
        'views': {
            'point': 'point/dataset.json',
            'pointWholeMotor': 'point/whole_motor/dataset.json',
            'graph': 'graph/dataset.json',
            'graphWholeMotor': 'graph/whole_motor/dataset.json',
        },
    })


def _union(first, second):
    return list(dict.fromkeys([*first, *second]))


def _aggregateMetadata(existing: dict, addition: dict) -> dict:
    existingMotorIds = existing.get('motorIds') or ([existing['motorId']] if existing.get('motorId') else [])
    additionMotorIds = addition.get('motorIds') or ([addition['motorId']] if addition.get('motorId') else [])
    return {
        'motorIds': _union(existingMotorIds, additionMotorIds),
        'systemTypes': _union(
            existing.get('systemTypes', ('electric_machine',)),
            addition.get('systemTypes', ('electric_machine',)),
        ),
        'assemblyTypes': _union(
            existing.get('assemblyTypes', ('motor',)),
            addition.get('assemblyTypes', ('motor',)),
        ),
        'assemblyIds': _union(
            existing.get('assemblyIds', existingMotorIds),
            addition.get('assemblyIds', additionMotorIds),
        ),
        'parentCaseIds': _union(existing.get('parentCaseIds', ()), addition.get('parentCaseIds', ())),
        'sourceArchives': _union(existing.get('sourceArchives', ()), addition.get('sourceArchives', ())),
        'fullResolution': True,
        'randomSubsampling': False,
        'selfContainedCases': True,
        'geometryAssetsSharedByMeshUuid': False,
        'multiMotor': True,
        'analysisTypes': _union(
            existing.get('analysisTypes', ()), addition.get('analysisTypes', ())
        ),
    }


def _merge(stagingRoot: Path, datasetRoot: Path) -> dict:
    existingViews = _loadJson(datasetRoot / 'views.json')
    additionViews = _loadJson(stagingRoot / 'views.json')
    aggregate = _aggregateMetadata(existingViews, additionViews)
    jsonUpdates = {}
    moves = []

    for view in ('point', 'graph'):
        existingIndexPath = datasetRoot / view / 'dataset.json'
        additionIndexPath = stagingRoot / view / 'dataset.json'
        existingIndex = _loadJson(existingIndexPath)
        additionIndex = _loadJson(additionIndexPath)
        existingEntries = list(existingIndex.get('cases', ()))
        for entry in existingEntries:
            entry.setdefault('motorId', Path(entry['path']).parts[1])
        additionEntries = list(additionIndex.get('cases', ()))
        existingIds = {(entry['motorId'], entry['caseId']) for entry in existingEntries}
        duplicateIds = [
            (entry['motorId'], entry['caseId'])
            for entry in additionEntries
            if (entry['motorId'], entry['caseId']) in existingIds
        ]
        if duplicateIds:
            raise ValueError(f'duplicate component cases: {duplicateIds[:3]}')
        for entry in additionEntries:
            source = stagingRoot / view / entry['path']
            target = datasetRoot / view / entry['path']
            moves.append((source, target))
        existingIndex['metadata'] = aggregate
        existingIndex['cases'] = [*existingEntries, *additionEntries]
        existingIndex['tasks'] = {
            'geometryComponentClassification': _geometryClassificationTask(
                existingIndex['cases']
            ),
        }
        jsonUpdates[existingIndexPath] = existingIndex

        existingWholePath = datasetRoot / view / 'whole_motor' / 'dataset.json'
        additionWholePath = stagingRoot / view / 'whole_motor' / 'dataset.json'
        existingWhole = _loadJson(existingWholePath)
        additionWhole = _loadJson(additionWholePath)
        existingWholeIds = {
            (entry['motorId'], entry['parentCaseId']) for entry in existingWhole.get('cases', ())
        }
        duplicateWhole = [
            (entry['motorId'], entry['parentCaseId'])
            for entry in additionWhole.get('cases', ())
            if (entry['motorId'], entry['parentCaseId']) in existingWholeIds
        ]
        if duplicateWhole:
            raise ValueError(f'duplicate whole-motor cases: {duplicateWhole[:3]}')
        for entry in additionWhole.get('cases', ()):
            source = additionWholePath.parent / entry['path']
            target = existingWholePath.parent / entry['path']
            moves.append((source, target))
        existingWhole['metadata'] = aggregate
        existingWhole['cases'] = [*existingWhole.get('cases', ()), *additionWhole.get('cases', ())]
        jsonUpdates[existingWholePath] = existingWhole

    existingViews.pop('motorId', None)
    existingViews.update(aggregate)
    existingViews['formatVersion'] = 'motor-full-dual-dataset-v2'
    jsonUpdates[datasetRoot / 'views.json'] = existingViews

    missing = [source for source, _ in moves if not source.is_dir()]
    occupied = [target for _, target in moves if target.exists()]
    if missing:
        raise FileNotFoundError(f'missing staging directory: {missing[0]}')
    if occupied:
        raise FileExistsError(f'dataset target already exists: {occupied[0]}')

    temporaryFiles = {}
    moved = []
    try:
        for path, value in jsonUpdates.items():
            temporary = path.with_name(f'.{path.name}.append-sta.tmp')
            temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
            temporaryFiles[path] = temporary
        for source, target in moves:
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)
            moved.append((source, target))
        for path, temporary in temporaryFiles.items():
            os.replace(temporary, path)
    except Exception:
        for source, target in reversed(moved):
            if target.exists() and not source.exists():
                source.parent.mkdir(parents=True, exist_ok=True)
                target.rename(source)
        for temporary in temporaryFiles.values():
            if temporary.exists():
                temporary.unlink()
        raise
    return {
        'movedDirectories': len(moves),
        'componentCasesAdded': sum(
            len(_loadJson(stagingRoot / view / 'dataset.json').get('cases', ()))
            for view in ('point',)
        ),
        'wholeMotorCasesAdded': len(_loadJson(stagingRoot / 'point' / 'whole_motor' / 'dataset.json').get('cases', ())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Recursively append compatible STA archives as classified full-resolution '
            'point/graph thermal assembly cases.'
        )
    )
    parser.add_argument('source_directory', type=Path)
    parser.add_argument('dataset_root', type=Path)
    parser.add_argument('--staging', type=Path)
    parser.add_argument('--plan', action='store_true', help='Inspect and print assignments without writing data')
    arguments = parser.parse_args()

    datasetRoot = arguments.dataset_root.resolve()
    stagingRoot = (
        arguments.staging.resolve()
        if arguments.staging is not None
        else datasetRoot.with_name(f'{datasetRoot.name}_sta_staging')
    )
    records, inventory = _records(arguments.source_directory.resolve(), datasetRoot)
    assignments = [{
        'source': str(record['source'].resolve()),
        'systemType': record['systemType'],
        'assemblyType': record['assemblyType'],
        'assemblyId': record['assemblyId'],
        'parentCaseId': record['parentCaseId'],
        'resultMember': record['info'].resultMember,
        'analysisType': record['info'].analysisType,
        'resultCount': len(record['info'].availableResultMembers),
        'components': record['info'].componentNames,
    } for record in records]
    print(json.dumps({
        'event': 'assignments',
        'cases': assignments,
        'inventory': inventory,
    }, indent=2), flush=True)
    if arguments.plan:
        return

    if not records:
        _writeJson(datasetRoot / 'sta_import_inventory.json', {
            'sourceRoot': str(arguments.source_directory.resolve()),
            'records': inventory,
        })
        print(json.dumps({'event': 'nothing-to-add'}))
        return

    stagingRoot.mkdir(parents=True, exist_ok=True)
    completedRecords = []
    for record in records:
        marker = stagingRoot / '.completed' / f'{record["parentCaseId"]}.json'
        if marker.is_file():
            completed = _loadJson(marker)
            if completed.get('source', '').lower() != str(record['source'].resolve()).lower():
                raise ValueError(f'staging marker source mismatch: {marker}')
            print(json.dumps({'event': 'case-resumed', 'caseId': record['parentCaseId']}), flush=True)
            completedRecords.append(record)
            continue
        try:
            _buildCase(record, stagingRoot)
        except Exception as error:
            for item in inventory:
                if item['source'].lower() == str(record['source'].resolve()).lower():
                    item['status'] = 'conversion_failed'
                    item['reason'] = f'{type(error).__name__}: {error}'
                    break
            print(json.dumps({
                'event': 'case-failed',
                'caseId': record['parentCaseId'],
                'source': str(record['source']),
                'error': f'{type(error).__name__}: {error}',
            }), flush=True)
            continue
        completedRecords.append(record)

    if completedRecords:
        _writeStagingIndexes(stagingRoot, completedRecords)
        result = _merge(stagingRoot, datasetRoot)
        writeMotorThermalAnalysisIndexes(datasetRoot)
    else:
        result = {
            'movedDirectories': 0,
            'componentCasesAdded': 0,
            'wholeMotorCasesAdded': 0,
        }
    imported = {str(record['source'].resolve()).lower() for record in completedRecords}
    completedDetails = {
        str(record['source'].resolve()).lower(): _loadJson(
            stagingRoot / '.completed' / f'{record["parentCaseId"]}.json'
        )
        for record in completedRecords
    }
    for item in inventory:
        if item['source'].lower() in imported and item['status'] == 'accepted':
            item['status'] = 'imported'
            skipped = completedDetails[item['source'].lower()].get(
                'skippedComponentsWithoutTemperature', ()
            )
            if skipped:
                item['skippedComponentsWithoutTemperature'] = list(skipped)
    _writeJson(datasetRoot / 'sta_import_inventory.json', {
        'sourceRoot': str(arguments.source_directory.resolve()),
        'records': inventory,
    })
    shutil.rmtree(stagingRoot)
    print(json.dumps({
        'event': 'complete',
        **result,
        'importedArchiveCount': len(completedRecords),
        'unsupportedOrFailedCount': sum(
            item['status'] in {'unsupported', 'conversion_failed'} for item in inventory
        ),
        'duplicateCount': sum(item['status'] == 'duplicate' for item in inventory),
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()

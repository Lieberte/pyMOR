from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence
from zipfile import ZipFile

import numpy as np

from .motorFemComponentCase import (
    PRESCRIBED_HEAT_FLUX_BOUNDARY_TYPE,
    RESULT_HEAT_FLUX_BOUNDARY_TYPE,
    _boundaryNodeGeometry,
    _h5py,
    _matchIntegratedPower,
    _overviewHeatFlowRates,
)
from .motorFullResolutionDataset import _writeJson
from .motorStaDataset import motorStaArchiveInfo, inspectMotorStaArchive


TRANSIENT_SERIES_FORMAT_VERSION = 'motor-transient-series-v1'


@dataclass(frozen=True)
class motorTransientFrame:
    timeS: float
    nodeTemperatureK: np.ndarray
    boundaryHeatFluxWPerM2: np.ndarray


@dataclass(frozen=True)
class motorTransientTransition:
    previousTimeS: float
    currentTimeS: float
    deltaTimeS: float
    previousTemperatureK: np.ndarray
    targetTemperatureK: np.ndarray
    boundaryHeatFluxWPerM2: np.ndarray


@dataclass(frozen=True)
class motorTransientSeriesRecord:
    caseId: str
    systemType: str
    assemblyType: str
    assemblyId: str
    componentType: str
    componentInstance: str
    meshUuid: str
    caseDirectory: Path
    seriesFile: str
    timeStepCount: int

    @property
    def seriesPath(self) -> Path:
        return self.caseDirectory / self.seriesFile

    @property
    def geometryPath(self) -> Path:
        return self.caseDirectory / 'data.npz'

    def times(self) -> np.ndarray:
        h5py = _h5py()
        with h5py.File(self.seriesPath, 'r') as series:
            return np.asarray(series['timeS'], dtype=np.float64)

    def loadFrame(self, index: int) -> motorTransientFrame:
        h5py = _h5py()
        with h5py.File(self.seriesPath, 'r') as series:
            count = int(series['timeS'].shape[0])
            resolvedIndex = int(index)
            if resolvedIndex < 0:
                resolvedIndex += count
            if resolvedIndex < 0 or resolvedIndex >= count:
                raise IndexError(f'transient frame index is out of range: {index}')
            return motorTransientFrame(
                timeS=float(series['timeS'][resolvedIndex]),
                nodeTemperatureK=np.asarray(
                    series['nodeTemperatureK'][resolvedIndex], dtype=np.float64
                ),
                boundaryHeatFluxWPerM2=np.asarray(
                    series['boundaryHeatFluxWPerM2'][resolvedIndex], dtype=np.float64
                ),
            )

    def iterTransitions(self):
        h5py = _h5py()
        with h5py.File(self.seriesPath, 'r') as series:
            times = np.asarray(series['timeS'], dtype=np.float64)
            for currentIndex in range(1, times.size):
                previousTime = float(times[currentIndex - 1])
                currentTime = float(times[currentIndex])
                yield motorTransientTransition(
                    previousTimeS=previousTime,
                    currentTimeS=currentTime,
                    deltaTimeS=currentTime - previousTime,
                    previousTemperatureK=np.asarray(
                        series['nodeTemperatureK'][currentIndex - 1], dtype=np.float64
                    ),
                    targetTemperatureK=np.asarray(
                        series['nodeTemperatureK'][currentIndex], dtype=np.float64
                    ),
                    boundaryHeatFluxWPerM2=np.asarray(
                        series['boundaryHeatFluxWPerM2'][currentIndex], dtype=np.float64
                    ),
                )


@dataclass
class _componentContext:
    entry: dict[str, Any]
    manifestPath: Path
    manifest: dict[str, Any]
    nodesMeters: np.ndarray
    sourceTriangles: np.ndarray
    boundarySourceNodes: np.ndarray
    boundaryWeightsM2: np.ndarray
    boundaryPatchIds: np.ndarray
    boundaryTypeIds: np.ndarray
    nodalBoundaryAreasM2: np.ndarray
    boundaryNodes: np.ndarray
    patchMetadata: tuple[dict[str, Any], ...]
    resultBase: str
    overviewBase: str
    finalBoundaryHeatFluxWPerM2: np.ndarray
    temporarySeriesPath: Path
    finalSeriesPath: Path
    seriesHandle: Any = None
    temperatureDataset: Any = None
    boundaryFluxDataset: Any = None


def _resultPaths(manifest: dict[str, Any]) -> tuple[str, str]:
    metadata = manifest['metadata']
    femSystemUuid = str(metadata['femSystemUuid'])
    regionUuid = str(metadata['regionUuid'])
    if metadata.get('femSchema') == 'nestedFem3DThermal':
        return (
            f'thermalComponents/thermal3dFem/{femSystemUuid}/{regionUuid}/3dResults/results',
            f'thermalComponents/thermal3dFem/{femSystemUuid}/{regionUuid}/overviewResults',
        )
    return (
        f'thermalComponents/femRegion/{regionUuid}/results',
        f'thermalComponents/femRegion/{regionUuid}/overviewResults',
    )


def _loadContext(viewRoot: Path, entry: dict[str, Any], frameCount: int) -> _componentContext:
    caseDirectory = viewRoot / entry['path']
    manifestPath = caseDirectory / 'manifest.json'
    manifest = json.loads(manifestPath.read_text(encoding='utf-8'))
    if manifest.get('meshUuid') != entry.get('meshUuid'):
        raise ValueError(f'meshUuid mismatch in transient case: {caseDirectory}')
    with np.load(caseDirectory / 'data.npz', allow_pickle=False) as data:
        required = (
            'nodesMeters',
            'surfaceTriangles',
            'elementCentersMeters',
            'elementVolumesM3',
            'boundarySampleNodeIndices',
            'boundaryWeightsM2',
            'boundaryPatchIds',
            'boundaryTypeIds',
            'boundaryHeatFluxWPerM2',
        )
        missing = tuple(name for name in required if name not in data)
        if missing:
            raise KeyError(f'transient graph case lacks arrays {missing}: {caseDirectory}')
        nodes = np.asarray(data['nodesMeters'], dtype=np.float64)
        triangles = np.asarray(data['surfaceTriangles'], dtype=np.int64)
        centers = np.asarray(data['elementCentersMeters'], dtype=np.float64)
        volumes = np.asarray(data['elementVolumesM3'], dtype=np.float64)
        sourceNodes = np.asarray(data['boundarySampleNodeIndices'], dtype=np.int64)
        boundaryWeights = np.asarray(data['boundaryWeightsM2'], dtype=np.float64)
        patchIds = np.asarray(data['boundaryPatchIds'], dtype=np.int64)
        boundaryTypes = np.asarray(data['boundaryTypeIds'], dtype=np.int64)
        finalFlux = np.asarray(data['boundaryHeatFluxWPerM2'], dtype=np.float64)
    if sourceNodes.size != boundaryWeights.size or sourceNodes.size != patchIds.size:
        raise ValueError(f'transient boundary arrays are not aligned: {caseDirectory}')
    componentCenter = np.average(centers, axis=0, weights=volumes)
    boundaryNodes, _, boundaryNodeWeights = _boundaryNodeGeometry(
        nodes, triangles, componentCenter
    )
    nodalAreas = np.zeros(nodes.shape[0], dtype=np.float64)
    nodalAreas[boundaryNodes] = boundaryNodeWeights
    resultBase, overviewBase = _resultPaths(manifest)
    finalPath = caseDirectory / 'series.h5'
    return _componentContext(
        entry=entry,
        manifestPath=manifestPath,
        manifest=manifest,
        nodesMeters=nodes,
        sourceTriangles=triangles,
        boundarySourceNodes=sourceNodes,
        boundaryWeightsM2=boundaryWeights,
        boundaryPatchIds=patchIds,
        boundaryTypeIds=boundaryTypes,
        nodalBoundaryAreasM2=nodalAreas,
        boundaryNodes=boundaryNodes,
        patchMetadata=tuple(manifest['metadata'].get('surfacePatches', ())),
        resultBase=resultBase,
        overviewBase=overviewBase,
        finalBoundaryHeatFluxWPerM2=finalFlux,
        temporarySeriesPath=finalPath.with_suffix('.h5.tmp'),
        finalSeriesPath=finalPath,
    )


def _chunkWidth(valueCount: int) -> int:
    return max(1, min(int(valueCount), 65536))


def _openSeries(context: _componentContext, times: np.ndarray, members: Sequence[str]) -> None:
    h5py = _h5py()
    context.temporarySeriesPath.unlink(missing_ok=True)
    series = h5py.File(context.temporarySeriesPath, 'w')
    series.attrs['formatVersion'] = TRANSIENT_SERIES_FORMAT_VERSION
    series.attrs['analysisType'] = 'transient'
    series.attrs['caseId'] = context.manifest['caseId']
    series.attrs['componentType'] = context.manifest['componentType']
    series.attrs['componentInstance'] = context.manifest['componentInstance']
    series.attrs['meshUuid'] = context.manifest['meshUuid']
    series.attrs['systemType'] = context.manifest['systemType']
    series.attrs['assemblyType'] = context.manifest['assemblyType']
    series.attrs['assemblyId'] = context.manifest['assemblyId']
    series.create_dataset('timeS', data=times, dtype=np.float64)
    stringType = h5py.string_dtype(encoding='utf-8')
    series.create_dataset(
        'sourceResultMember', data=np.asarray(tuple(members), dtype=object), dtype=stringType
    )
    context.temperatureDataset = series.create_dataset(
        'nodeTemperatureK',
        shape=(times.size, context.nodesMeters.shape[0]),
        dtype=np.float64,
        chunks=(1, _chunkWidth(context.nodesMeters.shape[0])),
        compression='lzf',
        shuffle=True,
    )
    context.boundaryFluxDataset = series.create_dataset(
        'boundaryHeatFluxWPerM2',
        shape=(times.size, context.boundarySourceNodes.size),
        dtype=np.float64,
        chunks=(1, _chunkWidth(context.boundarySourceNodes.size)),
        compression='lzf',
        shuffle=True,
    )
    context.seriesHandle = series


def _frameFields(resultFile: Any, context: _componentContext) -> tuple[np.ndarray, np.ndarray]:
    temperaturePath = f'{context.resultBase}/temperature'
    heatFlowPath = f'{context.resultBase}/heatFlowRate'
    if temperaturePath not in resultFile or heatFlowPath not in resultFile:
        raise KeyError(f'transient result lacks fields for {context.manifest["caseId"]}')
    temperatures = np.asarray(resultFile[temperaturePath], dtype=np.float64).reshape(-1)
    nodalHeatFlow = np.asarray(resultFile[heatFlowPath], dtype=np.float64).reshape(-1)
    nodeCount = context.nodesMeters.shape[0]
    if temperatures.size != nodeCount or nodalHeatFlow.size != nodeCount:
        raise ValueError(
            f'transient node count changed for {context.manifest["caseId"]}: '
            f'expected {nodeCount}, found {temperatures.size}/{nodalHeatFlow.size}'
        )
    if not np.all(np.isfinite(temperatures)) or not np.all(np.isfinite(nodalHeatFlow)):
        raise ValueError(f'transient result contains non-finite fields: {context.manifest["caseId"]}')

    rawBoundaryPower = float(nodalHeatFlow[context.boundaryNodes].sum())
    overviewPath = f'{context.overviewBase}/results/heatFlowRate'
    overviewBoundaryPower = (
        float(np.asarray(resultFile[overviewPath], dtype=np.float64).sum())
        if overviewPath in resultFile
        else None
    )
    scale = 1.0
    if overviewBoundaryPower is not None and abs(rawBoundaryPower) > np.finfo(float).eps:
        candidate = -overviewBoundaryPower / rawBoundaryPower
        if 0.25 <= candidate <= 4.0:
            scale = candidate
    rawNodalHeatFlux = np.divide(
        nodalHeatFlow,
        context.nodalBoundaryAreasM2,
        out=np.zeros_like(nodalHeatFlow),
        where=context.nodalBoundaryAreasM2 > 0.0,
    )
    patchFlux = rawNodalHeatFlux[context.boundarySourceNodes] * scale
    overviewRates = _overviewHeatFlowRates(resultFile, context.overviewBase)
    for patch in context.patchMetadata:
        mask = context.boundaryPatchIds == int(patch['patchId'])
        if not np.any(mask):
            raise ValueError(
                f'transient patch has no boundary samples: '
                f'{context.manifest["caseId"]}/{patch["patchId"]}'
            )
        expectedOverview = overviewRates.get(str(patch['patchName']))
        expectedPower = (
            -float(expectedOverview)
            if expectedOverview is not None
            else float(np.dot(patchFlux[mask], context.boundaryWeightsM2[mask]))
        )
        calibrated = patchFlux[mask].copy()
        _matchIntegratedPower(calibrated, context.boundaryWeightsM2[mask], expectedPower)
        patchFlux[mask] = calibrated
    boundaryInputFlux = np.zeros_like(patchFlux)
    resultMask = context.boundaryTypeIds == RESULT_HEAT_FLUX_BOUNDARY_TYPE
    boundaryInputFlux[resultMask] = patchFlux[resultMask]
    prescribedMask = (
        context.boundaryTypeIds == PRESCRIBED_HEAT_FLUX_BOUNDARY_TYPE
    )
    boundaryInputFlux[prescribedMask] = context.finalBoundaryHeatFluxWPerM2[
        prescribedMask
    ]
    return temperatures, boundaryInputFlux


def _extractMember(archive: ZipFile, member: str, destination: Path) -> None:
    info = archive.getinfo(member)
    with archive.open(info) as source, destination.open('wb') as target:
        shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
    if destination.stat().st_size != info.file_size:
        raise IOError(f'incomplete transient result extraction: {member}')


def _sequenceFields(info: motorStaArchiveInfo) -> dict[str, Any]:
    times = np.asarray(info.availableResultCoordinates, dtype=np.float64)
    return {
        'analysisType': 'transient',
        'isTransient': True,
        'temporalSampleType': 'transientSequence',
        'timeSeriesStored': True,
        'seriesFile': 'series.h5',
        'timeStepCount': int(times.size),
        'timeRangeS': [float(times[0]), float(times[-1])],
        'terminalSnapshotTimeS': float(times[-1]),
        'sourceTimeStepS': info.sourceTimeStepS,
        'sourceEndTimeS': info.sourceEndTimeS,
        'sourceSnapshotCount': int(times.size),
    }


def _updateManifestAndEntry(
    context: _componentContext,
    fields: dict[str, Any],
) -> None:
    manifest = context.manifest
    manifest.pop('snapshotTimeS', None)
    manifest.update(fields)
    operating = manifest.setdefault('operatingFeatures', {})
    operating.pop('snapshotTimeS', None)
    metadata = manifest.setdefault('metadata', {})
    metadata.pop('snapshotTimeS', None)
    metadata.update(fields)
    _writeJson(context.manifestPath, manifest)
    context.entry.pop('snapshotTimeS', None)
    context.entry.update(
        {
            key: fields[key]
            for key in (
                'analysisType',
                'temporalSampleType',
                'timeSeriesStored',
                'seriesFile',
                'timeStepCount',
                'timeRangeS',
                'terminalSnapshotTimeS',
            )
        }
    )


def buildMotorTransientSeriesFromSta(
    source: str | Path,
    graphViewRoot: str | Path,
    pointViewRoot: str | Path,
    graphEntries: Sequence[dict[str, Any]],
    pointEntries: Sequence[dict[str, Any]],
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    info = inspectMotorStaArchive(source)
    if info.isTransient is not True:
        raise ValueError(f'STA archive is not transient: {info.sourcePath}')
    if len(info.availableResultCoordinates) < 2:
        raise ValueError('transient sequence requires at least two result times')
    times = np.asarray(info.availableResultCoordinates, dtype=np.float64)
    if np.any(~np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        raise ValueError('transient result times must be finite and strictly increasing')
    if len(info.availableResultMembers) != times.size:
        raise ValueError('transient result times and members are not aligned')

    graphRoot = Path(graphViewRoot)
    pointRoot = Path(pointViewRoot)
    graphByInstance = {entry['componentInstance']: entry for entry in graphEntries}
    pointByInstance = {entry['componentInstance']: entry for entry in pointEntries}
    selectedComponents = set(graphByInstance)
    if selectedComponents != set(pointByInstance):
        raise ValueError('transient graph and point views contain different components')
    if not selectedComponents or not selectedComponents.issubset(set(info.componentNames)):
        raise ValueError('transient dataset components are not a non-empty STA FEM subset')
    contexts = [
        _loadContext(graphRoot, graphByInstance[name], times.size)
        for name in info.componentNames
        if name in selectedComponents
    ]
    occupied = [context.finalSeriesPath for context in contexts if context.finalSeriesPath.exists()]
    if occupied and not overwrite:
        raise FileExistsError(f'transient series already exists: {occupied[0]}')

    try:
        for context in contexts:
            _openSeries(context, times, info.availableResultMembers)
        try:
            h5py = _h5py()
            with tempfile.TemporaryDirectory(prefix='.transient-results-', dir=graphRoot.parent) as temporary:
                extracted = Path(temporary) / 'result.h5'
                with ZipFile(info.sourcePath) as archive:
                    for frameIndex, member in enumerate(info.availableResultMembers):
                        _extractMember(archive, member, extracted)
                        with h5py.File(extracted, 'r') as resultFile:
                            for context in contexts:
                                temperatures, boundaryFlux = _frameFields(resultFile, context)
                                context.temperatureDataset[frameIndex] = temperatures
                                context.boundaryFluxDataset[frameIndex] = boundaryFlux
                        extracted.unlink()
            for context in contexts:
                context.seriesHandle.flush()
                terminalFlux = np.asarray(context.boundaryFluxDataset[-1], dtype=np.float64)
                np.testing.assert_allclose(
                    terminalFlux,
                    context.finalBoundaryHeatFluxWPerM2,
                    rtol=5e-11,
                    atol=1e-9,
                    err_msg=f'terminal boundary flux mismatch: {context.manifest["caseId"]}',
                )
        finally:
            for context in contexts:
                if context.seriesHandle is not None:
                    context.seriesHandle.close()
    except Exception:
        for context in contexts:
            if context.seriesHandle is not None:
                try:
                    context.seriesHandle.close()
                except Exception:
                    pass
            context.temporarySeriesPath.unlink(missing_ok=True)
        raise

    fields = _sequenceFields(info)
    try:
        for context in contexts:
            context.temporarySeriesPath.replace(context.finalSeriesPath)
            _updateManifestAndEntry(context, fields)
        for context in contexts:
            pointEntry = pointByInstance[context.entry['componentInstance']]
            pointDirectory = pointRoot / pointEntry['path']
            pointSeries = pointDirectory / 'series.h5'
            pointTemporary = pointSeries.with_suffix('.h5.tmp')
            shutil.copy2(context.finalSeriesPath, pointTemporary)
            pointTemporary.replace(pointSeries)
            pointManifestPath = pointDirectory / 'manifest.json'
            pointManifest = json.loads(pointManifestPath.read_text(encoding='utf-8'))
            pointManifest.pop('snapshotTimeS', None)
            pointManifest.update(fields)
            pointManifest.setdefault('operatingFeatures', {}).pop('snapshotTimeS', None)
            pointMetadata = pointManifest.setdefault('metadata', {})
            pointMetadata.pop('snapshotTimeS', None)
            pointMetadata.update(fields)
            _writeJson(pointManifestPath, pointManifest)
            pointEntry.pop('snapshotTimeS', None)
            pointEntry.update(
                {
                    key: fields[key]
                    for key in (
                        'analysisType',
                        'temporalSampleType',
                        'timeSeriesStored',
                        'timeStepCount',
                        'timeRangeS',
                        'terminalSnapshotTimeS',
                    )
                }
            )
    except Exception:
        for context in contexts:
            context.temporarySeriesPath.unlink(missing_ok=True)
        raise
    return {
        'sourceSta': str(info.sourcePath.resolve()),
        'timeStepCount': int(times.size),
        'timeRangeS': [float(times[0]), float(times[-1])],
        'componentCount': len(contexts),
        'temporalMetadata': fields,
    }


def loadMotorTransientSeriesRecords(
    dataset: str | Path,
) -> tuple[motorTransientSeriesRecord, ...]:
    source = Path(dataset)
    indexPath = source / 'dataset.json' if source.is_dir() else source
    index = json.loads(indexPath.read_text(encoding='utf-8'))
    records = []
    for entry in index.get('cases', ()):
        if entry.get('analysisType') != 'transient' or not entry.get('timeSeriesStored'):
            continue
        caseDirectory = indexPath.parent / entry['path']
        seriesFile = str(entry.get('seriesFile', 'series.h5'))
        seriesPath = caseDirectory / seriesFile
        if not seriesPath.is_file():
            raise FileNotFoundError(f'transient series is missing: {seriesPath}')
        records.append(
            motorTransientSeriesRecord(
                caseId=str(entry['caseId']),
                systemType=str(entry['systemType']),
                assemblyType=str(entry['assemblyType']),
                assemblyId=str(entry['assemblyId']),
                componentType=str(entry['componentType']),
                componentInstance=str(entry['componentInstance']),
                meshUuid=str(entry['meshUuid']),
                caseDirectory=caseDirectory,
                seriesFile=seriesFile,
                timeStepCount=int(entry['timeStepCount']),
            )
        )
    return tuple(records)


def _relativeEntry(entry: dict[str, Any], prefix: Path) -> dict[str, Any]:
    return {**entry, 'path': (prefix / Path(entry['path'])).as_posix()}


def _thermalTask(analysisType: str) -> dict[str, Any]:
    if analysisType == 'steady':
        return {
            'taskType': 'fieldRegression',
            'inputFile': 'data.npz',
            'targetArray': 'nodeTemperatureK',
            'targetScope': 'node',
        }
    return {
        'taskType': 'sequenceFieldRegression',
        'seriesFile': 'series.h5',
        'timeArray': 'timeS',
        'targetArray': 'nodeTemperatureK',
        'boundaryInputArray': 'boundaryHeatFluxWPerM2',
        'targetScope': 'node',
        'meshPolicy': 'fixedWithinSequence',
    }


def writeMotorThermalAnalysisIndexes(datasetRoot: str | Path) -> Path:
    root = Path(datasetRoot)
    rootViewsPath = root / 'views.json'
    rootViews = json.loads(rootViewsPath.read_text(encoding='utf-8'))
    viewPaths = {}
    datasetAnalysisTypes = None
    for view in ('point', 'graph'):
        viewRoot = root / view
        sourceIndexPath = viewRoot / 'dataset.json'
        sourceIndex = json.loads(sourceIndexPath.read_text(encoding='utf-8'))
        currentAnalysisTypes = sorted(
            {entry.get('analysisType', 'unknown') for entry in sourceIndex.get('cases', ())}
        )
        if datasetAnalysisTypes is not None and currentAnalysisTypes != datasetAnalysisTypes:
            raise ValueError('point and graph views have different analysis types')
        datasetAnalysisTypes = currentAnalysisTypes
        wholeSourcePath = viewRoot / 'whole_motor' / 'dataset.json'
        wholeSource = json.loads(wholeSourcePath.read_text(encoding='utf-8'))
        componentByParent = {}
        for entry in sourceIndex.get('cases', ()):
            componentByParent.setdefault(entry['parentCaseId'], entry)

        for analysisType in ('steady', 'transient'):
            entries = [
                _relativeEntry(entry, Path('..'))
                for entry in sourceIndex.get('cases', ())
                if entry.get('analysisType') == analysisType
            ]
            if analysisType == 'transient':
                incomplete = [entry['caseId'] for entry in entries if not entry.get('timeSeriesStored')]
                if incomplete:
                    raise ValueError(
                        f'transient analysis index cannot include snapshot-only cases: {incomplete[:3]}'
                    )
            analysisRoot = viewRoot / analysisType
            metadata = {
                **sourceIndex.get('metadata', {}),
                'analysisType': analysisType,
                'analysisTypes': [analysisType],
                'motorIds': sorted({entry['motorId'] for entry in entries}),
                'parentCaseIds': sorted({entry['parentCaseId'] for entry in entries}),
                'componentCaseCount': len(entries),
            }
            metadata['sourceArchives'] = sorted(
                {
                    str(wholeEntry.get('metadata', {}).get('sourceSta'))
                    for wholeEntry in wholeSource.get('cases', ())
                    if wholeEntry['parentCaseId'] in metadata['parentCaseIds']
                    and wholeEntry.get('metadata', {}).get('sourceSta')
                }
            )
            metadata.pop('analysisViews', None)
            metadata['multiMotor'] = len(metadata['motorIds']) > 1
            tasks = dict(sourceIndex.get('tasks', {}))
            tasks[
                'thermalSteadyFieldPrediction'
                if analysisType == 'steady'
                else 'thermalTransientSequencePrediction'
            ] = _thermalTask(analysisType)
            _writeJson(
                analysisRoot / 'dataset.json',
                {
                    'formatVersion': sourceIndex.get('formatVersion'),
                    'viewType': sourceIndex.get('viewType'),
                    'analysisType': analysisType,
                    'connectivityStored': sourceIndex.get('connectivityStored'),
                    'selfContainedCases': True,
                    'metadata': metadata,
                    'tasks': tasks,
                    'cases': entries,
                },
            )

            wholeEntries = []
            for sourceEntry in wholeSource.get('cases', ()):
                componentEntry = componentByParent[sourceEntry['parentCaseId']]
                if componentEntry.get('analysisType') != analysisType:
                    continue
                temporalFields = {
                    key: componentEntry[key]
                    for key in (
                        'analysisType',
                        'temporalSampleType',
                        'timeSeriesStored',
                        'seriesFile',
                        'timeStepCount',
                        'timeRangeS',
                        'terminalSnapshotTimeS',
                    )
                    if key in componentEntry
                }
                for key in (
                    'analysisType',
                    'temporalSampleType',
                    'timeSeriesStored',
                    'timeStepCount',
                    'timeRangeS',
                    'terminalSnapshotTimeS',
                ):
                    sourceEntry.pop(key, None)
                sourceEntry.update(temporalFields)
                sourceEntry.setdefault('metadata', {}).update(temporalFields)
                wholeEntries.append(
                    {
                        **sourceEntry,
                        **temporalFields,
                        'path': (
                            Path('..') / '..' / 'whole_motor' / Path(sourceEntry['path'])
                        ).as_posix(),
                    }
                )
                wholeManifestPath = viewRoot / 'whole_motor' / sourceEntry['path'] / 'manifest.json'
                wholeManifest = json.loads(wholeManifestPath.read_text(encoding='utf-8'))
                for key in (
                    'analysisType',
                    'temporalSampleType',
                    'timeSeriesStored',
                    'timeStepCount',
                    'timeRangeS',
                    'terminalSnapshotTimeS',
                ):
                    wholeManifest.pop(key, None)
                wholeManifest.update(temporalFields)
                if analysisType == 'transient':
                    wholeManifest['seriesStorage'] = 'componentCaseFiles'
                wholeManifest.setdefault('metadata', {}).update(temporalFields)
                _writeJson(wholeManifestPath, wholeManifest)
            _writeJson(
                analysisRoot / 'whole_motor' / 'dataset.json',
                {
                    'formatVersion': wholeSource.get('formatVersion'),
                    'viewType': 'wholeMotor',
                    'analysisType': analysisType,
                    'metadata': metadata,
                    'cases': wholeEntries,
                },
            )
            componentKey = f'{view}{analysisType.capitalize()}'
            wholeKey = f'{componentKey}WholeMotor'
            viewPaths[componentKey] = f'{view}/{analysisType}/dataset.json'
            viewPaths[wholeKey] = f'{view}/{analysisType}/whole_motor/dataset.json'

        sourceIndex.setdefault('metadata', {})['analysisViews'] = {
            'steady': 'steady/dataset.json',
            'transient': 'transient/dataset.json',
        }
        _writeJson(sourceIndexPath, sourceIndex)
        wholeSource.setdefault('metadata', {})['analysisTypes'] = sorted(
            {entry.get('analysisType', 'unknown') for entry in wholeSource.get('cases', ())}
        )
        _writeJson(wholeSourcePath, wholeSource)

    rootViews.setdefault('views', {}).update(viewPaths)
    rootViews['analysisTypes'] = datasetAnalysisTypes or []
    rootViews['temporalStoragePolicy'] = {
        'steady': 'single field snapshot in data.npz',
        'transient': 'fixed case-local mesh plus complete fields in series.h5',
        'meshPolicy': 'fixed within each transient sequence',
        'geometryAssetsShared': False,
    }
    _writeJson(rootViewsPath, rootViews)
    from .assemblyHierarchy import writeAssemblyHierarchyIndexes

    writeAssemblyHierarchyIndexes(root)
    return rootViewsPath

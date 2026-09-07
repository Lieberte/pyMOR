from __future__ import annotations

import json
import re
import shutil
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Sequence
from zipfile import ZipFile, ZipInfo

from .motorFemComponentCase import (
    CONVECTION_BOUNDARY_TYPE,
    FIXED_TEMPERATURE_BOUNDARY_TYPE,
    PRESCRIBED_HEAT_FLUX_BOUNDARY_TYPE,
    RESULT_HEAT_FLUX_BOUNDARY_TYPE,
    buildMotorFemComponentCase,
    _normalizedComponentName,
    _resolveFemSystem,
    _thermalAnalysisMetadata,
)
from .thermalComponentCase import saveThermalComponentDataset
from .motorWholeDataset import saveWholeMotorDataset


@dataclass(frozen=True)
class motorStaArchiveInfo:
    sourcePath: Path
    resultMember: str
    resultTime: float | None
    analysisType: str
    isTransient: bool | None
    sourceTimeStepS: float | None
    sourceEndTimeS: float | None
    availableResultCoordinates: tuple[float, ...]
    availableResultMembers: tuple[str, ...]
    componentNames: tuple[str, ...]
    meshMembers: tuple[str, ...]


@dataclass(frozen=True)
class motorStaDatasetBuild:
    indexPath: Path
    wholeMotorIndexPath: Path
    viewsPath: Path
    motorId: str
    parentCaseId: str
    resultMember: str
    componentCaseIds: tuple[str, ...]
    reusedExtractedDirectory: Path | None


def _identifier(value: str, name: str) -> str:
    normalized = re.sub(r'[^A-Za-z0-9_-]+', '-', str(value).strip()).strip('-_')
    if not normalized:
        raise ValueError(f'{name} cannot be normalized to an empty identifier')
    return normalized


def _safeMemberName(value: str) -> str:
    normalized = str(value).replace('\\', '/')
    path = PurePosixPath(normalized)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError(f'unsafe STA archive member path: {value!r}')
    return path.as_posix()


def _controlFromArchive(archive: ZipFile) -> dict:
    try:
        data = archive.read('controlData.json')
    except KeyError as error:
        raise FileNotFoundError('STA archive does not contain controlData.json') from error
    return json.loads(data.decode('utf-8-sig'))


def _selectedResultMember(
    archive: ZipFile,
    *,
    resultTime: float | None,
    resultMember: str | None,
    systemUuid: str | None = None,
) -> tuple[str, float | None, tuple[float, ...], tuple[str, ...]]:
    if resultTime is not None and resultMember is not None:
        raise ValueError('resultTime and resultMember cannot both be provided')
    names = {
        _safeMemberName(info.filename): info
        for info in archive.infolist()
        if not info.is_dir()
    }
    resultNames = tuple(
        name
        for name in names
        if PurePosixPath(name).parts[0] == 'results'
        and (
            len(PurePosixPath(name).parts) < 2
            or PurePosixPath(name).parts[1].lower() != 'iterationinfo'
        )
        and PurePosixPath(name).suffix.lower() == '.h5'
    )
    if systemUuid is not None:
        matchingSystem = tuple(
            name
            for name in resultNames
            if len(PurePosixPath(name).parts) >= 3
            and PurePosixPath(name).parts[1].lower() == str(systemUuid).lower()
        )
        if matchingSystem:
            resultNames = matchingSystem
    if not resultNames:
        raise ValueError(
            'STA archive does not contain solved HDF5 result files; '
            'results/iterationInfo contains iteration metadata only'
        )
    numericResults = []
    iterationResults: dict[float, list[tuple[float, str]]] = {}
    for name in resultNames:
        path = PurePosixPath(name)
        try:
            numericResults.append((float(path.stem), name))
        except ValueError:
            continue
        if len(path.parts) >= 4 and path.parts[1].lower() == 'iterationinfo':
            try:
                physicalTime = float(path.parts[-2])
                iteration = float(path.stem)
            except ValueError:
                continue
            iterationResults.setdefault(physicalTime, []).append((iteration, name))

    if iterationResults:
        selectedPhysicalResults = tuple(
            (time, max(candidates, key=lambda item: item[0])[1])
            for time, candidates in sorted(iterationResults.items())
        )
    else:
        selectedPhysicalResults = tuple(sorted(numericResults))

    if resultMember is not None:
        selected = _safeMemberName(resultMember)
        if selected not in resultNames:
            raise FileNotFoundError(f'result member is not present in STA archive: {selected}')
    else:
        if resultTime is not None:
            matches = [
                name for time, name in selectedPhysicalResults if time == float(resultTime)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f'expected one result at time {float(resultTime):g}, found {len(matches)}'
                )
            selected = matches[0]
        elif len(resultNames) == 1:
            selected = resultNames[0]
        elif selectedPhysicalResults:
            selected = selectedPhysicalResults[-1][1]
        else:
            raise ValueError(
                'STA archive contains multiple non-numeric result files; provide resultMember'
            )
    selectedTimes = [time for time, name in selectedPhysicalResults if name == selected]
    if selectedTimes:
        selectedTime = selectedTimes[0]
    else:
        try:
            selectedTime = float(PurePosixPath(selected).stem)
        except ValueError:
            selectedTime = None
    return (
        selected,
        selectedTime,
        tuple(time for time, _ in selectedPhysicalResults),
        tuple(name for _, name in selectedPhysicalResults),
    )


def inspectMotorStaArchive(
    source: str | Path,
    *,
    resultTime: float | None = None,
    resultMember: str | None = None,
) -> motorStaArchiveInfo:
    sourcePath = Path(source)
    if not sourcePath.is_file():
        raise FileNotFoundError(f'STA archive does not exist: {sourcePath}')
    with ZipFile(sourcePath) as archive:
        control = _controlFromArchive(archive)
        femSystemUuid, femSystem, _ = _resolveFemSystem(control)
        regions = tuple(femSystem.get('femRegions', {}).values())
        if not regions:
            raise ValueError('3D FEM system does not contain any regions')
        componentNames = tuple(
            _normalizedComponentName(region.get('name', '')) for region in regions
        )
        if len(set(componentNames)) != len(componentNames):
            raise ValueError(f'normalized FEM component names must be unique: {componentNames}')
        meshMembers = tuple(f"mesh/{region['meshUuid']}.samesh" for region in regions)
        archiveNames = {
            _safeMemberName(info.filename) for info in archive.infolist() if not info.is_dir()
        }
        missingMeshes = tuple(name for name in meshMembers if name not in archiveNames)
        if missingMeshes:
            raise FileNotFoundError(f'STA archive is missing referenced FEM meshes: {missingMeshes}')
        selectedResult, selectedTime, resultCoordinates, resultMembers = _selectedResultMember(
            archive,
            resultTime=resultTime,
            resultMember=resultMember,
            systemUuid=femSystemUuid,
        )
        analysis = _thermalAnalysisMetadata(
            control,
            snapshotTimeS=(
                selectedTime
                if control.get('solverSetting', {}).get('timeDomain', {}).get('isTransient')
                is True
                else None
            ),
        )
    return motorStaArchiveInfo(
        sourcePath=sourcePath,
        resultMember=selectedResult,
        resultTime=selectedTime,
        analysisType=analysis['analysisType'],
        isTransient=analysis['isTransient'],
        sourceTimeStepS=analysis.get('sourceTimeStepS'),
        sourceEndTimeS=analysis.get('sourceEndTimeS'),
        availableResultCoordinates=resultCoordinates,
        availableResultMembers=resultMembers,
        componentNames=componentNames,
        meshMembers=meshMembers,
    )


def _staTemporalMetadata(info: motorStaArchiveInfo) -> dict:
    metadata = {
        'analysisType': info.analysisType,
        'isTransient': info.isTransient,
        'temporalSampleType': (
            'transientSnapshot'
            if info.isTransient is True
            else 'steadySolution'
            if info.isTransient is False
            else 'unknownSnapshot'
        ),
        'timeSeriesStored': False,
    }
    if info.isTransient is True:
        metadata.update(
            {
                'sourceTimeStepS': info.sourceTimeStepS,
                'sourceEndTimeS': info.sourceEndTimeS,
                'snapshotTimeS': info.resultTime,
                'sourceSnapshotCount': len(info.availableResultCoordinates),
            }
        )
    return metadata


def _requiredMembers(info: motorStaArchiveInfo) -> tuple[str, ...]:
    return (
        'controlData.json',
        'controlData.h5',
        *info.meshMembers,
        info.resultMember,
    )


def _completeExtractedDirectory(
    directory: Path,
    archive: ZipFile,
    members: Sequence[str],
) -> bool:
    if not directory.is_dir():
        return False
    for member in members:
        path = directory.joinpath(*PurePosixPath(member).parts)
        try:
            expectedSize = archive.getinfo(member).file_size
        except KeyError:
            return False
        if not path.is_file() or path.stat().st_size != expectedSize:
            return False
    return True


def _extractMember(archive: ZipFile, info: ZipInfo, root: Path) -> None:
    member = _safeMemberName(info.filename)
    target = root.joinpath(*PurePosixPath(member).parts)
    if target.is_file():
        if target.stat().st_size != info.file_size:
            raise FileExistsError(
                f'extracted STA member has an unexpected size and will not be overwritten: {target}'
            )
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(info) as sourceFile, target.open('wb') as targetFile:
        shutil.copyfileobj(sourceFile, targetFile, length=8 * 1024 * 1024)


def _prepareExtractedDirectory(
    archive: ZipFile,
    info: motorStaArchiveInfo,
    extractedDirectory: str | Path | None,
    temporaryParent: Path,
) -> tuple[Path, tempfile.TemporaryDirectory[str] | None, Path | None]:
    members = _requiredMembers(info)
    sibling = info.sourcePath.with_suffix('')
    if extractedDirectory is None and _completeExtractedDirectory(sibling, archive, members):
        return sibling, None, sibling
    if extractedDirectory is None:
        temporaryParent.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(
            prefix=f'{info.sourcePath.stem}-sta-',
            dir=temporaryParent,
        )
        root = Path(temporary.name)
    else:
        temporary = None
        root = Path(extractedDirectory)
        root.mkdir(parents=True, exist_ok=True)
    for member in members:
        try:
            memberInfo = archive.getinfo(member)
        except KeyError as error:
            raise FileNotFoundError(f'STA archive is missing required member: {member}') from error
        _extractMember(archive, memberInfo, root)
    return root, temporary, None


def buildMotorDatasetFromSta(
    source: str | Path,
    outputDirectory: str | Path,
    *,
    motorId: str | None = None,
    parentCaseId: str | None = None,
    components: Sequence[str] | None = None,
    resultTime: float | None = None,
    resultMember: str | None = None,
    extractedDirectory: str | Path | None = None,
    maximumVolumeSamples: int | None = None,
    maximumBoundarySamples: int | None = None,
    maximumQuerySamples: int | None = None,
    randomSeed: int = 0,
    overwrite: bool = False,
) -> motorStaDatasetBuild:
    outputRoot = Path(outputDirectory)
    existingOutputs = (
        outputRoot / 'dataset.json',
        outputRoot / 'whole_motor' / 'dataset.json',
        outputRoot / 'views.json',
    )
    if not overwrite and any(path.exists() for path in existingOutputs):
        raise FileExistsError(f'thermal component dataset already exists: {outputRoot}')
    info = inspectMotorStaArchive(
        source,
        resultTime=resultTime,
        resultMember=resultMember,
    )
    resolvedMotorId = _identifier(motorId or info.sourcePath.stem, 'motorId')
    resolvedParentCaseId = _identifier(parentCaseId or resolvedMotorId, 'parentCaseId')
    temporalMetadata = _staTemporalMetadata(info)
    selectedComponents = tuple(components) if components is not None else info.componentNames
    if not selectedComponents:
        raise ValueError('at least one FEM component must be selected')
    if len(set(selectedComponents)) != len(selectedComponents):
        raise ValueError(f'FEM component selection contains duplicates: {selectedComponents}')

    with ZipFile(info.sourcePath) as archive:
        extractedRoot, temporary, reusedDirectory = _prepareExtractedDirectory(
            archive,
            info,
            extractedDirectory,
            outputRoot.parent,
        )
        try:
            cases = []
            for componentIndex, component in enumerate(selectedComponents):
                case = buildMotorFemComponentCase(
                    extractedRoot,
                    component,
                    maximumVolumeSamples=maximumVolumeSamples,
                    maximumBoundarySamples=maximumBoundarySamples,
                    maximumQuerySamples=maximumQuerySamples,
                    randomSeed=int(randomSeed) + componentIndex,
                    resultFile=info.resultMember,
                    motorId=resolvedMotorId,
                    parentCaseId=resolvedParentCaseId,
                )
                cases.append(
                    replace(
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
                            'motorId': resolvedMotorId,
                            'parentCaseId': resolvedParentCaseId,
                            'sourceCase': info.sourcePath.name,
                            'sourceSta': str(info.sourcePath.resolve()),
                            'resultFile': info.resultMember,
                            'resultTime': info.resultTime,
                        },
                    )
                )
            indexPath = saveThermalComponentDataset(
                tuple(cases),
                outputRoot,
                boundaryTypeNames={
                    RESULT_HEAT_FLUX_BOUNDARY_TYPE: 'resultHeatFlux',
                    CONVECTION_BOUNDARY_TYPE: 'convection',
                    FIXED_TEMPERATURE_BOUNDARY_TYPE: 'fixedTemperature',
                    PRESCRIBED_HEAT_FLUX_BOUNDARY_TYPE: 'prescribedHeatFlux',
                },
                datasetMetadata={
                    'datasetType': 'singleMotorFemComponents',
                    'motorId': resolvedMotorId,
                    'parentCaseId': resolvedParentCaseId,
                    'sourceSta': str(info.sourcePath.resolve()),
                    'selectedResultMember': info.resultMember,
                    'selectedResultTime': info.resultTime,
                    **temporalMetadata,
                    'availableComponents': info.componentNames,
                    'selectedComponents': tuple(
                        case.metadata.get('componentInstance', case.componentType)
                        for case in cases
                    ),
                },
                overwrite=overwrite,
            )
            sharedMetadata = {
                'sourceSta': str(info.sourcePath.resolve()),
                'selectedResultMember': info.resultMember,
                'selectedResultTime': info.resultTime,
                **temporalMetadata,
                'availableComponents': info.componentNames,
                'selectedComponents': tuple(
                    case.metadata.get('componentInstance', case.componentType)
                    for case in cases
                ),
                'completeComponentSet': set(
                    case.metadata.get('componentInstance', case.componentType)
                    for case in cases
                )
                == set(info.componentNames),
            }
            wholeMotorIndexPath = saveWholeMotorDataset(
                tuple(cases),
                outputRoot / 'whole_motor',
                componentDatasetDirectory=outputRoot,
                motorId=resolvedMotorId,
                parentCaseId=resolvedParentCaseId,
                metadata=sharedMetadata,
                overwrite=overwrite,
            )
            viewsPath = outputRoot / 'views.json'
            viewsPath.write_text(
                json.dumps(
                    {
                        'formatVersion': 'motor-dataset-views-v1',
                        'motorId': resolvedMotorId,
                        'parentCaseId': resolvedParentCaseId,
                        'views': {
                            'component': 'dataset.json',
                            'wholeMotor': 'whole_motor/dataset.json',
                        },
                    },
                    indent=2,
                ),
                encoding='utf-8',
            )
        finally:
            if temporary is not None:
                temporary.cleanup()
    return motorStaDatasetBuild(
        indexPath=indexPath,
        wholeMotorIndexPath=wholeMotorIndexPath,
        viewsPath=viewsPath,
        motorId=resolvedMotorId,
        parentCaseId=resolvedParentCaseId,
        resultMember=info.resultMember,
        componentCaseIds=tuple(case.caseId for case in cases),
        reusedExtractedDirectory=reusedDirectory,
    )

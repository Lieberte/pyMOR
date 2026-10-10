from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .motorFemComponentCase import (
    buildMotorFemComponentCase,
    motorFemComponentNames,
)
from .motorFemBoundary import (
    CONVECTION_BOUNDARY_TYPE,
    FIXED_TEMPERATURE_BOUNDARY_TYPE,
    PRESCRIBED_HEAT_FLUX_BOUNDARY_TYPE,
    RESULT_HEAT_FLUX_BOUNDARY_TYPE,
)
from .motorWholeDataset import saveWholeMotorDataset
from .thermalComponentCase import saveThermalComponentDataset


@dataclass(frozen=True)
class motorDirectoryDatasetBuild:
    componentIndexPath: Path
    wholeMotorIndexPath: Path
    viewsPath: Path
    motorId: str
    parentCaseIds: tuple[str, ...]
    componentCaseIds: tuple[str, ...]


def _identifier(value: str, name: str) -> str:
    normalized = re.sub(r'[^A-Za-z0-9_-]+', '-', str(value).strip()).strip('-_')
    if not normalized:
        raise ValueError(f'{name} cannot be normalized to an empty identifier')
    return normalized


def _numericCaseDirectories(root: Path) -> tuple[Path, ...]:
    directories = tuple(
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.replace('.', '', 1).isdigit()
    )
    return tuple(sorted(directories, key=lambda path: float(path.name)))


def buildMotorDatasetFromDirectories(
    sourceDirectory: str | Path,
    outputDirectory: str | Path,
    *,
    motorId: str,
    caseIds: Sequence[str] | None = None,
    components: Sequence[str] | None = None,
    maximumVolumeSamples: int | None = None,
    maximumBoundarySamples: int | None = None,
    maximumQuerySamples: int | None = None,
    randomSeed: int = 0,
    overwrite: bool = False,
) -> motorDirectoryDatasetBuild:
    sourceRoot = Path(sourceDirectory)
    if not sourceRoot.is_dir():
        raise FileNotFoundError(f'motor case directory does not exist: {sourceRoot}')
    availableDirectories = _numericCaseDirectories(sourceRoot)
    selectedNames = (
        tuple(str(value) for value in caseIds)
        if caseIds is not None
        else tuple(path.name for path in availableDirectories)
    )
    if not selectedNames:
        raise ValueError(f'no numeric motor case directories were selected below {sourceRoot}')
    directoryByName = {path.name: path for path in availableDirectories}
    missing = tuple(name for name in selectedNames if name not in directoryByName)
    if missing:
        raise FileNotFoundError(f'motor case directories are missing: {missing}')
    if len(set(selectedNames)) != len(selectedNames):
        raise ValueError(f'motor case selection contains duplicates: {selectedNames}')

    outputRoot = Path(outputDirectory)
    existingOutputs = (
        outputRoot / 'dataset.json',
        outputRoot / 'whole_motor' / 'dataset.json',
        outputRoot / 'views.json',
    )
    if not overwrite and any(path.exists() for path in existingOutputs):
        raise FileExistsError(f'motor dataset already exists: {outputRoot}')

    resolvedMotorId = _identifier(motorId, 'motorId')
    cases = []
    groupedCases = []
    caseComponents = {}
    for caseIndex, name in enumerate(selectedNames):
        caseRoot = directoryByName[name]
        selectedComponents = (
            tuple(components) if components is not None else motorFemComponentNames(caseRoot)
        )
        if not selectedComponents:
            raise ValueError(f'no FEM components were selected for motor case {name!r}')
        componentCases = []
        for componentIndex, component in enumerate(selectedComponents):
            case = buildMotorFemComponentCase(
                caseRoot,
                component,
                maximumVolumeSamples=maximumVolumeSamples,
                maximumBoundarySamples=maximumBoundarySamples,
                maximumQuerySamples=maximumQuerySamples,
                randomSeed=int(randomSeed) + caseIndex * 1000 + componentIndex,
                motorId=resolvedMotorId,
                parentCaseId=name,
            )
            componentCases.append(case)
            cases.append(case)
        groupedCases.append((name, tuple(componentCases)))
        caseComponents[name] = tuple(
            case.metadata.get('componentInstance', case.componentType)
            for case in componentCases
        )

    componentIndexPath = saveThermalComponentDataset(
        tuple(cases),
        outputRoot,
        boundaryTypeNames={
            RESULT_HEAT_FLUX_BOUNDARY_TYPE: 'resultHeatFlux',
            CONVECTION_BOUNDARY_TYPE: 'convection',
            FIXED_TEMPERATURE_BOUNDARY_TYPE: 'fixedTemperature',
            PRESCRIBED_HEAT_FLUX_BOUNDARY_TYPE: 'prescribedHeatFlux',
        },
        datasetMetadata={
            'datasetType': 'motorFemComponentCasesFromDirectories',
            'motorId': resolvedMotorId,
            'sourceDirectory': str(sourceRoot.resolve()),
            'parentCaseIds': selectedNames,
            'componentsByParentCase': caseComponents,
        },
        overwrite=overwrite,
    )

    wholeMotorIndexPath = outputRoot / 'whole_motor' / 'dataset.json'
    for caseIndex, (name, componentCases) in enumerate(groupedCases):
        wholeMotorIndexPath = saveWholeMotorDataset(
            componentCases,
            outputRoot / 'whole_motor',
            componentDatasetDirectory=outputRoot,
            motorId=resolvedMotorId,
            parentCaseId=name,
            metadata={
                'sourceDirectory': str((sourceRoot / name).resolve()),
                'sourceCase': name,
                'selectedComponents': caseComponents[name],
            },
            overwrite=overwrite,
            append=caseIndex > 0,
        )

    viewsPath = outputRoot / 'views.json'
    viewsPath.write_text(
        json.dumps(
            {
                'formatVersion': 'motor-dataset-views-v1',
                'motorId': resolvedMotorId,
                'parentCaseIds': list(selectedNames),
                'views': {
                    'component': 'dataset.json',
                    'wholeMotor': 'whole_motor/dataset.json',
                },
            },
            indent=2,
        ),
        encoding='utf-8',
    )
    return motorDirectoryDatasetBuild(
        componentIndexPath=componentIndexPath,
        wholeMotorIndexPath=wholeMotorIndexPath,
        viewsPath=viewsPath,
        motorId=resolvedMotorId,
        parentCaseIds=tuple(selectedNames),
        componentCaseIds=tuple(case.caseId for case in cases),
    )

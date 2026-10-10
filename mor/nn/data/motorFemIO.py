from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def loadH5py():
    import h5py
    return h5py


def singleValue(dataset: Any, name: str) -> float:
    values = np.asarray(dataset, dtype=np.float64).reshape(-1)
    if values.size != 1 or not np.isfinite(values[0]):
        raise ValueError(f'{name} must contain one finite value')
    return float(values[0])


def fixedIsotropicMaterialProperty(controlFile: Any, materialUuid: str, name: str) -> float:
    path = f'{materialUuid}/{name}/isotropic/fixedValue'
    if path not in controlFile:
        raise KeyError(f'material property is missing from controlData.h5: {path}')
    return singleValue(controlFile[path], path)


def fixedMaterialConductivityTensor(controlFile: Any, materialUuid: str) -> np.ndarray:
    base = f'{materialUuid}/thermalConductivity'
    componentNames = ('XX', 'XY', 'XZ', 'YX', 'YY', 'YZ', 'ZX', 'ZY', 'ZZ')
    componentPaths = tuple(f'{base}/anisotropic/{name}/fixedValue' for name in componentNames)
    if all(path in controlFile for path in componentPaths):
        values = np.asarray(
            [singleValue(controlFile[path], path) for path in componentPaths],
            dtype=np.float64,
        ).reshape(3, 3)
        if np.any(np.abs(values) > np.finfo(np.float64).eps):
            if not np.allclose(values, values.T):
                raise ValueError('anisotropic thermal conductivity tensor must be symmetric')
            if np.any(np.linalg.eigvalsh(values) <= 0.0):
                raise ValueError(
                    'anisotropic thermal conductivity tensor must be positive definite'
                )
            return values

    principalPaths = tuple(
        f'{base}/orthotropic/propertyX{axis}/fixedValue' for axis in (1, 2, 3)
    )
    directionPaths = tuple(f'{base}/orthotropic/x{axis}' for axis in (1, 2, 3))
    if all(path in controlFile for path in (*principalPaths, *directionPaths)):
        principalValues = np.asarray(
            [singleValue(controlFile[path], path) for path in principalPaths],
            dtype=np.float64,
        )
        if np.all(principalValues > 0.0):
            directions = np.column_stack(
                [
                    np.asarray(controlFile[path], dtype=np.float64).reshape(-1)
                    for path in directionPaths
                ]
            )
            if directions.shape != (3, 3) or not np.all(np.isfinite(directions)):
                raise ValueError('orthotropic conductivity directions must form a 3x3 matrix')
            if not np.allclose(directions.T @ directions, np.eye(3), atol=1e-8):
                raise ValueError('orthotropic conductivity directions must be orthonormal')
            return directions @ np.diag(principalValues) @ directions.T

    isotropic = fixedIsotropicMaterialProperty(
        controlFile, materialUuid, 'thermalConductivity'
    )
    if isotropic <= 0.0:
        raise ValueError('isotropic thermal conductivity must be positive')
    return np.eye(3) * isotropic


def decodedStrings(values: Any) -> tuple[str, ...]:
    return tuple(
        value.decode('utf-8') if isinstance(value, bytes) else str(value)
        for value in np.asarray(values).reshape(-1)
    )


def overviewHeatFlowRates(resultFile: Any, overviewBase: str) -> dict[str, float]:
    identifiersPath = f'{overviewBase}/locationIDs/heatFlowRate'
    valuesPath = f'{overviewBase}/results/heatFlowRate'
    if identifiersPath not in resultFile or valuesPath not in resultFile:
        return {}
    identifiers = decodedStrings(resultFile[identifiersPath])
    values = np.asarray(resultFile[valuesPath], dtype=np.float64).reshape(-1)
    if len(identifiers) != values.size or not np.all(np.isfinite(values)):
        raise ValueError('overview heat-flow identifiers and values must have matching sizes')
    return {identifier: float(value) for identifier, value in zip(identifiers, values)}


def connectionEndpoint(value: Any) -> tuple[str, str]:
    decoded = (
        bytes(value).decode('utf-8', errors='replace')
        if isinstance(value, (bytes, np.bytes_))
        else str(value)
    )
    componentUuid, separator, remainder = decoded.partition('@')
    patchUuid = remainder.split('&', 1)[0] if separator else ''
    return componentUuid, patchUuid


def connectionHeatFlowRates(resultFile: Any, regionUuid: str) -> dict[str, float]:
    """Return signed solver-connection heat flow for each patch of one FEM region.

    The result connection table stores ``left endpoint, right endpoint, power``
    triples. Its sign convention is positive from left to right, so heat entering
    the left component is negative and heat entering the right component is
    positive. This is the same sign convention used by the component PDE balance,
    ``volume source + boundary heat flow = 0``.
    """

    path = 'connections/heatFlowRate'
    if path not in resultFile:
        return {}
    values = np.asarray(resultFile[path]).reshape(-1)
    if values.size % 3:
        raise ValueError(
            'connections/heatFlowRate must contain endpoint pairs and heat-flow values'
        )
    result: dict[str, float] = {}
    for leftRaw, rightRaw, powerRaw in values.reshape(-1, 3):
        leftComponent, leftPatch = connectionEndpoint(leftRaw)
        rightComponent, rightPatch = connectionEndpoint(rightRaw)
        decodedPower = (
            bytes(powerRaw).decode('utf-8', errors='replace')
            if isinstance(powerRaw, (bytes, np.bytes_))
            else str(powerRaw)
        )
        power = float(decodedPower)
        if not np.isfinite(power):
            raise ValueError('connections/heatFlowRate must contain finite values')
        if leftComponent == str(regionUuid) and leftPatch:
            result[leftPatch] = result.get(leftPatch, 0.0) - power
        if rightComponent == str(regionUuid) and rightPatch:
            result[rightPatch] = result.get(rightPatch, 0.0) + power
    return result


def interfaceMetadataByPatch(
    femSystem: dict[str, Any], regionUuid: str
) -> dict[str, dict[str, Any]]:
    result = {}
    for interfaceUuid, interface in femSystem.get('tccInterfaces', {}).items():
        for sideName, oppositeSideName in (
            ('contactBody1', 'contactBody2'),
            ('contactBody2', 'contactBody1'),
        ):
            side = interface.get(sideName, {})
            if str(side.get('femUuid', '')) != regionUuid:
                continue
            opposite = interface.get(oppositeSideName, {})
            result[str(side.get('patchUuid', ''))] = {
                'interfaceUuid': str(interfaceUuid),
                'interfaceName': str(interface.get('name', interfaceUuid)),
                'oppositeRegionUuid': str(opposite.get('femUuid', '')),
                'oppositePatchUuid': str(opposite.get('patchUuid', '')),
                'contactConductanceWPerM2K': float(
                    interface.get('tccSetting', {}).get('conductance', 0.0)
                ),
                'searchRadiusM': float(interface.get('searchRadius', 0.0)),
            }
    return result


def resultFile(
    caseDirectory: Path,
    resultTime: float | None = None,
    resultFile: str | Path | None = None,
) -> Path:
    if resultTime is not None and resultFile is not None:
        raise ValueError('resultTime and resultFile cannot both be provided')
    if resultFile is not None:
        path = Path(resultFile)
        if not path.is_absolute():
            path = caseDirectory / path
        if not path.is_file():
            raise FileNotFoundError(f'result HDF5 file does not exist: {path}')
        return path
    if resultTime is not None:
        resultName = f'{float(resultTime):.6f}.h5'
        matches = tuple((caseDirectory / 'results').rglob(resultName))
        if len(matches) != 1:
            raise ValueError(
                f'expected one result HDF5 file named {resultName!r} below '
                f'{caseDirectory / "results"}, found {len(matches)}'
            )
        return matches[0]
    files = tuple((caseDirectory / 'results').rglob('*.h5'))
    if len(files) != 1:
        raise ValueError(
            f'expected one result HDF5 file below {caseDirectory / "results"}, '
            f'found {len(files)}'
        )
    return files[0]


def fixedHeatSourceValue(group: dict[str, Any]) -> tuple[str, float] | None:
    heatSource = group.get('volumetricGroups', {}).get('heatSource')
    if heatSource is None:
        return None
    table = heatSource.get('heatSource1D', {})
    if table.get('type') != 'fixedValue':
        raise NotImplementedError('only fixed-value steady heat sources are currently supported')
    value = float(table.get('fixedValue', 0.0))
    if not np.isfinite(value):
        raise ValueError('heat-source fixedValue must be finite')
    return str(heatSource.get('inputVariable', 'power')), value


def elementHeatSources(
    meshFile: Any,
    region: dict[str, Any],
    tetrahedronVolumes: np.ndarray,
) -> np.ndarray:
    sources = np.zeros_like(tetrahedronVolumes)
    for volumePatch in region.get('volumePatches', {}).values():
        for childUuid, group in volumePatch.get('children', {}).items():
            source = fixedHeatSourceValue(group)
            if source is None:
                continue
            datasetPath = f'volumes/{childUuid}'
            if datasetPath not in meshFile:
                raise KeyError(f'volume group is missing from mesh: {datasetPath}')
            elements = np.asarray(meshFile[datasetPath], dtype=np.int64).reshape(-1)
            if elements.size == 0 or np.any(elements < 0) or np.any(elements >= sources.size):
                raise ValueError(f'volume group {childUuid!r} contains invalid element indices')
            inputVariable, value = source
            if inputVariable == 'power':
                groupVolume = tetrahedronVolumes[elements].sum()
                sources[elements] += value / groupVolume
            elif inputVariable in ('volumetricHeatSource', 'heatSource'):
                sources[elements] += value
            else:
                raise NotImplementedError(
                    f'unsupported heat-source input variable: {inputVariable!r}'
                )
    return sources

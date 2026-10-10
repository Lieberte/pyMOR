from __future__ import annotations

from typing import Any

import numpy as np

from .motorFemGeometry import boundaryNodeGeometry


RESULT_HEAT_FLUX_BOUNDARY_TYPE = 0


CONVECTION_BOUNDARY_TYPE = 1


FIXED_TEMPERATURE_BOUNDARY_TYPE = 2


PRESCRIBED_HEAT_FLUX_BOUNDARY_TYPE = 3


def fixedTableValue(table: dict[str, Any], name: str) -> float:
    if table.get('type') != 'fixedValue':
        raise NotImplementedError(f'only fixed-value {name} is currently supported')
    value = float(table.get('fixedValue', 0.0))
    if not np.isfinite(value):
        raise ValueError(f'{name} fixedValue must be finite')
    return value


def patchBoundaryInput(patch: dict[str, Any]) -> dict[str, Any]:
    conditions = patch.get('boundaryConditions') or {}
    convective = conditions.get('convective')
    if convective is not None:
        htc = fixedTableValue(convective.get('htc1D', {}), 'convective HTC')
        referenceTemperature = fixedTableValue(
            convective.get('refT1D', {}), 'convective reference temperature'
        )
        scaleFactor = float(convective.get('scaleFactor', 1.0))
        if not np.isfinite(scaleFactor) or scaleFactor < 0.0:
            raise ValueError('convective scaleFactor must be non-negative and finite')
        return {
            'boundaryInputType': 'convection',
            'boundaryTypeId': CONVECTION_BOUNDARY_TYPE,
            'htcWPerM2K': htc * scaleFactor,
            'ambientTemperatureK': referenceTemperature,
            'absoluteTemperatureAnchor': htc * scaleFactor > 0.0,
        }
    constantTemperature = conditions.get('constT')
    if constantTemperature is not None:
        table = next(
            (
                value
                for key, value in constantTemperature.items()
                if key.lower().endswith('1d') and isinstance(value, dict)
            ),
            constantTemperature,
        )
        temperature = fixedTableValue(table, 'fixed boundary temperature')
        return {
            'boundaryInputType': 'fixedTemperature',
            'boundaryTypeId': FIXED_TEMPERATURE_BOUNDARY_TYPE,
            'htcWPerM2K': 0.0,
            'ambientTemperatureK': temperature,
            'absoluteTemperatureAnchor': True,
        }
    heatFlux = conditions.get('heatFlux')
    if heatFlux is not None:
        inputVariable = str(heatFlux.get('inputVariable', 'heatFlux'))
        if inputVariable not in ('heatFlux', 'power'):
            raise NotImplementedError(
                f'unsupported prescribed heat-flux input variable: {inputVariable!r}'
            )
        value = fixedTableValue(
            heatFlux.get('heatFlux1D', {}), 'prescribed heat flux'
        )
        return {
            'boundaryInputType': 'prescribedHeatFlux',
            'boundaryTypeId': PRESCRIBED_HEAT_FLUX_BOUNDARY_TYPE,
            'htcWPerM2K': 0.0,
            'ambientTemperatureK': 0.0,
            'absoluteTemperatureAnchor': False,
            'prescribedHeatFluxInputVariable': inputVariable,
            'prescribedHeatFluxValue': value,
        }
    return {
        'boundaryInputType': 'resultHeatFlux',
        'boundaryTypeId': RESULT_HEAT_FLUX_BOUNDARY_TYPE,
        'htcWPerM2K': 0.0,
        'ambientTemperatureK': 0.0,
        'absoluteTemperatureAnchor': False,
    }


def networkBoundaryInputs(
    femSystem: dict[str, Any], regionUuid: str
) -> dict[str, dict[str, Any]]:
    """Return explicit heat-flux inputs declared on FEM-network interfaces."""
    overrides: dict[str, dict[str, Any]] = {}
    for interfaceUuid, interface in femSystem.get('femNetworkInterfaces', {}).items():
        if not interface.get('isActive', True):
            continue
        fem = interface.get('fem', {})
        if str(fem.get('femUuid', '')) != str(regionUuid):
            continue
        if not bool(fem.get('isHeatFluxInput', False)):
            continue
        heatFlux = fem.get('heatFlux') or {}
        inputVariable = str(heatFlux.get('inputVariable', 'heatFlux'))
        if inputVariable not in ('heatFlux', 'power'):
            raise NotImplementedError(
                f'unsupported FEM-network heat-flux input variable: {inputVariable!r}'
            )
        value = fixedTableValue(
            heatFlux.get('heatFlux1D', {}), 'FEM-network prescribed heat flux'
        )
        patchUuid = str(fem.get('patchUuid', '')).strip()
        if not patchUuid:
            raise ValueError(
                f'FEM-network interface {interfaceUuid!r} has no FEM patch UUID'
            )
        if patchUuid in overrides:
            raise ValueError(
                f'multiple FEM-network heat-flux inputs target patch {patchUuid!r}'
            )
        overrides[patchUuid] = {
            'boundaryInputType': 'prescribedHeatFlux',
            'boundaryTypeId': PRESCRIBED_HEAT_FLUX_BOUNDARY_TYPE,
            'htcWPerM2K': 0.0,
            'ambientTemperatureK': 0.0,
            'absoluteTemperatureAnchor': False,
            'prescribedHeatFluxInputVariable': inputVariable,
            'prescribedHeatFluxValue': value,
            'networkInterfaceUuid': str(interfaceUuid),
            'networkBoundaryEnabled': True,
        }
    return overrides


def surfacePatchSamples(
    meshFile: Any,
    region: dict[str, Any],
    nodes: np.ndarray,
    surfaceConnectivity: np.ndarray,
    componentCenter: np.ndarray,
    boundaryInputOverrides: dict[str, dict[str, Any]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    sourceNodes = []
    normals = []
    weights = []
    patchIds = []
    patchMetadata = []
    triangleAssignments = np.zeros(surfaceConnectivity.shape[0], dtype=np.int64)
    overrides = boundaryInputOverrides or {}
    appliedOverrides: set[str] = set()

    def appendPatch(
        triangleIndices: np.ndarray,
        *,
        patchUuid: str,
        patchName: str,
        parentPatchUuid: str,
        parentPatchName: str,
        patch: dict[str, Any],
    ) -> None:
        patchId = len(patchMetadata)
        patchTriangles = surfaceConnectivity[triangleIndices]
        patchNodes, patchNormals, patchWeights = boundaryNodeGeometry(
            nodes, patchTriangles, componentCenter
        )
        sourceNodes.append(patchNodes)
        normals.append(patchNormals)
        weights.append(patchWeights)
        patchIds.append(np.full(patchNodes.size, patchId, dtype=np.int64))
        boundaryInput = patchBoundaryInput(patch)
        if patchUuid in overrides:
            boundaryInput = {**boundaryInput, **overrides[patchUuid]}
            appliedOverrides.add(patchUuid)
        area = float(patchWeights.sum())
        if boundaryInput.get('boundaryInputType') == 'prescribedHeatFlux':
            value = float(boundaryInput['prescribedHeatFluxValue'])
            inputVariable = boundaryInput['prescribedHeatFluxInputVariable']
            prescribedFlux = value / area if inputVariable == 'power' else value
            boundaryInput.update(
                {
                    'prescribedHeatFluxWPerM2': prescribedFlux,
                    'prescribedHeatFlowRateW': prescribedFlux * area,
                    'prescribedSurfaceHeatSourcePowerW': prescribedFlux * area,
                }
            )
        patchMetadata.append(
            {
                'patchId': patchId,
                'patchUuid': patchUuid,
                'patchName': patchName,
                'parentPatchUuid': parentPatchUuid,
                'parentPatchName': parentPatchName,
                'normalBoundaryEnabled': bool(patch.get('normalBoundaryEnabled', False)),
                'networkBoundaryEnabled': bool(patch.get('networkBoundaryEnabled', False)),
                'tccEnabled': bool(patch.get('tccEnabled', False)),
                'surfaceTriangleCount': int(triangleIndices.size),
                'boundarySampleCount': int(patchNodes.size),
                'surfaceAreaM2': area,
                **boundaryInput,
            }
        )

    for parentPatchUuid, parentPatch in region.get('patches', {}).items():
        for childUuid, patch in parentPatch.get('children', {}).items():
            datasetPath = f'patches/{parentPatchUuid}/{childUuid}'
            if datasetPath not in meshFile:
                raise KeyError(f'surface patch is missing from mesh: {datasetPath}')
            triangleIndices = np.asarray(meshFile[datasetPath], dtype=np.int64).reshape(-1)
            if (
                triangleIndices.size == 0
                or np.any(triangleIndices < 0)
                or np.any(triangleIndices >= surfaceConnectivity.shape[0])
            ):
                raise ValueError(f'surface patch {childUuid!r} contains invalid triangle indices')
            triangleAssignments[triangleIndices] += 1
            appendPatch(
                triangleIndices,
                patchUuid=str(patch.get('uuid', childUuid)),
                patchName=str(patch.get('name', childUuid)),
                parentPatchUuid=str(parentPatchUuid),
                parentPatchName=str(parentPatch.get('name', parentPatchUuid)),
                patch=patch,
            )

    if not patchMetadata:
        appendPatch(
            np.arange(surfaceConnectivity.shape[0]),
            patchUuid='all-surfaces',
            patchName='all-surfaces',
            parentPatchUuid='all-surfaces',
            parentPatchName='all-surfaces',
            patch={},
        )
        triangleAssignments[:] = 1
    elif np.any(triangleAssignments == 0):
        appendPatch(
            np.flatnonzero(triangleAssignments == 0),
            patchUuid='unassigned-surfaces',
            patchName='unassigned-surfaces',
            parentPatchUuid='unassigned-surfaces',
            parentPatchName='unassigned-surfaces',
            patch={},
        )
        triangleAssignments[triangleAssignments == 0] = 1

    if np.any(triangleAssignments != 1):
        raise ValueError('surface triangles must belong to exactly one surface patch')
    missingOverrides = set(overrides) - appliedOverrides
    if missingOverrides:
        raise KeyError(
            f'FEM-network heat-flux inputs reference unknown patches: {sorted(missingOverrides)}'
        )
    return (
        np.concatenate(sourceNodes),
        np.vstack(normals),
        np.concatenate(weights),
        np.concatenate(patchIds),
        patchMetadata,
    )

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import replace
from typing import Sequence

import numpy as np

from .thermalComponentCase import thermalComponentCase


THERMAL_ASSEMBLY_CONTEXT_V1 = 'thermalAssemblyContextV1'


def _featureToken(value: str) -> str:
    return re.sub(r'[^A-Za-z0-9]+', '_', str(value).strip()).strip('_').lower()


def thermalAssemblyContextFeatureNames(
    componentTypes: Sequence[str],
) -> tuple[str, ...]:
    names = [
        'assemblyComponentCountRatio',
        'assemblySignedLogHeatSourcePower',
        'assemblySignedLogBoundaryHeatFlow',
        'assemblySignedLogNetPower',
        'assemblyLogVolume',
        'assemblyLogBoundaryArea',
        'assemblyAnchorFraction',
        'assemblyReferenceTemperatureRatio',
        'localInterfaceAreaFraction',
        'localInterfaceCountRatio',
        'localInterfaceConductanceLog',
    ]
    for componentType in componentTypes:
        token = _featureToken(componentType)
        names.extend(
            (
                f'componentTypeIs_{token}',
                f'assembly_{token}_countRatio',
                f'assembly_{token}_signedLogHeatSourcePower',
                f'assembly_{token}_signedLogBoundaryHeatFlow',
                f'assembly_{token}_anchorFraction',
            )
        )
    return tuple(names)


def thermalAssemblyBoundaryFeatureNames(
    componentTypes: Sequence[str],
) -> tuple[str, ...]:
    return (
        'isThermalInterface',
        'isNetworkInterface',
        'interfaceConductanceLog',
        'oppositeSignedLogHeatSourcePower',
        'oppositeSignedLogBoundaryHeatFlow',
        'oppositeLogVolume',
        'oppositeHasLocalTemperatureAnchor',
        'oppositeReferenceTemperatureRatio',
        *(f'oppositeComponentIs_{_featureToken(value)}' for value in componentTypes),
    )


def _signedLogScale(value: float) -> float:
    return float(np.sign(value) * np.log1p(abs(value)) / 10.0)


def _positiveLogScale(value: float, multiplier: float = 1.0) -> float:
    return float(np.log1p(max(0.0, value) * multiplier) / 10.0)


def _groupKey(case: thermalComponentCase) -> tuple[str, str, str]:
    metadata = case.metadata
    return (
        str(metadata.get('sourceDataset', '')),
        str(metadata.get('assemblyId') or metadata.get('motorId') or ''),
        str(metadata.get('parentCaseId') or case.caseId),
    )


def _anchorReference(cases: Sequence[thermalComponentCase]) -> float | None:
    fixedValues = []
    fixedWeights = []
    convectionValues = []
    convectionWeights = []
    for case in cases:
        fixed = case.boundary.boundaryTypeIds == 2
        convection = case.boundary.htcWPerM2K > 0.0
        if np.any(fixed):
            fixedValues.append(case.boundary.ambientTemperatureK[fixed])
            fixedWeights.append(case.boundary.weightsM2[fixed])
        if np.any(convection):
            convectionValues.append(case.boundary.ambientTemperatureK[convection])
            convectionWeights.append(
                case.boundary.weightsM2[convection]
                * case.boundary.htcWPerM2K[convection]
            )
    if fixedValues:
        return float(
            np.average(np.concatenate(fixedValues), weights=np.concatenate(fixedWeights))
        )
    if convectionValues:
        return float(
            np.average(
                np.concatenate(convectionValues),
                weights=np.concatenate(convectionWeights),
            )
        )
    return None


def _interfaceFeatures(
    case: thermalComponentCase,
    componentTypes: Sequence[str],
    caseByRegion: dict[str, thermalComponentCase],
) -> tuple[np.ndarray, dict[str, float]]:
    featureCount = 8 + len(componentTypes)
    features = np.zeros((case.boundary.weightsM2.size, featureCount), dtype=np.float64)
    interfaceArea = 0.0
    interfacePatches = 0
    conductance = 0.0
    componentIndex = {value: index for index, value in enumerate(componentTypes)}
    for rawPatch in case.metadata.get('surfacePatches', ()):
        patch = dict(rawPatch)
        if patch.get('patchId') is None:
            continue
        mask = case.boundary.patchIds == int(patch['patchId'])
        if not np.any(mask):
            continue
        isInterface = bool(patch.get('interfaceUuid'))
        isNetwork = bool(patch.get('networkInterfaceUuid'))
        if not isInterface and not isNetwork:
            continue
        rawConductance = max(
            0.0,
            float(patch.get('contactConductanceWPerM2K', 0.0)),
        )
        features[mask, 0] = float(isInterface)
        features[mask, 1] = float(isNetwork)
        features[mask, 2] = _positiveLogScale(rawConductance)
        oppositeCase = caseByRegion.get(str(patch.get('oppositeRegionUuid', '')))
        oppositeType = oppositeCase.componentType if oppositeCase is not None else None
        if oppositeCase is not None:
            oppositeReference = _anchorReference((oppositeCase,))
            features[mask, 3] = _signedLogScale(
                oppositeCase.volume.totalHeatSourcePowerW
            )
            features[mask, 4] = _signedLogScale(
                oppositeCase.boundary.totalHeatFlowRateW
            )
            features[mask, 5] = _positiveLogScale(
                float(np.sum(oppositeCase.volume.weightsM3)),
                1e6,
            )
            features[mask, 6] = float(
                oppositeCase.operatingFeatures.get(
                    'absoluteTemperatureAnchor', 0.0
                )
                > 0.5
            )
            features[mask, 7] = (
                float(oppositeReference) / 500.0
                if oppositeReference is not None
                else 0.0
            )
        if oppositeType in componentIndex:
            features[mask, 8 + componentIndex[oppositeType]] = 1.0
        area = float(np.sum(case.boundary.weightsM2[mask]))
        interfaceArea += area
        interfacePatches += 1
        conductance += rawConductance
    totalArea = max(float(np.sum(case.boundary.weightsM2)), np.finfo(float).eps)
    return features, {
        'localInterfaceAreaFraction': interfaceArea / totalArea,
        'localInterfaceCountRatio': interfacePatches / 16.0,
        'localInterfaceConductanceLog': _positiveLogScale(conductance),
    }


def enrichThermalAssemblyCases(
    cases: Sequence[thermalComponentCase],
    *,
    componentTypes: Sequence[str] | None = None,
) -> tuple[thermalComponentCase, ...]:
    """Add target-free assembly and interface context to component cases.

    The group context uses only geometry, material, heat-source and boundary
    inputs from the same parent case. Temperature targets define no feature;
    the reference temperature comes from prescribed convection or fixed-
    temperature boundaries and is therefore available at inference.
    """
    resolvedCases = tuple(cases)
    resolvedTypes = tuple(
        componentTypes
        if componentTypes is not None
        else sorted({case.componentType for case in resolvedCases})
    )
    if len(set(resolvedTypes)) != len(resolvedTypes):
        raise ValueError('componentTypes must be unique')
    groups: dict[tuple[str, str, str], list[thermalComponentCase]] = defaultdict(list)
    for case in resolvedCases:
        groups[_groupKey(case)].append(case)

    enrichedByCaseId = {}
    for groupCases in groups.values():
        groupReference = _anchorReference(groupCases)
        groupAnchored = groupReference is not None
        totalSource = sum(case.volume.totalHeatSourcePowerW for case in groupCases)
        totalBoundary = sum(case.boundary.totalHeatFlowRateW for case in groupCases)
        totalVolume = sum(float(np.sum(case.volume.weightsM3)) for case in groupCases)
        totalArea = sum(float(np.sum(case.boundary.weightsM2)) for case in groupCases)
        locallyAnchored = sum(
            bool(case.operatingFeatures.get('absoluteTemperatureAnchor', 0.0))
            for case in groupCases
        )
        byType = defaultdict(list)
        caseByRegion = {}
        for case in groupCases:
            byType[case.componentType].append(case)
            regionUuid = case.metadata.get('regionUuid')
            if regionUuid:
                caseByRegion[str(regionUuid)] = case

        shared = {
            'assemblyComponentCountRatio': len(groupCases) / 16.0,
            'assemblySignedLogHeatSourcePower': _signedLogScale(totalSource),
            'assemblySignedLogBoundaryHeatFlow': _signedLogScale(totalBoundary),
            'assemblySignedLogNetPower': _signedLogScale(totalSource + totalBoundary),
            'assemblyLogVolume': _positiveLogScale(totalVolume, 1e6),
            'assemblyLogBoundaryArea': _positiveLogScale(totalArea, 1e3),
            'assemblyAnchorFraction': locallyAnchored / max(1, len(groupCases)),
            'assemblyReferenceTemperatureRatio': (
                float(groupReference) / 500.0 if groupReference is not None else 0.0
            ),
        }
        for componentType in resolvedTypes:
            token = _featureToken(componentType)
            typed = byType.get(componentType, ())
            typedSource = sum(case.volume.totalHeatSourcePowerW for case in typed)
            typedBoundary = sum(case.boundary.totalHeatFlowRateW for case in typed)
            typedAnchored = sum(
                bool(case.operatingFeatures.get('absoluteTemperatureAnchor', 0.0))
                for case in typed
            )
            shared.update(
                {
                    f'assembly_{token}_countRatio': len(typed) / 4.0,
                    f'assembly_{token}_signedLogHeatSourcePower': _signedLogScale(
                        typedSource
                    ),
                    f'assembly_{token}_signedLogBoundaryHeatFlow': _signedLogScale(
                        typedBoundary
                    ),
                    f'assembly_{token}_anchorFraction': typedAnchored
                    / max(1, len(typed)),
                }
            )

        for case in groupCases:
            boundaryContext, local = _interfaceFeatures(
                case,
                resolvedTypes,
                caseByRegion,
            )
            componentFlags = {
                f'componentTypeIs_{_featureToken(value)}': float(
                    case.componentType == value
                )
                for value in resolvedTypes
            }
            operating = {
                **case.operatingFeatures,
                **shared,
                **local,
                **componentFlags,
                'absoluteTemperatureAnchor': float(groupAnchored),
            }
            if groupReference is not None:
                operating['absoluteTemperatureReferenceK'] = float(groupReference)
            existingExtra = case.boundary.extraFeatures
            boundary = replace(
                case.boundary,
                extraFeatures=np.column_stack((existingExtra, boundaryContext)),
            )
            metadata = {
                **case.metadata,
                'thermalAssemblyContext': {
                    'version': THERMAL_ASSEMBLY_CONTEXT_V1,
                    'componentTypes': list(resolvedTypes),
                    'boundaryFeatureNames': list(
                        thermalAssemblyBoundaryFeatureNames(resolvedTypes)
                    ),
                    'groupHasAbsoluteTemperatureAnchor': groupAnchored,
                    'groupReferenceTemperatureK': groupReference,
                },
            }
            enrichedByCaseId[case.caseId] = replace(
                case,
                boundary=boundary,
                operatingFeatures=operating,
                metadata=metadata,
            )
    return tuple(enrichedByCaseId[case.caseId] for case in resolvedCases)

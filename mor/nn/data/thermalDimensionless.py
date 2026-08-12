from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np

from .thermalComponentCase import thermalComponentCase, thermalComponentFeatureSchema
from .thermalOperatorCase import thermalOperatorCase


STEADY_THERMAL_DIMENSIONLESS_V1 = 'steadyThermalDimensionlessV1'
STEADY_THERMAL_DIMENSIONLESS_V2 = 'steadyThermalDimensionlessV2'


def _signedLog1p(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.sign(values) * np.log1p(np.abs(values))


def hasAbsoluteTemperatureAnchor(case: thermalComponentCase) -> bool:
    return bool(case.operatingFeatures.get('absoluteTemperatureAnchor', 0.0))


def _fixedTemperatureMask(
    case: thermalComponentCase,
    schema: thermalComponentFeatureSchema | None,
) -> np.ndarray:
    if schema is not None and schema.fixedTemperatureBoundaryTypeIds:
        return np.isin(
            case.boundary.boundaryTypeIds,
            schema.fixedTemperatureBoundaryTypeIds,
        )
    if schema is not None and schema.temperatureBoundaryTypeIds:
        return np.isin(
            case.boundary.boundaryTypeIds,
            schema.temperatureBoundaryTypeIds,
        ) & (case.boundary.htcWPerM2K <= 0.0)
    # Preserve the established motor schema convention when no schema is given.
    return case.boundary.boundaryTypeIds == 2


def anchorReferenceTemperatureK(
    case: thermalComponentCase,
    *,
    schema: thermalComponentFeatureSchema | None = None,
) -> float:
    if not hasAbsoluteTemperatureAnchor(case):
        raise ValueError(f'case has no absolute temperature anchor: {case.caseId}')
    convection = case.boundary.htcWPerM2K > 0.0
    fixedTemperature = _fixedTemperatureMask(case, schema)
    if np.any(fixedTemperature):
        return float(
            np.average(
                case.boundary.ambientTemperatureK[fixedTemperature],
                weights=case.boundary.weightsM2[fixedTemperature],
            )
        )
    if np.any(convection):
        weights = (
            case.boundary.weightsM2[convection]
            * case.boundary.htcWPerM2K[convection]
        )
        return float(
            np.average(
                case.boundary.ambientTemperatureK[convection],
                weights=weights,
            )
        )
    if schema is not None and schema.temperatureBoundaryTypeIds:
        active = np.isin(
            case.boundary.boundaryTypeIds,
            schema.temperatureBoundaryTypeIds,
        )
    else:
        active = fixedTemperature
    if not np.any(active):
        raise ValueError(f'anchored case has no active temperature boundary: {case.caseId}')
    return float(
        np.average(
            case.boundary.ambientTemperatureK[active],
            weights=case.boundary.weightsM2[active],
        )
    )


@dataclass(frozen=True)
class steadyThermalDimensionlessScales:
    lengthMeters: float
    conductivityWPerMK: float
    densityKgPerM3: float
    specificHeatJPerKgK: float
    temperatureReferenceK: float
    temperatureScaleK: float
    boundaryFluxTemperatureScaleK: float
    netConductionTemperatureScaleK: float
    convectionTemperatureScaleK: float
    absoluteTemperatureAnchorAvailable: bool


@dataclass(frozen=True)
class steadyThermalDimensionlessCase:
    operatorCase: thermalOperatorCase
    scales: steadyThermalDimensionlessScales


def steadyThermalDimensionlessScaleForCase(
    case: thermalComponentCase,
    *,
    schema: thermalComponentFeatureSchema | None = None,
    minimumTemperatureScaleK: float = 1.0,
) -> steadyThermalDimensionlessScales:
    minimumTemperatureScaleK = float(minimumTemperatureScaleK)
    if not np.isfinite(minimumTemperatureScaleK) or minimumTemperatureScaleK <= 0.0:
        raise ValueError('minimumTemperatureScaleK must be positive and finite')
    similarityMetadata = case.metadata.get('similarityAugmentation', {})
    similarityTemperatureRatio = float(
        similarityMetadata.get('temperatureDifferenceRatio', 1.0)
    )
    if (
        not np.isfinite(similarityTemperatureRatio)
        or similarityTemperatureRatio <= 0.0
    ):
        raise ValueError('similarity augmentation temperature ratio must be positive')
    effectiveMinimumTemperatureScaleK = (
        minimumTemperatureScaleK * similarityTemperatureRatio
    )

    volumeWeights = case.volume.weightsM3
    boundaryWeights = case.boundary.weightsM2
    volumeM3 = float(np.sum(volumeWeights))
    lengthMeters = volumeM3 ** (1.0 / 3.0)
    isotropicConductivity = np.trace(
        case.volume.conductivityWPerMK, axis1=1, axis2=2
    ) / 3.0
    conductivityWPerMK = float(
        np.average(isotropicConductivity, weights=volumeWeights)
    )
    densityKgPerM3 = float(
        np.average(case.volume.densityKgPerM3, weights=volumeWeights)
    )
    specificHeatJPerKgK = float(
        np.average(case.volume.specificHeatJPerKgK, weights=volumeWeights)
    )

    boundaryFluxRmsWPerM2 = float(
        np.sqrt(
            np.average(
                np.square(case.boundary.heatFluxWPerM2),
                weights=boundaryWeights,
            )
        )
    )
    boundaryFluxTemperatureScaleK = (
        boundaryFluxRmsWPerM2 * lengthMeters / conductivityWPerMK
    )
    netPowerW = (
        case.volume.totalHeatSourcePowerW + case.boundary.totalHeatFlowRateW
    )
    netConductionTemperatureScaleK = abs(netPowerW) / (
        conductivityWPerMK * lengthMeters
    )
    htcAreaWPerK = float(
        np.dot(case.boundary.htcWPerM2K, boundaryWeights)
    )
    convectionTemperatureScaleK = (
        abs(netPowerW) / htcAreaWPerK
        if htcAreaWPerK > np.finfo(np.float64).eps
        else 0.0
    )
    temperatureScaleK = max(
        boundaryFluxTemperatureScaleK,
        netConductionTemperatureScaleK,
        convectionTemperatureScaleK,
        effectiveMinimumTemperatureScaleK,
    )
    anchored = hasAbsoluteTemperatureAnchor(case)
    temperatureReferenceK = (
        anchorReferenceTemperatureK(case, schema=schema) if anchored else 0.0
    )
    return steadyThermalDimensionlessScales(
        lengthMeters=lengthMeters,
        conductivityWPerMK=conductivityWPerMK,
        densityKgPerM3=densityKgPerM3,
        specificHeatJPerKgK=specificHeatJPerKgK,
        temperatureReferenceK=temperatureReferenceK,
        temperatureScaleK=temperatureScaleK,
        boundaryFluxTemperatureScaleK=boundaryFluxTemperatureScaleK,
        netConductionTemperatureScaleK=netConductionTemperatureScaleK,
        convectionTemperatureScaleK=convectionTemperatureScaleK,
        absoluteTemperatureAnchorAvailable=anchored,
    )


def _weightedSpatialMoments(
    coordinates: np.ndarray,
    magnitude: np.ndarray,
    measureWeights: np.ndarray,
) -> np.ndarray:
    effectiveWeights = np.abs(magnitude) * measureWeights
    total = float(np.sum(effectiveWeights))
    if total <= np.finfo(np.float64).eps:
        return np.zeros(6, dtype=np.float64)
    center = np.sum(coordinates * effectiveWeights[:, None], axis=0) / total
    variance = np.sum(
        np.square(coordinates - center) * effectiveWeights[:, None], axis=0
    ) / total
    return np.concatenate((center, np.sqrt(np.maximum(variance, 0.0))))


def _weightedTemperatureStatistics(
    temperatures: np.ndarray,
    weights: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    if not np.any(active):
        return np.zeros(3, dtype=np.float64)
    values = temperatures[active]
    activeWeights = weights[active]
    return np.array(
        [
            float(np.min(values)),
            float(np.average(values, weights=activeWeights)),
            float(np.max(values)),
        ],
        dtype=np.float64,
    )


def steadyThermalDimensionlessV2GlobalFeatureNames(
    schema: thermalComponentFeatureSchema,
) -> tuple[str, ...]:
    names = [
        'halfExtentX',
        'halfExtentY',
        'halfExtentZ',
        'dimensionlessVolume',
        'dimensionlessBoundaryArea',
        'signedLogVolumeSourcePower',
        'signedLogBoundaryFluxPower',
        'signedLogNetPower',
        'logConvectionConductance',
        'boundaryFluxScaleRatio',
        'netConductionScaleRatio',
        'convectionScaleRatio',
        'heatSourceCenterX',
        'heatSourceCenterY',
        'heatSourceCenterZ',
        'heatSourceSpreadX',
        'heatSourceSpreadY',
        'heatSourceSpreadZ',
        'boundaryFluxCenterX',
        'boundaryFluxCenterY',
        'boundaryFluxCenterZ',
        'boundaryFluxSpreadX',
        'boundaryFluxSpreadY',
        'boundaryFluxSpreadZ',
    ]
    for boundaryType in range(schema.boundaryTypeCount):
        names.extend(
            (
                f'boundaryType{boundaryType}AreaFraction',
                f'boundaryType{boundaryType}CenterX',
                f'boundaryType{boundaryType}CenterY',
                f'boundaryType{boundaryType}CenterZ',
            )
        )
    names.extend(
        (
            'fixedTemperatureAreaFraction',
            'convectionAreaFraction',
            'fixedTemperatureMinimum',
            'fixedTemperatureMean',
            'fixedTemperatureMaximum',
            'convectionTemperatureMinimum',
            'convectionTemperatureMean',
            'convectionTemperatureMaximum',
        )
    )
    names.extend(schema.operatingFeatureNames)
    return tuple(names)


def _steadyThermalDimensionlessV2GlobalFeatures(
    case: thermalComponentCase,
    *,
    schema: thermalComponentFeatureSchema,
    scales: steadyThermalDimensionlessScales,
    volumeCoordinates: np.ndarray,
    boundaryCoordinates: np.ndarray,
    boundaryTemperature: np.ndarray,
    operating: np.ndarray,
) -> np.ndarray:
    length = scales.lengthMeters
    conductivity = scales.conductivityWPerMK
    temperatureScale = scales.temperatureScaleK
    powerScale = conductivity * length * temperatureScale
    volumePower = case.volume.totalHeatSourcePowerW / powerScale
    boundaryFluxPower = case.boundary.totalHeatFlowRateW / powerScale
    netPower = volumePower + boundaryFluxPower
    convectionConductance = float(
        np.dot(case.boundary.htcWPerM2K, case.boundary.weightsM2)
        / (conductivity * length)
    )
    boundaryArea = float(np.sum(case.boundary.weightsM2))
    boundaryAreaScale = max(boundaryArea, np.finfo(np.float64).eps)
    fixedMask = _fixedTemperatureMask(case, schema)
    convectionMask = case.boundary.htcWPerM2K > 0.0
    featureValues = [
        *(case.coordinateFrame.halfExtentMeters / length),
        float(np.sum(case.volume.weightsM3) / length**3),
        float(boundaryArea / length**2),
        float(_signedLog1p(np.array(volumePower))),
        float(_signedLog1p(np.array(boundaryFluxPower))),
        float(_signedLog1p(np.array(netPower))),
        float(np.log1p(convectionConductance)),
        scales.boundaryFluxTemperatureScaleK / temperatureScale,
        scales.netConductionTemperatureScaleK / temperatureScale,
        scales.convectionTemperatureScaleK / temperatureScale,
        *_weightedSpatialMoments(
            volumeCoordinates,
            case.volume.heatSourceWPerM3,
            case.volume.weightsM3,
        ),
        *_weightedSpatialMoments(
            boundaryCoordinates,
            case.boundary.heatFluxWPerM2,
            case.boundary.weightsM2,
        ),
    ]
    for boundaryType in range(schema.boundaryTypeCount):
        active = case.boundary.boundaryTypeIds == boundaryType
        area = float(np.sum(case.boundary.weightsM2[active]))
        if area > np.finfo(np.float64).eps:
            center = np.average(
                boundaryCoordinates[active],
                axis=0,
                weights=case.boundary.weightsM2[active],
            )
        else:
            center = np.zeros(3, dtype=np.float64)
        featureValues.extend((area / boundaryAreaScale, *center))
    featureValues.extend(
        (
            float(np.sum(case.boundary.weightsM2[fixedMask]) / boundaryAreaScale),
            float(
                np.sum(case.boundary.weightsM2[convectionMask])
                / boundaryAreaScale
            ),
            *_weightedTemperatureStatistics(
                boundaryTemperature,
                case.boundary.weightsM2,
                fixedMask,
            ),
            *_weightedTemperatureStatistics(
                boundaryTemperature,
                case.boundary.weightsM2 * case.boundary.htcWPerM2K,
                convectionMask,
            ),
            *operating,
        )
    )
    return np.asarray(featureValues, dtype=np.float64)


def steadyThermalDimensionlessOperatorCase(
    case: thermalComponentCase,
    *,
    schema: thermalComponentFeatureSchema,
    operatingScales: Mapping[str, float],
    minimumTemperatureScaleK: float = 1.0,
    featureMode: str = STEADY_THERMAL_DIMENSIONLESS_V1,
) -> steadyThermalDimensionlessCase:
    if featureMode not in (
        STEADY_THERMAL_DIMENSIONLESS_V1,
        STEADY_THERMAL_DIMENSIONLESS_V2,
    ):
        raise ValueError(f'unsupported steady thermal feature mode: {featureMode!r}')
    if np.any(case.boundary.boundaryTypeIds >= schema.boundaryTypeCount):
        raise ValueError('boundaryTypeIds exceed schema boundaryTypeCount')
    scales = steadyThermalDimensionlessScaleForCase(
        case,
        schema=schema,
        minimumTemperatureScaleK=minimumTemperatureScaleK,
    )
    frame = case.coordinateFrame
    length = scales.lengthMeters
    volumeCoordinates = (case.volume.coordinatesMeters - frame.centerMeters) / length
    boundaryCoordinates = (case.boundary.coordinatesMeters - frame.centerMeters) / length
    queryCoordinates = (case.temperature.coordinatesMeters - frame.centerMeters) / length

    conductivity = (
        case.volume.conductivityWPerMK.reshape(-1, 9)
        / scales.conductivityWPerMK
    )
    density = case.volume.densityKgPerM3 / scales.densityKgPerM3
    specificHeat = (
        case.volume.specificHeatJPerKgK / scales.specificHeatJPerKgK
    )
    heatSource = (
        case.volume.heatSourceWPerM3
        * length**2
        / (scales.conductivityWPerMK * scales.temperatureScaleK)
    )
    volumeFeatures = np.column_stack(
        (
            volumeCoordinates,
            conductivity,
            density,
            specificHeat,
            _signedLog1p(heatSource),
            case.volume.extraFeatures,
        )
    )

    boundaryTypeOneHot = np.eye(schema.boundaryTypeCount, dtype=np.float64)[
        case.boundary.boundaryTypeIds
    ]
    temperatureActive = case.boundary.htcWPerM2K > 0.0
    if schema.temperatureBoundaryTypeIds:
        temperatureActive |= np.isin(
            case.boundary.boundaryTypeIds,
            schema.temperatureBoundaryTypeIds,
        )
    boundaryTemperature = np.where(
        temperatureActive,
        (
            case.boundary.ambientTemperatureK - scales.temperatureReferenceK
        )
        / scales.temperatureScaleK,
        0.0,
    )
    biot = case.boundary.htcWPerM2K * length / scales.conductivityWPerMK
    boundaryHeatFlux = (
        case.boundary.heatFluxWPerM2
        * length
        / (scales.conductivityWPerMK * scales.temperatureScaleK)
    )
    boundaryFeatures = np.column_stack(
        (
            boundaryCoordinates,
            case.boundary.normals,
            np.log1p(biot),
            boundaryTemperature,
            _signedLog1p(boundaryHeatFlux),
            boundaryTypeOneHot,
            case.boundary.extraFeatures,
        )
    )

    operating = []
    for name in schema.operatingFeatureNames:
        if name not in case.operatingFeatures:
            raise KeyError(
                f'operatingFeatures does not contain required feature {name!r}'
            )
        scale = float(operatingScales.get(name, 1.0))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f'operating scale for {name!r} must be positive')
        operating.append(case.operatingFeatures[name] / scale)
    operatingArray = np.asarray(operating, dtype=np.float64)
    if featureMode == STEADY_THERMAL_DIMENSIONLESS_V1:
        globalFeatures = np.concatenate(
            (
                frame.halfExtentMeters / length,
                np.array(
                    [
                        np.sum(case.volume.weightsM3) / length**3,
                        np.sum(case.boundary.weightsM2) / length**2,
                    ]
                ),
                operatingArray,
            )
        )
    else:
        globalFeatures = _steadyThermalDimensionlessV2GlobalFeatures(
            case,
            schema=schema,
            scales=scales,
            volumeCoordinates=volumeCoordinates,
            boundaryCoordinates=boundaryCoordinates,
            boundaryTemperature=boundaryTemperature,
            operating=operatingArray,
        )

    targetReferenceK = (
        scales.temperatureReferenceK
        if scales.absoluteTemperatureAnchorAvailable
        else float(
            np.average(
                case.temperature.temperatureK,
                weights=case.temperature.weightsM3,
            )
        )
    )
    operatorCase = thermalOperatorCase(
        caseId=case.caseId,
        geometryId=case.geometryId,
        volumeFeatures=volumeFeatures,
        volumeWeights=case.volume.weightsM3 / length**3,
        boundaryFeatures=boundaryFeatures,
        boundaryWeights=case.boundary.weightsM2 / length**2,
        globalFeatures=globalFeatures,
        queryCoordinates=queryCoordinates,
        queryTemperatures=(
            (case.temperature.temperatureK - targetReferenceK)
            / scales.temperatureScaleK
        )[:, None],
        queryWeights=case.temperature.weightsM3 / length**3,
    )
    return steadyThermalDimensionlessCase(operatorCase=operatorCase, scales=scales)


def enforceMatchingDirichletQueries(
    prediction: Any,
    operatorCase: thermalOperatorCase,
    *,
    schema: thermalComponentFeatureSchema,
    coordinateTolerance: float = 1e-6,
    boundaryTemperatureFeatureIndex: int = 7,
    boundaryTypeFeatureStart: int = 9,
) -> tuple[np.ndarray, int]:
    coordinateTolerance = float(coordinateTolerance)
    if not np.isfinite(coordinateTolerance) or coordinateTolerance <= 0.0:
        raise ValueError('coordinateTolerance must be positive and finite')
    values = np.asarray(prediction, dtype=np.float64).copy()
    originalShape = values.shape
    values = values.reshape(-1)
    if values.size != operatorCase.queryCoordinates.shape[0]:
        raise ValueError('prediction count must match operatorCase queries')
    if not schema.fixedTemperatureBoundaryTypeIds:
        return values.reshape(originalShape), 0
    oneHot = operatorCase.boundaryFeatures[
        ...,
        boundaryTypeFeatureStart : (
            boundaryTypeFeatureStart + schema.boundaryTypeCount
        ),
    ]
    fixedMask = np.zeros(operatorCase.boundaryFeatures.shape[0], dtype=bool)
    for boundaryType in schema.fixedTemperatureBoundaryTypeIds:
        fixedMask |= oneHot[:, boundaryType] > 0.5
    if not np.any(fixedMask):
        return values.reshape(originalShape), 0
    queryCoordinates = operatorCase.queryCoordinates
    queryKeys = np.rint(queryCoordinates / coordinateTolerance).astype(np.int64)
    queryMap: dict[tuple[int, ...], list[int]] = {}
    for queryIndex, key in enumerate(queryKeys):
        queryMap.setdefault(tuple(key), []).append(queryIndex)
    matched = set()
    boundaryCoordinates = operatorCase.boundaryFeatures[fixedMask, : operatorCase.queryDim]
    boundaryTemperatures = operatorCase.boundaryFeatures[
        fixedMask, boundaryTemperatureFeatureIndex
    ]
    boundaryKeys = np.rint(
        boundaryCoordinates / coordinateTolerance
    ).astype(np.int64)
    for coordinate, temperature, key in zip(
        boundaryCoordinates,
        boundaryTemperatures,
        boundaryKeys,
    ):
        for queryIndex in queryMap.get(tuple(key), ()):
            if np.linalg.norm(queryCoordinates[queryIndex] - coordinate) <= coordinateTolerance:
                values[queryIndex] = temperature
                matched.add(queryIndex)
    return values.reshape(originalShape), len(matched)


def similarityAugmentedThermalComponentCase(
    case: thermalComponentCase,
    *,
    lengthRatio: float = 1.0,
    conductivityRatio: float = 1.0,
    temperatureDifferenceRatio: float = 1.0,
    referenceTemperatureShiftK: float = 0.0,
    schema: thermalComponentFeatureSchema | None = None,
    caseId: str | None = None,
) -> thermalComponentCase:
    ratios = np.asarray(
        (lengthRatio, conductivityRatio, temperatureDifferenceRatio),
        dtype=np.float64,
    )
    if np.any(~np.isfinite(ratios)) or np.any(ratios <= 0.0):
        raise ValueError('thermal similarity ratios must be positive and finite')
    referenceTemperatureShiftK = float(referenceTemperatureShiftK)
    if not np.isfinite(referenceTemperatureShiftK):
        raise ValueError('referenceTemperatureShiftK must be finite')
    lengthRatio, conductivityRatio, temperatureDifferenceRatio = map(float, ratios)
    frame = case.coordinateFrame
    anchored = hasAbsoluteTemperatureAnchor(case)
    referenceTemperature = (
        anchorReferenceTemperatureK(case, schema=schema)
        if anchored
        else float(
            np.average(
                case.temperature.temperatureK,
                weights=case.temperature.weightsM3,
            )
        )
    )
    shiftedReference = referenceTemperature + referenceTemperatureShiftK

    def scaledCoordinates(coordinates: np.ndarray) -> np.ndarray:
        return frame.centerMeters + lengthRatio * (
            coordinates - frame.centerMeters
        )

    def scaledTemperature(values: np.ndarray) -> np.ndarray:
        return shiftedReference + temperatureDifferenceRatio * (
            values - referenceTemperature
        )

    suffix = (
        f'sim-l{lengthRatio:g}-k{conductivityRatio:g}'
        f'-t{temperatureDifferenceRatio:g}-r{referenceTemperatureShiftK:g}'
    )
    return replace(
        case,
        caseId=caseId or f'{case.caseId}-{suffix}',
        volume=replace(
            case.volume,
            coordinatesMeters=scaledCoordinates(case.volume.coordinatesMeters),
            conductivityWPerMK=(
                case.volume.conductivityWPerMK * conductivityRatio
            ),
            heatSourceWPerM3=(
                case.volume.heatSourceWPerM3
                * conductivityRatio
                * temperatureDifferenceRatio
                / lengthRatio**2
            ),
            weightsM3=case.volume.weightsM3 * lengthRatio**3,
        ),
        boundary=replace(
            case.boundary,
            coordinatesMeters=scaledCoordinates(case.boundary.coordinatesMeters),
            htcWPerM2K=(
                case.boundary.htcWPerM2K
                * conductivityRatio
                / lengthRatio
            ),
            ambientTemperatureK=scaledTemperature(
                case.boundary.ambientTemperatureK
            ),
            weightsM2=case.boundary.weightsM2 * lengthRatio**2,
            heatFluxWPerM2=(
                case.boundary.heatFluxWPerM2
                * conductivityRatio
                * temperatureDifferenceRatio
                / lengthRatio
            ),
        ),
        temperature=replace(
            case.temperature,
            coordinatesMeters=scaledCoordinates(
                case.temperature.coordinatesMeters
            ),
            temperatureK=scaledTemperature(case.temperature.temperatureK),
            weightsM3=case.temperature.weightsM3 * lengthRatio**3,
        ),
        metadata={
            **case.metadata,
            'similarityAugmentation': {
                'sourceCaseId': case.caseId,
                'lengthRatio': lengthRatio,
                'conductivityRatio': conductivityRatio,
                'temperatureDifferenceRatio': temperatureDifferenceRatio,
                'referenceTemperatureShiftK': referenceTemperatureShiftK,
            },
        },
    )


def randomSimilarityAugmentations(
    cases: Sequence[thermalComponentCase],
    *,
    augmentationsPerCase: int,
    lengthRatioRange: tuple[float, float] = (0.5, 2.0),
    conductivityRatioRange: tuple[float, float] = (0.5, 2.0),
    temperatureDifferenceRatioRange: tuple[float, float] = (0.5, 2.0),
    schema: thermalComponentFeatureSchema | None = None,
    randomSeed: int = 0,
) -> tuple[thermalComponentCase, ...]:
    augmentationsPerCase = int(augmentationsPerCase)
    if augmentationsPerCase < 0:
        raise ValueError('augmentationsPerCase must be non-negative')
    ranges = (
        lengthRatioRange,
        conductivityRatioRange,
        temperatureDifferenceRatioRange,
    )
    for lower, upper in ranges:
        if (
            not np.isfinite(lower)
            or not np.isfinite(upper)
            or lower <= 0.0
            or upper < lower
        ):
            raise ValueError('similarity augmentation ranges must be positive')
    generator = np.random.default_rng(randomSeed)

    def logUniform(bounds: tuple[float, float]) -> float:
        lower, upper = bounds
        return float(np.exp(generator.uniform(np.log(lower), np.log(upper))))

    augmented = list(cases)
    for case in cases:
        for augmentationIndex in range(augmentationsPerCase):
            augmented.append(
                similarityAugmentedThermalComponentCase(
                    case,
                    lengthRatio=logUniform(lengthRatioRange),
                    conductivityRatio=logUniform(conductivityRatioRange),
                    temperatureDifferenceRatio=logUniform(
                        temperatureDifferenceRatioRange
                    ),
                    schema=schema,
                    caseId=f'{case.caseId}-similarity-{augmentationIndex}',
                )
            )
    return tuple(augmented)

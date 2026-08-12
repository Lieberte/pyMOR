from dataclasses import replace

import numpy as np

from mor.nn.data import (
    STEADY_THERMAL_DIMENSIONLESS_V2,
    anchorReferenceTemperatureK,
    enforceMatchingDirichletQueries,
    similarityAugmentedThermalComponentCase,
    steadyThermalDimensionlessOperatorCase,
    steadyThermalDimensionlessV2GlobalFeatureNames,
    thermalBoundarySamples,
    thermalComponentFeatureSchema,
)
from tests.nn.data.test_thermal_component_case import makeComponentCase


def _geometricallySimilarCase(case, lengthRatio: float):
    return replace(
        case,
        caseId=f'{case.caseId}-scaled',
        geometryId=f'{case.geometryId}-scaled',
        volume=replace(
            case.volume,
            coordinatesMeters=case.volume.coordinatesMeters * lengthRatio,
            heatSourceWPerM3=case.volume.heatSourceWPerM3 / lengthRatio**2,
            weightsM3=case.volume.weightsM3 * lengthRatio**3,
        ),
        boundary=replace(
            case.boundary,
            coordinatesMeters=case.boundary.coordinatesMeters * lengthRatio,
            htcWPerM2K=case.boundary.htcWPerM2K / lengthRatio,
            heatFluxWPerM2=case.boundary.heatFluxWPerM2 / lengthRatio,
            weightsM2=case.boundary.weightsM2 * lengthRatio**2,
        ),
        temperature=replace(
            case.temperature,
            coordinatesMeters=case.temperature.coordinatesMeters * lengthRatio,
            weightsM3=case.temperature.weightsM3 * lengthRatio**3,
        ),
    )


def test_steady_dimensionless_features_are_invariant_for_similar_scaled_case():
    case = replace(
        makeComponentCase(),
        operatingFeatures={
            'speedRpm': 12000.0,
            'absoluteTemperatureAnchor': 1.0,
        },
    )
    scaledCase = _geometricallySimilarCase(case, 3.0)
    schema = thermalComponentFeatureSchema(
        operatingFeatureNames=('speedRpm', 'absoluteTemperatureAnchor'),
        boundaryTypeCount=2,
        temperatureBoundaryTypeIds=(1,),
    )
    arguments = {
        'schema': schema,
        'operatingScales': {
            'speedRpm': 25000.0,
            'absoluteTemperatureAnchor': 1.0,
        },
    }

    original = steadyThermalDimensionlessOperatorCase(case, **arguments)
    scaled = steadyThermalDimensionlessOperatorCase(scaledCase, **arguments)

    np.testing.assert_allclose(
        original.operatorCase.volumeFeatures,
        scaled.operatorCase.volumeFeatures,
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        original.operatorCase.boundaryFeatures,
        scaled.operatorCase.boundaryFeatures,
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        original.operatorCase.globalFeatures,
        scaled.operatorCase.globalFeatures,
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        original.operatorCase.queryCoordinates,
        scaled.operatorCase.queryCoordinates,
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        original.operatorCase.queryTemperatures,
        scaled.operatorCase.queryTemperatures,
        rtol=2e-6,
        atol=2e-6,
    )
    assert np.isclose(
        original.scales.temperatureScaleK,
        scaled.scales.temperatureScaleK,
    )


def test_mixed_boundary_reference_prefers_dirichlet_temperature():
    case = makeComponentCase()
    boundary = thermalBoundarySamples(
        coordinatesMeters=case.boundary.coordinatesMeters,
        normals=case.boundary.normals,
        htcWPerM2K=[100.0, 200.0, 300.0, 0.0, 0.0, 0.0],
        ambientTemperatureK=[280.0, 290.0, 300.0, 340.0, 350.0, 360.0],
        weightsM2=case.boundary.weightsM2,
        boundaryTypeIds=[1, 1, 1, 2, 2, 2],
        heatFluxWPerM2=case.boundary.heatFluxWPerM2,
        patchIds=case.boundary.patchIds,
    )
    case = replace(
        case,
        boundary=boundary,
        operatingFeatures={
            **case.operatingFeatures,
            'absoluteTemperatureAnchor': 1.0,
        },
    )
    schema = thermalComponentFeatureSchema(
        operatingFeatureNames=('speedRpm', 'absoluteTemperatureAnchor'),
        boundaryTypeCount=3,
        temperatureBoundaryTypeIds=(1, 2),
        fixedTemperatureBoundaryTypeIds=(2,),
    )

    assert np.isclose(
        anchorReferenceTemperatureK(case, schema=schema),
        350.0,
    )


def test_v2_similarity_augmentation_preserves_dimensionless_case():
    case = replace(
        makeComponentCase(),
        operatingFeatures={
            'speedRpm': 12000.0,
            'absoluteTemperatureAnchor': 1.0,
        },
    )
    schema = thermalComponentFeatureSchema(
        operatingFeatureNames=('speedRpm', 'absoluteTemperatureAnchor'),
        boundaryTypeCount=2,
        temperatureBoundaryTypeIds=(1,),
    )
    augmented = similarityAugmentedThermalComponentCase(
        case,
        lengthRatio=1.7,
        conductivityRatio=2.3,
        temperatureDifferenceRatio=1.4,
        referenceTemperatureShiftK=12.0,
        schema=schema,
    )
    arguments = {
        'schema': schema,
        'operatingScales': {
            'speedRpm': 25000.0,
            'absoluteTemperatureAnchor': 1.0,
        },
        'minimumTemperatureScaleK': 1e-6,
        'featureMode': STEADY_THERMAL_DIMENSIONLESS_V2,
    }

    original = steadyThermalDimensionlessOperatorCase(case, **arguments)
    transformed = steadyThermalDimensionlessOperatorCase(augmented, **arguments)

    np.testing.assert_allclose(
        original.operatorCase.volumeFeatures,
        transformed.operatorCase.volumeFeatures,
        rtol=3e-6,
        atol=3e-6,
    )
    np.testing.assert_allclose(
        original.operatorCase.boundaryFeatures,
        transformed.operatorCase.boundaryFeatures,
        rtol=3e-6,
        atol=3e-6,
    )
    np.testing.assert_allclose(
        original.operatorCase.globalFeatures,
        transformed.operatorCase.globalFeatures,
        rtol=3e-6,
        atol=3e-6,
    )
    np.testing.assert_allclose(
        original.operatorCase.queryTemperatures,
        transformed.operatorCase.queryTemperatures,
        rtol=3e-6,
        atol=3e-6,
    )
    assert original.operatorCase.globalFeatureDim == len(
        steadyThermalDimensionlessV2GlobalFeatureNames(schema)
    )


def test_similarity_augmentation_scales_temperature_floor_for_floating_case():
    case = makeComponentCase()
    case = replace(
        case,
        volume=replace(case.volume, heatSourceWPerM3=np.zeros(8)),
        boundary=thermalBoundarySamples(
            coordinatesMeters=case.boundary.coordinatesMeters,
            normals=case.boundary.normals,
            htcWPerM2K=np.zeros(6),
            ambientTemperatureK=case.boundary.ambientTemperatureK,
            weightsM2=case.boundary.weightsM2,
            boundaryTypeIds=np.zeros(6, dtype=int),
            heatFluxWPerM2=np.ones(6),
            patchIds=case.boundary.patchIds,
        ),
        operatingFeatures={'absoluteTemperatureAnchor': 0.0},
    )
    schema = thermalComponentFeatureSchema(
        operatingFeatureNames=('absoluteTemperatureAnchor',),
        boundaryTypeCount=1,
    )
    augmented = similarityAugmentedThermalComponentCase(
        case,
        temperatureDifferenceRatio=1.6,
        schema=schema,
    )
    arguments = {
        'schema': schema,
        'operatingScales': {'absoluteTemperatureAnchor': 1.0},
        'featureMode': STEADY_THERMAL_DIMENSIONLESS_V2,
    }

    original = steadyThermalDimensionlessOperatorCase(case, **arguments)
    transformed = steadyThermalDimensionlessOperatorCase(augmented, **arguments)

    assert np.isclose(original.scales.temperatureScaleK, 1.0)
    assert np.isclose(transformed.scales.temperatureScaleK, 1.6)
    np.testing.assert_allclose(
        original.operatorCase.queryTemperatures,
        transformed.operatorCase.queryTemperatures,
        rtol=2e-6,
        atol=2e-6,
    )


def test_matching_dirichlet_queries_are_hard_enforced():
    case = makeComponentCase()
    case = replace(
        case,
        boundary=thermalBoundarySamples(
            coordinatesMeters=case.boundary.coordinatesMeters,
            normals=case.boundary.normals,
            htcWPerM2K=[100.0, 100.0, 100.0, 0.0, 0.0, 0.0],
            ambientTemperatureK=[300.0, 300.0, 300.0, 340.0, 350.0, 360.0],
            weightsM2=case.boundary.weightsM2,
            boundaryTypeIds=[1, 1, 1, 2, 2, 2],
            heatFluxWPerM2=case.boundary.heatFluxWPerM2,
            patchIds=case.boundary.patchIds,
        ),
        operatingFeatures={'absoluteTemperatureAnchor': 1.0},
    )
    schema = thermalComponentFeatureSchema(
        operatingFeatureNames=('absoluteTemperatureAnchor',),
        boundaryTypeCount=3,
        temperatureBoundaryTypeIds=(1, 2),
        fixedTemperatureBoundaryTypeIds=(2,),
    )
    operatorCase = steadyThermalDimensionlessOperatorCase(
        case,
        schema=schema,
        operatingScales={'absoluteTemperatureAnchor': 1.0},
        featureMode=STEADY_THERMAL_DIMENSIONLESS_V2,
    ).operatorCase

    enforced, matched = enforceMatchingDirichletQueries(
        np.zeros(operatorCase.queryCoordinates.shape[0]),
        operatorCase,
        schema=schema,
    )

    assert matched == 3
    np.testing.assert_allclose(
        enforced[3:6],
        operatorCase.boundaryFeatures[3:6, 7],
    )

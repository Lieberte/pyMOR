from dataclasses import replace

import numpy as np
import torch

from mor.nn.data import (
    STEADY_THERMAL_DIMENSIONLESS_V2,
    enrichThermalAssemblyCases,
    thermalAssemblyBoundaryFeatureNames,
    thermalAssemblyContextFeatureNames,
    thermalBoundarySamples,
    thermalCheckpointDataContract,
)
from mor.nn.models import (
    thermalCategorizedOperatorTorch,
    thermalQuerySetOperatorTorch,
)
from mor.nn.runtime import thermalCategorizedPredictor
from tests.nn.data.test_thermal_component_case import makeComponentCase


def _assemblyCases():
    base = makeComponentCase()
    common = {
        'sourceDataset': 'test-dataset',
        'assemblyId': 'motor-1',
        'parentCaseId': 'operating-point-1',
    }
    statorBoundary = thermalBoundarySamples(
        coordinatesMeters=base.boundary.coordinatesMeters,
        normals=base.boundary.normals,
        htcWPerM2K=np.zeros(6),
        ambientTemperatureK=np.full(6, 999.0),
        weightsM2=base.boundary.weightsM2,
        boundaryTypeIds=np.zeros(6, dtype=np.int64),
        heatFluxWPerM2=base.boundary.heatFluxWPerM2,
        patchIds=base.boundary.patchIds,
    )
    stator = replace(
        base,
        caseId='stator',
        geometryId='stator-geometry',
        componentType='stator',
        boundary=statorBoundary,
        operatingFeatures={'speedRpm': 12000.0, 'absoluteTemperatureAnchor': 0.0},
        metadata={
            **common,
            'regionUuid': 'stator-region',
            'upstreamDirectory': 'electromagnetic',
            'surfacePatches': [
                {
                    'patchId': 0,
                    'interfaceUuid': 'stator-housing',
                    'oppositeRegionUuid': 'housing-region',
                    'contactConductanceWPerM2K': 250.0,
                }
            ],
        },
    )
    housing = replace(
        base,
        caseId='housing',
        geometryId='housing-geometry',
        componentType='housing',
        operatingFeatures={'speedRpm': 12000.0, 'absoluteTemperatureAnchor': 1.0},
        metadata={
            **common,
            'regionUuid': 'housing-region',
            'upstreamDirectory': 'mechanical',
            'surfacePatches': [
                {
                    'patchId': 0,
                    'interfaceUuid': 'stator-housing',
                    'oppositeRegionUuid': 'stator-region',
                    'contactConductanceWPerM2K': 250.0,
                }
            ],
        },
    )
    return stator, housing


def _metadata(componentTypes=('housing', 'stator')):
    contextNames = thermalAssemblyContextFeatureNames(componentTypes)
    return {
        'featureMode': STEADY_THERMAL_DIMENSIONLESS_V2,
        'operatingScales': {
            'speedRpm': 25000.0,
            'absoluteTemperatureAnchor': 1.0,
        },
        'minimumTemperatureScaleK': 1.0,
        'schema': {
            'operatingFeatureNames': [
                'speedRpm',
                *contextNames,
                'absoluteTemperatureAnchor',
            ],
            'boundaryTypeCount': 4,
            'temperatureBoundaryTypeIds': [1, 2],
            'fixedTemperatureBoundaryTypeIds': [2],
        },
        'assemblyContext': {
            'enabled': True,
            'componentTypes': list(componentTypes),
            'featureNames': list(contextNames),
        },
    }


def test_assembly_context_propagates_anchor_and_interface_neighbor_features():
    stator, housing = _assemblyCases()
    enrichedStator, enrichedHousing = enrichThermalAssemblyCases(
        (stator, housing), componentTypes=('housing', 'stator')
    )
    boundaryNames = thermalAssemblyBoundaryFeatureNames(('housing', 'stator'))
    oppositeHousingIndex = boundaryNames.index('oppositeComponentIs_housing')

    assert enrichedStator.operatingFeatures['absoluteTemperatureAnchor'] == 1.0
    assert enrichedStator.operatingFeatures['absoluteTemperatureReferenceK'] == 300.0
    assert enrichedHousing.operatingFeatures['componentTypeIs_housing'] == 1.0
    assert enrichedHousing.operatingFeatures['componentTypeIs_stator'] == 0.0
    np.testing.assert_allclose(enrichedStator.boundary.extraFeatures[:3, 0], 1.0)
    np.testing.assert_allclose(
        enrichedStator.boundary.extraFeatures[:3, oppositeHousingIndex],
        1.0,
    )
    np.testing.assert_allclose(enrichedStator.boundary.extraFeatures[:3, 7], 0.6)


def test_assembly_context_does_not_use_temperature_targets_as_features():
    cases = _assemblyCases()
    changed = tuple(
        replace(
            case,
            temperature=replace(
                case.temperature,
                temperatureK=case.temperature.temperatureK + 500.0,
            ),
        )
        for case in cases
    )

    original = enrichThermalAssemblyCases(cases, componentTypes=('housing', 'stator'))
    shifted = enrichThermalAssemblyCases(changed, componentTypes=('housing', 'stator'))

    for originalCase, shiftedCase in zip(original, shifted):
        assert originalCase.operatingFeatures == shiftedCase.operatingFeatures
        np.testing.assert_allclose(
            originalCase.boundary.extraFeatures,
            shiftedCase.boundary.extraFeatures,
        )


def test_checkpoint_contract_and_runtime_reconstruct_absolute_temperature():
    cases = _assemblyCases()
    metadata = _metadata()
    provisional = thermalCheckpointDataContract.fromCheckpoint(
        {'metadata': metadata, 'global': {'model': {}}}
    )
    _, dimensionless = provisional.prepareAndTransform(cases)
    example = dimensionless[0].operatorCase
    config = {
        'volumeFeatureDim': example.volumeFeatureDim,
        'boundaryFeatureDim': example.boundaryFeatureDim,
        'globalFeatureDim': example.globalFeatureDim,
        'queryDim': example.queryDim,
        'hiddenDim': 8,
        'kernelScales': (0.3,),
        'queryFourierFrequencies': 1,
        'decomposeMeanAndShape': False,
    }
    model = thermalQuerySetOperatorTorch(**config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    checkpoint = thermalCategorizedOperatorTorch(
        model,
        {},
        categoryFields=('componentType',),
    ).toCheckpoint(globalModelConfig=config, metadata=metadata)

    predictor = thermalCategorizedPredictor.fromCheckpoint(checkpoint, device='cpu')
    prepared = predictor.dataContract.prepareCases(cases)
    preparedAgain = predictor.dataContract.prepareCases(prepared)
    assert preparedAgain[0] is prepared[0]
    predictions = predictor.predictCases(cases)

    assert [value.caseId for value in predictions] == ['stator', 'housing']
    assert all(value.absoluteTemperatureAvailable for value in predictions)
    assert all(value.routing['usedGlobalFallback'] for value in predictions)
    for value in predictions:
        np.testing.assert_allclose(value.temperatureK, 300.0)


def test_runtime_uses_unsupported_components_only_as_assembly_context():
    cases = _assemblyCases()
    metadata = {
        **_metadata(),
        'supportedComponentTypes': ['stator'],
    }
    provisional = thermalCheckpointDataContract.fromCheckpoint(
        {'metadata': metadata, 'global': {'model': {}}}
    )
    _, dimensionless = provisional.prepareAndTransform(cases)
    example = dimensionless[0].operatorCase
    config = {
        'volumeFeatureDim': example.volumeFeatureDim,
        'boundaryFeatureDim': example.boundaryFeatureDim,
        'globalFeatureDim': example.globalFeatureDim,
        'queryDim': example.queryDim,
        'hiddenDim': 8,
        'kernelScales': (0.3,),
        'queryFourierFrequencies': 1,
        'decomposeMeanAndShape': False,
    }
    model = thermalQuerySetOperatorTorch(**config)
    checkpoint = thermalCategorizedOperatorTorch(
        model, {}, categoryFields=('componentType',)
    ).toCheckpoint(globalModelConfig=config, metadata=metadata)

    prediction = thermalCategorizedPredictor.fromCheckpoint(
        checkpoint, device='cpu'
    ).predictCases(cases)

    assert [value.componentType for value in prediction] == ['stator']

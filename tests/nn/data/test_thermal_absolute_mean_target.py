import numpy as np
from dataclasses import replace

from scripts.run_motor_fem_absolute_mean_holdout import (
    TEMPERATURE_REFERENCE_K,
    TEMPERATURE_SCALE_K,
    _speedInterpolationBaseline,
    _podCoefficientInterpolationBaselines,
    _targets,
)
from tests.nn.data.test_thermal_component_case import makeComponentCase


def test_absolute_mean_target_uses_kelvin_mean_without_relative_gauge():
    case = makeComponentCase()

    target = _targets((case,), 'cpu').numpy()[0, 0]
    expectedMean = np.average(
        case.temperature.temperatureK,
        weights=case.temperature.weightsM3,
    )

    assert np.isclose(
        target,
        (expectedMean - TEMPERATURE_REFERENCE_K) / TEMPERATURE_SCALE_K,
    )


def test_speed_mean_baseline_predicts_absolute_temperature_for_every_component():
    trainCases = (
        replace(makeComponentCase('shaft-0'), operatingFeatures={'speedRpm': 0.0}),
        replace(makeComponentCase('shaft-2'), operatingFeatures={'speedRpm': 2.0}),
    )
    testCase = replace(
        makeComponentCase('shaft-1'), operatingFeatures={'speedRpm': 1.0}
    )

    records = _speedInterpolationBaseline(trainCases, (testCase,))

    assert records[0]['predictedMeanTemperatureK'] is not None
    assert 'absoluteTemperatureAnchorAvailable' not in records[0]


def test_full_rank_pod_coefficient_interpolation_preserves_mean_interpolation():
    base = makeComponentCase('shaft-0')
    trainCases = (
        replace(base, operatingFeatures={'speedRpm': 0.0}),
        replace(
            base,
            caseId='shaft-2',
            operatingFeatures={'speedRpm': 2.0},
            temperature=replace(
                base.temperature,
                temperatureK=base.temperature.temperatureK + 2.0,
            ),
        ),
    )
    testCase = replace(
        base,
        caseId='shaft-1',
        operatingFeatures={'speedRpm': 1.0},
        temperature=replace(
            base.temperature,
            temperatureK=base.temperature.temperatureK + 1.0,
        ),
    )

    recordsByRank, diagnostics = _podCoefficientInterpolationBaselines(
        trainCases,
        (testCase,),
        requestedRanks=(None,),
        neighborCount=2,
    )
    record = next(iter(recordsByRank.values()))[0]

    assert diagnostics['shaft']['availableRank'] >= 1
    assert np.isclose(record['errorK'], 0.0, atol=1e-6)

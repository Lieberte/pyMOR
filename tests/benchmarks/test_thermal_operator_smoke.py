import math

import pytest

from scripts.run_thermal_operator_smoke import buildSyntheticCases, runVariant


@pytest.mark.benchmark
def test_cross_geometry_thermal_operator_training_smoke_metrics():
    result = runVariant(
        'query_set_smoke',
        {
            'modelType': 'query_set',
            'kernelScales': (0.3,),
            'queryFourierFrequencies': 1,
        },
        buildSyntheticCases(casesPerGeometry=2, seed=3),
        epochs=4,
        deviceName='cpu',
        seed=3,
    )

    assert result['bestEpochIndex'] >= 0
    assert len(result['trainLossHistory']) == 4
    assert math.isfinite(result['metrics']['rmseKelvin'])
    assert math.isfinite(result['metrics']['relativeL2'])
    assert result['metrics']['caseMetrics']
    assert result['shuffledFieldMetrics']['rmseKelvin'] > 0.0

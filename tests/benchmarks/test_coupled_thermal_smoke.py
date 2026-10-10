import pytest

from scripts.generate_coupled_thermal_report import runCharacterization


@pytest.mark.benchmark
def test_coupled_thermal_reduction_held_out_input_accuracy_and_conservation():
    report = runCharacterization(nodeCount=8, ranks=(2, 4, 8))
    rows = report['rows']
    assert rows[0]['reducedOrder'] < rows[0]['fullOrder']
    assert rows[-1]['maxAbsErrorKelvin'] < 1e-9
    assert rows[-1]['maxInterfacePowerErrorW'] < 1e-8
    assert rows[1]['rmseKelvin'] < rows[0]['rmseKelvin']
    for row in rows:
        assert row['maxEnergyBalanceResidualW'] < 1e-8
        assert row['factorizationCount'] == 1
        assert row['reducedOnlineSeconds'] > 0

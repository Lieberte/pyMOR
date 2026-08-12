import numpy as np
import pytest

from scripts.mor_application_cases import generatedHeatModel, runGeneratedHeatCase, sampledReductionError
from mor.reductors.balancedTruncation import balancedTruncationReductor


pytestmark = pytest.mark.application


def test_generated_heat_balanced_truncation_has_physical_quality_metrics():
    result = runGeneratedHeatCase(gridSize=10, reducedOrder=6)

    assert result["success"]
    assert result["quality_pass"]
    assert result["algorithm"] == "lradi"
    assert result["fom_order"] == 100
    assert result["rom_order"] == 6
    assert result["fom_stability_metric"] < 0
    assert result["rom_stability_metric"] < 0
    assert result["sampled_max_frequency_error"] <= result["hankel_error_bound"] * (1 + 1e-6) + 1e-10


def test_generated_heat_frequency_error_improves_with_order():
    model = generatedHeatModel(gridSize=8)
    frequencies = np.logspace(-2, 2, 32)
    reductor = balancedTruncationReductor(globalBackendName="scipy")

    rom2 = reductor.reduce(model, order=2)
    error2 = sampledReductionError(model, rom2.model, frequencies)
    rom6 = reductor.reduce(model, order=6)
    error6 = sampledReductionError(model, rom6.model, frequencies)

    assert error6 < error2
    assert error2 <= rom2.solverInfo["hankelErrorBound"] * (1 + 1e-6) + 1e-10
    assert error6 <= rom6.solverInfo["hankelErrorBound"] * (1 + 1e-6) + 1e-10

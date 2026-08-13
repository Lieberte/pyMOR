from pathlib import Path
import numpy as np
import pytest

from scripts.mor_application_cases import (
    SLICOT_CASES,
    SLICOT_CASES_BY_KEY,
    casePath,
    loadCaseData,
    referenceMagnitudeRelativeError,
    runReductionCase,
    transferResponse,
)


pytestmark = pytest.mark.application


def test_transfer_response_uses_unit_circle_for_discrete_models():
    A = np.diag([0.2, 0.5])
    B = np.ones((2, 1))
    C = np.ones((1, 2))

    response = transferResponse(A, B, C, [0.0], isContinuous=False)
    expected = C @ np.linalg.solve(np.eye(2) - A, B)

    np.testing.assert_allclose(response[0], expected)
    assert not np.allclose(response, transferResponse(A, B, C, [0.0], isContinuous=True))


def _requireData(case):
    path = casePath(case)
    if not path.exists():
        pytest.skip("external SLICOT data is not installed; run scripts/fetch_mor_benchmark_data.py")
    return path


@pytest.mark.parametrize("case", SLICOT_CASES, ids=lambda case: case.key)
def test_slicot_case_metadata_and_reference_response(case):
    path = _requireData(case)
    assert isinstance(path, Path)
    data = loadCaseData(case)

    assert data["A"].shape == (case.order, case.order)
    assert data["B"].shape[0] == case.order
    assert data["C"].shape[1] == case.order
    assert data["B"].shape[1] == case.inputs
    assert data["C"].shape[0] == case.outputs

    referenceError = referenceMagnitudeRelativeError(case, data)
    if case.referenceMagnitudeTolerance is not None:
        assert referenceError is not None
        assert referenceError <= case.referenceMagnitudeTolerance


@pytest.mark.parametrize("caseKey", ["heat_continuous", "heat_discrete", "cd_player", "building", "clamped_beam", "transmission_line", "peec"])
def test_slicot_supported_reduction_matches_reference(caseKey):
    case = SLICOT_CASES_BY_KEY[caseKey]
    _requireData(case)
    result = runReductionCase(case)

    assert result["success"]
    assert result["quality_pass"]
    if caseKey == "heat_continuous":
        assert result["algorithm"] == "lradi"
    elif caseKey == "peec":
        assert result["algorithm"] == "frequencyResponse"
    else:
        assert result["algorithm"] == "bartelsStewart"
    if case.compareReferenceHsv:
        assert result["hsv_relative_error"] <= case.hsvTolerance
    if case.frequencyRelativeTolerance is not None:
        assert result["frequency_relative_error"] <= case.frequencyRelativeTolerance
    else:
        assert result["sampled_max_frequency_error"] <= result["hankel_error_bound"] * (1 + 1e-6) + 1e-10
    if caseKey == "peec":
        assert result["rom_order"] == 60
    elif caseKey == "heat_discrete":
        assert result["rom_stability_metric"] == pytest.approx(result["fom_stability_metric"])
    else:
        assert result["rom_stability_metric"] < 0

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import linalg
from scipy.io import loadmat
from scipy.sparse import diags, eye, issparse, kron


SLICOT_REPOSITORY = "https://github.com/SLICOT/Benchmark-ModelReduction"
SLICOT_RAW_BASE = "https://raw.githubusercontent.com/SLICOT/Benchmark-ModelReduction/main"


@dataclass(frozen=True)
class applicationCase:
    key: str
    filename: str
    title: str
    description: str
    order: int
    inputs: int
    outputs: int
    reducedOrder: int
    sha256: str
    isContinuous: bool = True
    descriptor: bool = False
    primaryPolicy: str = "auto"
    expectedStatus: str = "pass"
    expectedRisk: str = "keep"
    expectedFailure: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    hsvTolerance: float = 1e-6
    referenceMagnitudeTolerance: float | None = 1e-8
    preflightStability: bool = False
    compareReferenceHsv: bool = True
    frequencyRelativeTolerance: float | None = None

    @property
    def url(self) -> str:
        return f"{SLICOT_RAW_BASE}/{self.filename}"

    def toManifest(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            sourceRepository=SLICOT_REPOSITORY,
            sourceUrl=self.url,
            upstreamLicense="not declared at repository level",
            redistribution="not bundled; explicit checksum-verified download",
        )
        return result


SLICOT_CASES = (
    applicationCase(
        key="heat_continuous",
        filename="heat-cont.mat",
        title="Continuous thin-rod heat equation",
        description="Spatial heat transfer in a thin rod.",
        order=200,
        inputs=1,
        outputs=1,
        reducedOrder=8,
        sha256="351988885FF2768A42896B4901E6D82CA3F42BD36E4DF9C686FA12A9C12C533B",
    ),
    applicationCase(
        key="heat_discrete",
        filename="heat-disc.mat",
        title="Spatially discrete heat-equation benchmark",
        description="Spatial discretization of the heat equation with an unstable continuous-time mode.",
        order=200,
        inputs=1,
        outputs=1,
        reducedOrder=8,
        sha256="00FBAA8143D55BD8E663DFC607B94A9B8FA66E7430B10E9644AC611B7FF59B7A",
        expectedRisk="keep",
        expectedStatus="pass",
        options={"unstableHandling": "preserve"},
        compareReferenceHsv=False,
    ),
    applicationCase(
        key="cd_player",
        filename="CDplayer.mat",
        title="Classical CD-player model",
        description="Two-input, two-output servo-control model.",
        order=120,
        inputs=2,
        outputs=2,
        reducedOrder=12,
        sha256="354044D90F5F3A6CD07D3C7E5EEE3CB2E8CB5A638295B5074B44DAF5E670E453",
        expectedStatus="pass",
        expectedRisk="keep",
    ),
    applicationCase(
        key="building",
        filename="build.mat",
        title="Building-motion model",
        description="Structural response of a building.",
        order=48,
        inputs=1,
        outputs=1,
        reducedOrder=8,
        sha256="9513BFDF12FB9382A4491755464702BF0E75567D9E4090F9F67F27EBF92ECAC7",
        expectedStatus="pass",
        expectedRisk="keep",
    ),
    applicationCase(
        key="clamped_beam",
        filename="beam.mat",
        title="Clamped-beam model",
        description="Flexible clamped-beam structural dynamics.",
        order=348,
        inputs=1,
        outputs=1,
        reducedOrder=12,
        sha256="9E20DC1C48B8255918E9C9C9D4B092C8C6367FBA58AB5553C2AF80A12311F7A6",
        expectedStatus="pass",
        expectedRisk="keep",
    ),
    applicationCase(
        key="transmission_line",
        filename="tline.mat",
        title="Transmission-line descriptor model",
        description="Two-port transmission-line model with a descriptor matrix.",
        order=256,
        inputs=2,
        outputs=2,
        reducedOrder=12,
        sha256="6EE3A3AEB00ED8AFA19145BF46F63984F20602C7F89DFCC3548930A3D9F5B553",
        descriptor=True,
        primaryPolicy="dense_bartels_stewart",
        expectedStatus="pass",
        expectedRisk="keep",
        hsvTolerance=2e-6,
    ),
    applicationCase(
        key="peec",
        filename="peec.mat",
        title="PEEC descriptor circuit",
        description="Partial-element equivalent-circuit descriptor model.",
        order=480,
        inputs=1,
        outputs=1,
        reducedOrder=60,
        sha256="89DAFFEDF5E21ABE15D9731FBD7741CD5504CEBC84D806237D460BFB1AB7DB6B",
        descriptor=True,
        primaryPolicy="frequency_response",
        expectedStatus="pass",
        expectedRisk="keep",
        referenceMagnitudeTolerance=None,
        compareReferenceHsv=False,
        frequencyRelativeTolerance=1e-6,
    ),
)

SLICOT_CASES_BY_KEY = {case.key: case for case in SLICOT_CASES}


def projectRoot() -> Path:
    return Path(__file__).resolve().parents[1]


def defaultDataDirectory() -> Path:
    return projectRoot() / "tests" / "data" / "slicot"


def fileSha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def casePath(case: applicationCase, dataDirectory: Path | None = None) -> Path:
    return (dataDirectory or defaultDataDirectory()) / case.filename


def loadCaseData(case: applicationCase, dataDirectory: Path | None = None) -> dict[str, Any]:
    path = casePath(case, dataDirectory)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; run scripts/fetch_mor_benchmark_data.py to fetch external benchmark data"
        )
    actualHash = fileSha256(path)
    if actualHash != case.sha256:
        raise ValueError(f"Checksum mismatch for {path}: expected {case.sha256}, got {actualHash}")
    return loadmat(path)


def denseData(value: Any) -> np.ndarray:
    return value.toarray() if issparse(value) else np.asarray(value)


def caseArrays(data: dict[str, Any], denseOperators: bool = False) -> dict[str, Any]:
    arrays = {
        "A": data["A"],
        "B": data["B"],
        "C": data["C"],
        "D": data.get("D"),
        "E": data.get("E"),
    }
    arrays["B"] = denseData(arrays["B"])
    arrays["C"] = denseData(arrays["C"])
    if arrays["D"] is not None:
        arrays["D"] = denseData(arrays["D"])
    if denseOperators:
        arrays["A"] = denseData(arrays["A"])
        if arrays["E"] is not None:
            arrays["E"] = denseData(arrays["E"])
    return arrays


def transferResponse(A: Any, B: Any, C: Any, frequencies: np.ndarray, E: Any | None = None, D: Any | None = None) -> np.ndarray:
    aData, bData, cData = denseData(A), denseData(B), denseData(C)
    eData = np.eye(aData.shape[0], dtype=aData.dtype) if E is None else denseData(E)
    dData = np.zeros((cData.shape[0], bData.shape[1]), dtype=aData.dtype) if D is None else denseData(D)
    responses = []
    for frequency in np.asarray(frequencies).reshape(-1):
        responses.append(cData @ np.linalg.solve(1j * frequency * eData - aData, bData) + dData)
    return np.asarray(responses)


def referenceMagnitudeRelativeError(case: applicationCase, data: dict[str, Any]) -> float | None:
    if case.referenceMagnitudeTolerance is None or "mag" not in data or "w" not in data:
        return None
    arrays = caseArrays(data)
    response = transferResponse(**arrays, frequencies=np.ravel(data["w"]))
    calculated = np.abs(response).reshape(response.shape[0], -1, order="F")
    reference = np.asarray(data["mag"])
    return float(linalg.norm(calculated - reference) / max(linalg.norm(reference), np.finfo(float).tiny))


def referenceHankelSingularValues(case: applicationCase, data: dict[str, Any]) -> np.ndarray | None:
    if case.descriptor and all(name in data for name in ("R", "S", "E")):
        return linalg.svdvals(denseData(data["R"]) @ denseData(data["E"]) @ denseData(data["S"]).T)
    if "hsv" in data:
        return np.ravel(data["hsv"])
    return None


def generalizedEigenvalues(A: Any, E: Any | None = None) -> np.ndarray:
    return linalg.eigvals(denseData(A), denseData(E)) if E is not None else linalg.eigvals(denseData(A))


def stabilityMetric(A: Any, E: Any | None, isContinuous: bool) -> float:
    eigenvalues = generalizedEigenvalues(A, E)
    return float(np.max(eigenvalues.real) if isContinuous else np.max(np.abs(eigenvalues)))


def buildCaseModel(case: applicationCase, data: dict[str, Any], denseOperators: bool = False):
    from mor.models.lti import ltiModel
    from mor.operators import matrixOperator

    arrays = caseArrays(data, denseOperators=denseOperators)
    operators = {
        name: matrixOperator(value, backendName="scipy") if value is not None else None
        for name, value in arrays.items()
    }
    return ltiModel(
        operators["A"],
        operators["B"],
        operators["C"],
        operators["D"],
        operators["E"],
        backendName="scipy",
        isContinuous=case.isContinuous,
    )


def sampledReductionError(model, reducedModel, frequencies: np.ndarray, sampleCount: int = 24) -> float:
    indices = np.unique(np.linspace(0, len(frequencies) - 1, min(sampleCount, len(frequencies)), dtype=int))
    selected = np.asarray(frequencies).reshape(-1)[indices]
    full = transferResponse(
        model.A.toBackendData(), model.B.toBackendData(), model.C.toBackendData(), selected,
        model.E.toBackendData() if model.E is not None else None,
        model.D.toBackendData() if model.D is not None else None,
    )
    reduced = transferResponse(
        reducedModel.A.toBackendData(), reducedModel.B.toBackendData(), reducedModel.C.toBackendData(), selected,
        reducedModel.E.toBackendData() if reducedModel.E is not None else None,
        reducedModel.D.toBackendData() if reducedModel.D is not None else None,
    )
    return float(max(linalg.norm(full[index] - reduced[index], 2) for index in range(len(selected))))


def runReductionCase(case: applicationCase, dataDirectory: Path | None = None, policy: str | None = None) -> dict[str, Any]:
    from mor.reductors.balancedTruncation import balancedTruncationReductor

    selectedPolicy = policy or case.primaryPolicy
    denseOperators = selectedPolicy in ("dense_bartels_stewart", "frequency_response")
    options = dict(case.options)
    if denseOperators:
        options["lyapunovVariant"] = "bartelsStewart"
    data = loadCaseData(case, dataDirectory)
    model = buildCaseModel(case, data, denseOperators=denseOperators)
    started = time.perf_counter()
    result = {
        "case": case.key,
        "title": case.title,
        "source": "SLICOT Benchmark-ModelReduction",
        "policy": selectedPolicy,
        "expected_status": case.expectedStatus,
        "expected_risk": case.expectedRisk,
        "fom_order": case.order,
        "rom_order": case.reducedOrder,
        "inputs": case.inputs,
        "outputs": case.outputs,
        "is_continuous": case.isContinuous,
        "descriptor": case.descriptor,
        "reference_magnitude_relative_error": referenceMagnitudeRelativeError(case, data),
        "success": False,
    }
    if case.preflightStability:
        arrays = caseArrays(data, denseOperators=True)
        metric = stabilityMetric(arrays["A"], arrays["E"], case.isContinuous)
        result.update(
            elapsed_seconds=float(time.perf_counter() - started),
            fom_stability_metric=metric,
            error_type="UnsupportedBenchmarkRegime",
            error=(
                f"balanced truncation requires a stable continuous model; spectral abscissa is {metric:.6g}"
                if case.isContinuous
                else f"balanced truncation requires a stable discrete model; spectral radius is {metric:.6g}"
            ),
            quality_pass=False,
        )
        return result
    try:
        if selectedPolicy == "frequency_response":
            from mor.reductors.frequencyResponse import frequencyResponseReductor
            reduced = frequencyResponseReductor(
                globalBackendName="scipy", frequencies=np.ravel(data["w"])
            ).reduce(model, order=case.reducedOrder)
        else:
            reduced = balancedTruncationReductor(globalBackendName="scipy", **options).reduce(
                model, order=case.reducedOrder
            )
        result.update(
            success=True,
            elapsed_seconds=float(time.perf_counter() - started),
            algorithm=reduced.solverInfo.get("lyapunov", reduced.solverInfo.get("reductor")),
            fom_stability_metric=stabilityMetric(
                model.A.toBackendData(), model.E.toBackendData() if model.E is not None else None, case.isContinuous
            ),
            rom_stability_metric=stabilityMetric(
                reduced.Ar.toBackendData(), reduced.Er.toBackendData(), case.isContinuous
            ),
        )
        if "hankelErrorBound" in reduced.solverInfo:
            result["hankel_error_bound"] = float(reduced.solverInfo["hankelErrorBound"])
        if reduced.solverInfo.get("unstableHandling") == "preserve":
            result["preserved_unstable_modes"] = reduced.solverInfo["preservedUnstableModes"]
        referenceHsv = referenceHankelSingularValues(case, data) if case.compareReferenceHsv else None
        if referenceHsv is not None:
            referenceHsv = referenceHsv[:case.reducedOrder]
            computedHsv = np.asarray(reduced.hsv)
            result["hsv_relative_error"] = float(
                linalg.norm(computedHsv - referenceHsv) / max(linalg.norm(referenceHsv), np.finfo(float).tiny)
            )
            if case.descriptor and "hsv" in data:
                upstreamHsv = np.ravel(data["hsv"])[:case.reducedOrder]
                result["upstream_hsv_relative_error"] = float(
                    linalg.norm(computedHsv - upstreamHsv)
                    / max(linalg.norm(upstreamHsv), np.finfo(float).tiny)
                )
        if "w" in data:
            result["sampled_max_frequency_error"] = sampledReductionError(
                model, reduced.model, np.ravel(data["w"])
            )
            if case.frequencyRelativeTolerance is not None:
                frequencies = np.ravel(data["w"])
                fullResponse = transferResponse(
                    model.A.toBackendData(), model.B.toBackendData(), model.C.toBackendData(), frequencies,
                    model.E.toBackendData() if model.E is not None else None,
                    model.D.toBackendData() if model.D is not None else None,
                )
                reducedResponse = transferResponse(
                    reduced.Ar.toBackendData(), reduced.Br.toBackendData(), reduced.Cr.toBackendData(), frequencies,
                    reduced.Er.toBackendData(), reduced.Dr.toBackendData(),
                )
                result["frequency_relative_error"] = float(
                    linalg.norm(reducedResponse - fullResponse)
                    / max(linalg.norm(fullResponse), np.finfo(float).tiny)
                )
        stabilityPass = True if selectedPolicy == "frequency_response" else (
            abs(result["rom_stability_metric"] - result["fom_stability_metric"])
            <= 1e-8 * max(1.0, abs(result["fom_stability_metric"]))
            if reduced.solverInfo.get("unstableHandling") == "preserve"
            else (result["rom_stability_metric"] < 0.0 if case.isContinuous else result["rom_stability_metric"] < 1.0)
        )
        frequencyPass = (
            result.get("frequency_relative_error", 0.0) <= case.frequencyRelativeTolerance
            if case.frequencyRelativeTolerance is not None
            else result.get("sampled_max_frequency_error", 0.0)
            <= result["hankel_error_bound"] * (1.0 + 1e-6) + 1e-10
        )
        result["quality_pass"] = bool(
            result.get("hsv_relative_error", 0.0) <= case.hsvTolerance
            and frequencyPass
            and stabilityPass
        )
    except Exception as error:
        result.update(
            elapsed_seconds=float(time.perf_counter() - started),
            error_type=type(error).__name__,
            error=str(error),
            quality_pass=False,
        )
    return result


def generatedHeatModel(gridSize: int = 10):
    from mor.models.lti import ltiModel
    from mor.operators import matrixOperator

    oneDimensional = diags(
        [-np.ones(gridSize - 1), 2.0 * np.ones(gridSize), -np.ones(gridSize - 1)],
        [-1, 0, 1],
        format="csr",
    )
    stiffness = kron(eye(gridSize, format="csr"), oneDimensional, format="csr") + kron(
        oneDimensional, eye(gridSize, format="csr"), format="csr"
    )
    A = -stiffness
    B = np.zeros((gridSize * gridSize, 1))
    B[np.arange(gridSize) * gridSize, 0] = 1.0 / gridSize
    C = np.zeros((1, gridSize * gridSize))
    C[0, np.arange(gridSize) * gridSize + gridSize - 1] = 1.0 / gridSize
    return ltiModel(
        matrixOperator(A, backendName="scipy"),
        matrixOperator(B, backendName="scipy"),
        matrixOperator(C, backendName="scipy"),
        backendName="scipy",
    )


def runGeneratedHeatCase(gridSize: int = 10, reducedOrder: int = 6) -> dict[str, Any]:
    from mor.reductors.balancedTruncation import balancedTruncationReductor

    model = generatedHeatModel(gridSize)
    frequencies = np.logspace(-2, 2, 40)
    started = time.perf_counter()
    reduced = balancedTruncationReductor(globalBackendName="scipy").reduce(model, order=reducedOrder)
    result = {
        "case": "generated_heat_2d",
        "title": "Generated two-dimensional heat equation",
        "source": "locally generated finite-difference model",
        "policy": "auto",
        "expected_status": "pass",
        "expected_risk": "keep",
        "fom_order": model.order,
        "rom_order": reduced.order,
        "inputs": model.inputDim,
        "outputs": model.outputDim,
        "is_continuous": True,
        "descriptor": False,
        "success": True,
        "elapsed_seconds": float(time.perf_counter() - started),
        "algorithm": reduced.solverInfo.get("lyapunov"),
        "hankel_error_bound": float(reduced.solverInfo["hankelErrorBound"]),
        "sampled_max_frequency_error": sampledReductionError(model, reduced.model, frequencies),
        "fom_stability_metric": stabilityMetric(model.A.toBackendData(), None, True),
        "rom_stability_metric": stabilityMetric(reduced.Ar.toBackendData(), reduced.Er.toBackendData(), True),
    }
    result["quality_pass"] = bool(
        result["sampled_max_frequency_error"] <= result["hankel_error_bound"] * (1.0 + 1e-6) + 1e-10
        and result["rom_stability_metric"] < 0.0
    )
    return result

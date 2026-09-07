"""Rebuild compact stator cases using solved FEM connection heat flows.

This deliberately writes a separate component dataset. It does not modify the
published geometrymotor dataset; the latter can be replaced only after this
dataset and a retrained checkpoint have been reviewed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Iterable
from zipfile import ZipFile

from mor.nn.data import saveThermalComponentDataset
from mor.nn.data.motorFemComponentCase import (
    CONVECTION_BOUNDARY_TYPE,
    FIXED_TEMPERATURE_BOUNDARY_TYPE,
    PRESCRIBED_HEAT_FLUX_BOUNDARY_TYPE,
    RESULT_HEAT_FLUX_BOUNDARY_TYPE,
    _componentTypeName,
    _findRegion,
    _normalizedComponentName,
    _resolveFemSystem,
    buildMotorFemComponentCase,
)
from mor.nn.data.motorStaDataset import inspectMotorStaArchive


ROOT = Path(r"F:\mor")
HTC_ROOT = ROOT / "motor" / "motor htc" / "motor htc"
ARCHIVE_SOURCES = (
    (ROOT / "motor" / "df10000final.sta", "motor-df-prototype", "tutorial-k"),
    (ROOT / "motor" / "df5457final.sta", "motor-df-prototype", "tutorial-p"),
    (ROOT / "motor" / "df5457final1218.sta", "motor-df-prototype", "tutorial-q"),
    (ROOT / "motor" / "hk2.sta", "motor-hk-prototype", "tutorial-r"),
    (ROOT / "gearbox" / "1D-3D 电机网络 (1).sta", "motor-network-prototype", "tutorial-s"),
    (ROOT / "gearbox" / "1D-3D 电机网络 (12.sta", "motor-network-prototype", "tutorial-t"),
    (ROOT / "motor" / "结果文件" / "结果文件" / "case1109_12000r_  稳态.sta", "motor-project-prototype", "tutorial-ad"),
    (ROOT / "motor" / "结果文件" / "结果文件" / "case1109_4000r_稳态.sta", "motor-project-prototype", "tutorial-ag"),
    (ROOT / "motor" / "结果文件" / "结果文件" / "case1115_7000r_稳态.sta", "motor-project-prototype", "tutorial-aj"),
)


def _extract(archive: ZipFile, member: str, root: Path) -> None:
    info = archive.getinfo(member)
    target = root.joinpath(*PurePosixPath(member).parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(info) as source, target.open("wb") as destination:
        shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)


def _compactCase(case, *, motorId: str, parentCaseId: str, sourceSta: Path, sourceCase: str):
    from dataclasses import replace

    return replace(
        case,
        metadata={
            **case.metadata,
            "motorId": motorId,
            "assemblyId": motorId,
            "parentCaseId": parentCaseId,
            "sourceSta": str(sourceSta.resolve()),
            "sourceCase": sourceCase,
        },
    )


def _build_numeric_case(speed: str):
    source = HTC_ROOT / speed
    return _compactCase(
        buildMotorFemComponentCase(
            source,
            "stator",
            maximumVolumeSamples=128,
            maximumBoundarySamples=256,
            maximumQuerySamples=128,
            randomSeed=23 + int(float(speed)),
            speedRpm=float(speed),
            motorId="motor-htc-prototype",
            parentCaseId=f"tutorial-{chr(ord('a') + ('2500', '5000', '7500', '10000', '12500', '15000', '17500', '20000', '22500', '25000').index(speed))}",
        ),
        motorId="motor-htc-prototype",
        parentCaseId=f"tutorial-{chr(ord('a') + ('2500', '5000', '7500', '10000', '12500', '15000', '17500', '20000', '22500', '25000').index(speed))}",
        sourceSta=source,
        sourceCase=speed,
    )


def _statorName(names: Iterable[str]) -> str:
    for name in names:
        if _componentTypeName(name) == "stator":
            return name
    raise KeyError(f"could not identify stator component in {tuple(names)!r}")


def _build_archive_case(source: Path, motorId: str, parentCaseId: str, temporaryParent: Path):
    info = inspectMotorStaArchive(source)
    with ZipFile(source) as archive:
        control = json.loads(archive.read("controlData.json").decode("utf-8-sig"))
        _, femSystem, _ = _resolveFemSystem(control)
        component = _statorName(
            _normalizedComponentName(region.get("name", ""))
            for region in femSystem.get("femRegions", {}).values()
        )
        _, region = _findRegion(femSystem, component)
        extracted = Path(tempfile.mkdtemp(prefix="stator-rebuild-", dir=temporaryParent))
        try:
            _extract(archive, "controlData.json", extracted)
            _extract(archive, "controlData.h5", extracted)
            _extract(archive, f"mesh/{region['meshUuid']}.samesh", extracted)
            _extract(archive, info.resultMember, extracted)
            case = buildMotorFemComponentCase(
                extracted,
                component,
                maximumVolumeSamples=128,
                maximumBoundarySamples=256,
                maximumQuerySamples=128,
                randomSeed=23,
                resultFile=info.resultMember,
                speedRpm=None,
                motorId=motorId,
                parentCaseId=parentCaseId,
            )
            return _compactCase(
                case,
                motorId=motorId,
                parentCaseId=parentCaseId,
                sourceSta=source,
                sourceCase=source.name,
            )
        finally:
            shutil.rmtree(extracted, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    speeds = ("2500", "5000", "7500", "10000", "12500", "15000", "17500", "20000", "22500", "25000")
    for position, speed in enumerate(speeds, start=1):
        print(f"[{position}/19] {speed} stator", flush=True)
        cases.append(_build_numeric_case(speed))
    for position, (source, motorId, parentCaseId) in enumerate(ARCHIVE_SOURCES, start=11):
        print(f"[{position}/19] {source.name} stator", flush=True)
        cases.append(_build_archive_case(source, motorId, parentCaseId, output.parent))

    indexPath = saveThermalComponentDataset(
        cases,
        output,
        boundaryTypeNames={
            RESULT_HEAT_FLUX_BOUNDARY_TYPE: "resultHeatFlux",
            CONVECTION_BOUNDARY_TYPE: "convection",
            FIXED_TEMPERATURE_BOUNDARY_TYPE: "fixedTemperature",
            PRESCRIBED_HEAT_FLUX_BOUNDARY_TYPE: "prescribedHeatFlux",
        },
        datasetMetadata={
            "datasetType": "statorConnectionHeatFlowRebuild",
            "componentType": "stator",
            "caseCount": len(cases),
            "sourcePolicy": "result connections/heatFlowRate for resultHeatFlux patches",
        },
        overwrite=True,
    )
    summary = {
        "dataset": str(indexPath),
        "caseCount": len(cases),
        "cases": [
            {
                "caseId": case.caseId,
                "parentCaseId": case.metadata.get("parentCaseId"),
                "motorId": case.metadata.get("motorId"),
                "configuredHeatSourcePowerW": case.metadata.get("configuredHeatSourcePowerW"),
                "hybridBoundaryInputHeatFlowRateW": case.metadata.get("hybridBoundaryInputHeatFlowRateW"),
                "connectionBoundaryHeatFlowRateW": case.metadata.get("connectionBoundaryHeatFlowRateW"),
                "steadyEnergyResidualW": case.metadata.get("steadyEnergyResidualW"),
                "boundarySampleCount": int(case.boundary.weightsM2.size),
            }
            for case in cases
        ],
    }
    (output / "rebuild_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

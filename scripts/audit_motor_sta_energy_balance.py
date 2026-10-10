from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import ZipFile

import h5py
import numpy as np

from mor.nn.data.motorFemMetadata import (
    normalizedComponentName,
    regionName,
    resolveFemSystem,
)
from mor.nn.data.motorFemGeometry import (
    boundaryNodeGeometry,
)
from mor.nn.data.motorFemBoundary import (
    surfacePatchSamples,
)
from mor.nn.data.motorFemIO import (
    fixedHeatSourceValue,
)
from mor.nn.data.motorStaDataset import inspectMotorStaArchive


def _extract(archive: ZipFile, member: str, destination: Path) -> None:
    info = archive.getinfo(member)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(info) as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)


def _connectivity(dataset: Any, width: int, name: str) -> np.ndarray:
    values = np.asarray(dataset, dtype=np.int64).reshape(-1)
    if values.size == 0 or values.size % width:
        raise ValueError(f"{name} is not a non-empty width-{width} connectivity array")
    return values.reshape(-1, width)


def _selected_tetrahedron_volume(
    nodes: np.ndarray,
    tetrahedra: np.ndarray,
    elements: np.ndarray,
    *,
    chunk_size: int = 250_000,
) -> float:
    total = 0.0
    for start in range(0, elements.size, chunk_size):
        selection = tetrahedra[elements[start : start + chunk_size]]
        points = nodes[selection]
        six_volumes = np.einsum(
            "ij,ij->i",
            np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]),
            points[:, 3] - points[:, 0],
        )
        total += float(np.abs(six_volumes).sum() / 6.0)
    return total


def _configured_source_power(mesh: h5py.File, region: dict[str, Any]) -> float:
    power = 0.0
    nodes: np.ndarray | None = None
    tetrahedra: np.ndarray | None = None
    for volume_patch in region.get("volumePatches", {}).values():
        for child_uuid, group in volume_patch.get("children", {}).items():
            source = fixedHeatSourceValue(group)
            if source is None:
                continue
            input_variable, value = source
            if input_variable == "power":
                power += value
                continue
            if input_variable not in ("volumetricHeatSource", "heatSource"):
                raise NotImplementedError(
                    f"unsupported heat-source input variable: {input_variable!r}"
                )
            if nodes is None:
                nodes = np.asarray(mesh["nodes"], dtype=np.float64).reshape(-1, 3)
                tetrahedra = _connectivity(
                    mesh["connectivityVolume"], 4, "connectivityVolume"
                )
            dataset_path = f"volumes/{child_uuid}"
            if dataset_path not in mesh:
                raise KeyError(f"volume group is missing from mesh: {dataset_path}")
            elements = np.asarray(mesh[dataset_path], dtype=np.int64).reshape(-1)
            if (
                elements.size == 0
                or np.any(elements < 0)
                or np.any(elements >= tetrahedra.shape[0])
            ):
                raise ValueError(f"volume group {child_uuid!r} has invalid elements")
            power += value * _selected_tetrahedron_volume(nodes, tetrahedra, elements)
    return float(power)


def _result_paths(schema: str, system_uuid: str, region_uuid: str) -> tuple[str, str]:
    if schema == "nestedFem3DThermal":
        root = f"thermalComponents/thermal3dFem/{system_uuid}/{region_uuid}"
        return f"{root}/3dResults/results", f"{root}/overviewResults/results/heatFlowRate"
    root = f"thermalComponents/femRegion/{region_uuid}"
    return f"{root}/results", f"{root}/overviewResults/results/heatFlowRate"


def _decoded(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)


def _connection_endpoint(value: Any) -> tuple[str, str]:
    decoded = _decoded(value)
    component, _, remainder = decoded.partition("@")
    patch = remainder.split("&", 1)[0] if remainder else ""
    return component, patch


def _connection_net_heat_flow(
    result: h5py.File,
) -> tuple[dict[str, float], dict[tuple[str, str], float]]:
    """Return signed heat entering each component from the solver connection table."""
    path = "connections/heatFlowRate"
    if path not in result:
        return {}, {}
    values = np.asarray(result[path]).reshape(-1)
    if values.size % 3:
        raise ValueError("connections/heatFlowRate is not a flattened endpoint pair plus value table")
    totals: dict[str, float] = {}
    ports: dict[tuple[str, str], float] = {}
    for left_raw, right_raw, power_raw in values.reshape(-1, 3):
        left, left_patch = _connection_endpoint(left_raw)
        right, right_patch = _connection_endpoint(right_raw)
        power = float(_decoded(power_raw))
        if not math.isfinite(power):
            raise ValueError("connections/heatFlowRate contains a non-finite value")
        totals[left] = totals.get(left, 0.0) - power
        totals[right] = totals.get(right, 0.0) + power
        ports[(left, left_patch)] = ports.get((left, left_patch), 0.0) - power
        ports[(right, right_patch)] = ports.get((right, right_patch), 0.0) + power
    return totals, ports


def _component_record(
    mesh: h5py.File,
    result: h5py.File,
    *,
    source: Path,
    system_uuid: str,
    schema: str,
    system: dict[str, Any],
    region_uuid: str,
    region: dict[str, Any],
    connection_net_heat_flow: dict[str, float],
    connection_net_heat_flow_by_port: dict[tuple[str, str], float],
    absolute_tolerance_w: float,
    strict_relative_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    result_base, overview_path = _result_paths(schema, system_uuid, region_uuid)
    heat_flow_path = f"{result_base}/heatFlowRate"
    if heat_flow_path not in result:
        raise KeyError(f"result is missing nodal heat flow: {heat_flow_path}")

    nodes = np.asarray(mesh["nodes"], dtype=np.float64).reshape(-1, 3)
    surface = _connectivity(mesh["connectivitySurface"], 3, "connectivitySurface")
    boundary_nodes, _, boundary_node_areas = boundaryNodeGeometry(
        nodes, surface, nodes.mean(axis=0)
    )
    nodal_heat_flow = np.asarray(result[heat_flow_path], dtype=np.float64).reshape(-1)
    node_count = int(nodes.shape[0])
    if nodal_heat_flow.size != node_count:
        raise ValueError(
            f"nodal heat-flow count {nodal_heat_flow.size} does not match node count {node_count}"
        )
    boundary_values = nodal_heat_flow[boundary_nodes]
    raw_boundary_power = float(boundary_values.sum())
    boundary_throughput = float(np.abs(boundary_values).sum())
    patch_source_nodes, _, patch_weights, patch_ids, patch_metadata = surfacePatchSamples(
        mesh, region, nodes, surface, nodes.mean(axis=0)
    )
    nodal_areas = np.zeros(node_count, dtype=np.float64)
    nodal_areas[boundary_nodes] = boundary_node_areas
    nodal_heat_flux = nodal_heat_flow / np.where(nodal_areas > 0.0, nodal_areas, 1.0)
    patch_powers: dict[str, float] = {}
    raw_connected_power = 0.0
    scale_candidates = []
    for patch in patch_metadata:
        mask = patch_ids == patch["patchId"]
        patch_power = float(
            np.dot(nodal_heat_flux[patch_source_nodes[mask]], patch_weights[mask])
        )
        patch_uuid = str(patch["patchUuid"])
        patch_powers[patch_uuid] = patch_power
        connection_patch_power = connection_net_heat_flow_by_port.get(
            (region_uuid, patch_uuid)
        )
        if connection_patch_power is None:
            continue
        raw_connected_power += patch_power
        if abs(patch_power) > 1e-9 and abs(connection_patch_power) > 1e-9:
            candidate = connection_patch_power / patch_power
            if math.isfinite(candidate) and 0.1 <= candidate <= 10.0:
                scale_candidates.append(candidate)
    raw_local_power = raw_boundary_power - raw_connected_power
    overview_power = (
        float(np.asarray(result[overview_path], dtype=np.float64).sum())
        if overview_path in result
        else None
    )
    connection_power = connection_net_heat_flow.get(region_uuid)
    connection_scale = (
        float(np.median(np.asarray(scale_candidates, dtype=np.float64)))
        if scale_candidates
        else None
    )
    if connection_power is not None and connection_scale is not None:
        boundary_power = connection_power + connection_scale * raw_local_power
        boundary_power_source = "connectionCalibratedNodalBoundary"
    elif connection_power is not None:
        boundary_power = connection_power
        boundary_power_source = "connectionTable"
    elif overview_power is not None:
        boundary_power = -overview_power
        boundary_power_source = "overview"
    else:
        boundary_power = raw_boundary_power
        boundary_power_source = "boundaryNodalSum"
    source_power = _configured_source_power(mesh, region)
    residual = source_power + boundary_power
    scale = max(abs(source_power), boundary_throughput, 1.0)
    relative_residual = abs(residual) / scale
    strict_limit = max(absolute_tolerance_w, strict_relative_tolerance * scale)
    limit = max(absolute_tolerance_w, relative_tolerance * scale)
    status = "pass" if abs(residual) <= strict_limit else "warning" if abs(residual) <= limit else "fail"

    return {
        "sourceSta": str(source),
        "sourceFile": source.name,
        "component": normalizedComponentName(regionName(region)),
        "regionUuid": region_uuid,
        "configuredHeatSourcePowerW": source_power,
        "rawNodalBoundaryHeatFlowRateW": raw_boundary_power,
        "overviewBoundaryHeatFlowRateW": overview_power,
        "connectionNetHeatFlowRateW": connection_power,
        "rawConnectedBoundaryHeatFlowRateW": raw_connected_power,
        "rawLocalBoundaryHeatFlowRateW": raw_local_power,
        "connectionDerivedBoundaryScale": connection_scale,
        "connectionScaleSampleCount": len(scale_candidates),
        "auditedBoundaryHeatFlowRateW": boundary_power,
        "boundaryAbsoluteNodalThroughputW": boundary_throughput,
        "energyResidualW": residual,
        "energyRelativeResidual": relative_residual,
        "strictLimitW": strict_limit,
        "limitW": limit,
        "energyBalanceStatus": status,
        "boundaryPowerSource": boundary_power_source,
    }


def _audit_archive(
    source: Path,
    *,
    absolute_tolerance_w: float,
    strict_relative_tolerance: float,
    relative_tolerance: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        info = inspectMotorStaArchive(source)
    except Exception as error:
        return (
            {
                "sourceSta": str(source),
                "sourceFile": source.name,
                "status": "unreadable",
                "error": f"{type(error).__name__}: {error}",
            },
            [],
        )
    if info.analysisType != "steady":
        return (
            {
                "sourceSta": str(source),
                "sourceFile": source.name,
                "analysisType": info.analysisType,
                "resultTime": info.resultTime,
                "resultFrameCount": len(info.availableResultCoordinates),
                "componentCount": len(info.componentNames),
                "status": "transient_requires_storage_term",
            },
            [],
        )

    records: list[dict[str, Any]] = []
    component_errors: list[dict[str, str]] = []
    with ZipFile(source) as archive:
        control = json.loads(archive.read("controlData.json").decode("utf-8-sig"))
        system_uuid, system, schema = resolveFemSystem(control)
        regions = system.get("femRegions", {})
        with tempfile.TemporaryDirectory(prefix="sta-energy-audit-") as directory:
            temporary = Path(directory)
            result_path = temporary / "result.h5"
            mesh_path = temporary / "mesh.samesh"
            _extract(archive, info.resultMember, result_path)
            with h5py.File(result_path, "r") as result:
                (
                    connection_net_heat_flow,
                    connection_net_heat_flow_by_port,
                ) = _connection_net_heat_flow(result)
                for region_uuid, region in regions.items():
                    mesh_member = PurePosixPath("mesh") / f"{region['meshUuid']}.samesh"
                    try:
                        _extract(archive, mesh_member.as_posix(), mesh_path)
                        with h5py.File(mesh_path, "r") as mesh:
                            records.append(
                                _component_record(
                                    mesh,
                                    result,
                                    source=source,
                                    system_uuid=system_uuid,
                                    schema=schema,
                                    system=system,
                                    region_uuid=str(region_uuid),
                                    region=region,
                                    connection_net_heat_flow=connection_net_heat_flow,
                                    connection_net_heat_flow_by_port=connection_net_heat_flow_by_port,
                                    absolute_tolerance_w=absolute_tolerance_w,
                                    strict_relative_tolerance=strict_relative_tolerance,
                                    relative_tolerance=relative_tolerance,
                                )
                            )
                    except Exception as error:
                        component_errors.append(
                            {
                                "component": normalizedComponentName(regionName(region)),
                                "error": f"{type(error).__name__}: {error}",
                            }
                        )
                    finally:
                        mesh_path.unlink(missing_ok=True)

    source_power = float(sum(record["configuredHeatSourcePowerW"] for record in records))
    boundary_power = float(sum(record["auditedBoundaryHeatFlowRateW"] for record in records))
    residual = source_power + boundary_power
    statuses = [record["energyBalanceStatus"] for record in records]
    status = "component_error" if component_errors else "fail" if "fail" in statuses else "warning" if "warning" in statuses else "pass"
    thermal = control.get("rootSystem", {}).get("thermalComponents", {})
    summary = {
        "sourceSta": str(source),
        "sourceFile": source.name,
        "analysisType": info.analysisType,
        "resultMember": info.resultMember,
        "resultTime": info.resultTime,
        "resultFrameCount": len(info.availableResultCoordinates),
        "componentCount": len(regions),
        "auditedComponentCount": len(records),
        "passComponentCount": statuses.count("pass"),
        "warningComponentCount": statuses.count("warning"),
        "failComponentCount": statuses.count("fail"),
        "configuredHeatSourcePowerW": source_power,
        "auditedBoundaryHeatFlowRateW": boundary_power,
        "energyResidualW": residual,
        "femNetworkInterfaceCount": len(thermal.get("femNetworkInterfaces", {})),
        "femFemInterfaceCount": len(thermal.get("femFemTccInterfaces", {})),
        "status": status,
        "componentErrors": component_errors,
    }
    return summary, records


def _json_ready(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit FEM component energy balance directly from original STA archives."
    )
    parser.add_argument("source", type=Path, help="STA file or directory to scan recursively")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--absolute-tolerance-w", type=float, default=1.0)
    parser.add_argument("--strict-relative-tolerance", type=float, default=1e-3)
    parser.add_argument("--relative-tolerance", type=float, default=1e-2)
    arguments = parser.parse_args()

    sources = (
        [arguments.source]
        if arguments.source.is_file()
        else sorted(arguments.source.rglob("*.sta"))
    )
    case_records: list[dict[str, Any]] = []
    component_records: list[dict[str, Any]] = []
    for position, source in enumerate(sources, start=1):
        print(f"[{position}/{len(sources)}] {source}", flush=True)
        summary, components = _audit_archive(
            source,
            absolute_tolerance_w=arguments.absolute_tolerance_w,
            strict_relative_tolerance=arguments.strict_relative_tolerance,
            relative_tolerance=arguments.relative_tolerance,
        )
        case_records.append(summary)
        component_records.extend(components)

    output = arguments.output
    output.parent.mkdir(parents=True, exist_ok=True)
    json_path = output.with_suffix(".json")
    csv_path = output.with_suffix(".csv")
    case_csv_path = output.with_name(f"{output.name}_cases").with_suffix(".csv")
    payload = {
        "formatVersion": "original-sta-energy-audit-v1",
        "source": str(arguments.source),
        "policy": {
            "steadyEquation": "configuredHeatSourcePowerW + auditedBoundaryHeatFlowRateW = 0",
            "absoluteToleranceW": arguments.absolute_tolerance_w,
            "strictRelativeTolerance": arguments.strict_relative_tolerance,
            "relativeTolerance": arguments.relative_tolerance,
            "transientEquation": "Qsource + Qboundary = dU/dt; not classified without storage term",
        },
        "cases": case_records,
        "components": component_records,
    }
    json_path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")

    if component_records:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=list(component_records[0]))
            writer.writeheader()
            writer.writerows(component_records)
    if case_records:
        fields = sorted({key for record in case_records for key in record if key != "componentErrors"})
        with case_csv_path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(case_records)
    print(json.dumps({"json": str(json_path), "componentsCsv": str(csv_path), "casesCsv": str(case_csv_path)}, indent=2))


if __name__ == "__main__":
    main()

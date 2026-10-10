from __future__ import annotations

import re
from typing import Any

import numpy as np


def regionName(region: dict[str, Any]) -> str:
    return str(region.get('name', '')).strip()


def activeFemRegions(femSystem: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Legacy regions without isActive are active; disabled regions have no results."""
    return {
        uid: region for uid, region in femSystem.get('femRegions', {}).items()
        if region.get('isActive', True) is not False
    }


_COMPONENT_INSTANCE_ALIASES = {
    '转子铁芯': 'rotor_core',
    '端板': 'end_plate',
    '端板1': 'end_plate_1',
    '端板2': 'end_plate_2',
    '轴承外圈': 'bearing_outer_ring',
    '轴承内圈': 'bearing_inner_ring',
    '轴承支架': 'bearing_support',
    '转轴': 'shaft',
    '球': 'bearing_ball',
    '绕组槽部': 'winding_slot',
    '绕组皇冠端': 'winding_crown_end',
    '绕组焊接端': 'winding_weld_end',
    '定子铁芯': 'stator',
    '轴': 'shaft',
    '隔磁板': 'magnetic_isolation_plate',
    '隔磁板1': 'magnetic_isolation_plate_1',
    '隔磁板2': 'magnetic_isolation_plate_2',
    '油箱': 'oil_tank',
    '机箱': 'housing',
    '轴承': 'bearing',
    '磁钢': 'magnets',
    '端盖': 'end_cap',
    '阻尼条': 'damper_bar',
    '行星轮': 'planet_gear',
    '支架': 'carrier',
    '支架轴承': 'carrier_bearing',
    '齿圈': 'ring_gear',
    '外壳': 'housing',
}


_NUMBERED_COMPONENT_INSTANCE_ALIASES = {
    '端板': 'end_plate',
    '轴承外圈': 'bearing_outer_ring',
    '隔磁板': 'magnetic_isolation_plate',
    '轴承': 'bearing',
    '端盖': 'end_cap',
    '阻尼条': 'damper_bar',
    '行星轮': 'planet_gear',
    '支架轴承': 'carrier_bearing',
}


def normalizedComponentName(name: str) -> str:
    normalized = str(name).strip()
    normalized = re.sub(r'\.(?:stp|step|igs|iges)$', '', normalized, flags=re.IGNORECASE)
    if normalized.lower().endswith('_mm'):
        normalized = normalized[:-3]
    compact = re.sub(r'\s+', '', normalized)
    if compact in _COMPONENT_INSTANCE_ALIASES:
        return _COMPONENT_INSTANCE_ALIASES[compact]
    for prefix, alias in sorted(
        _NUMBERED_COMPONENT_INSTANCE_ALIASES.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        match = re.fullmatch(re.escape(prefix) + r'([0-9]+)', compact)
        if match is not None:
            return f'{alias}_{match.group(1)}'
    normalized = re.sub(r'[^A-Za-z0-9_-]+', '_', normalized).strip('_')
    if not normalized:
        codepoints = '_'.join(f'u{ord(character):x}' for character in compact)
        if not codepoints:
            raise ValueError('component name cannot be normalized to an empty identifier')
        return f'component_{codepoints}'
    return normalized.lower()


def componentTypeName(componentName: str) -> str:
    """Return the semantic class while keeping the FEM region name as its instance ID."""
    normalized = normalizedComponentName(componentName)
    if re.fullmatch(r'disk(?:[_-]?\d+)?', normalized):
        return 'magnetic_isolation_plate'
    if normalized.startswith('magnetic_isolation_plate'):
        return 'magnetic_isolation_plate'
    if 'shaft' in normalized or re.fullmatch(r'zhou[_-]?\d+', normalized):
        return 'shaft'
    if re.fullmatch(r'(?:cap(?:s|[_-]?\d+)?|end[_-]?cap(?:[_-]?\d+)?)', normalized):
        return 'end_cap'
    if 'insulating_paper' in normalized or normalized.startswith('insulation'):
        return 'insulation'
    if 'bearing' in normalized or re.fullmatch(r'b[_-]?\d+', normalized):
        return 'bearing'
    if 'magnet' in normalized:
        return 'magnets'
    if normalized.startswith('stator') or normalized.startswith('st_fe'):
        return 'stator'
    if (
        normalized in {'shell', 'pan'}
        or 'shell' in normalized
        or normalized.startswith('housing')
    ):
        return 'housing'
    if 'coil' in normalized or 'winding' in normalized:
        return 'winding'
    if (
        normalized == 'rotor'
        or normalized.endswith('_rotor')
        or normalized.startswith('rotor_fe')
        or normalized.startswith('rt_fe')
    ):
        return 'rotor_core'
    if 'sleeve' in normalized:
        return 'sleeve'
    if normalized.startswith('end_plate'):
        return 'end_plate'
    if normalized == 'oil_tank':
        return 'oil_tank'
    if normalized.startswith('damper_bar'):
        return 'damper_bar'
    if normalized.startswith('planet_gear'):
        return 'planet_gear'
    if normalized.startswith('ring_gear'):
        return 'ring_gear'
    if normalized == 'carrier':
        return 'carrier'
    if 'hub' in normalized:
        return 'wheel_hub'
    if 'sprayring' in normalized or 'spray_ring' in normalized:
        return 'cooling_ring'
    if 'rotorplate' in normalized or 'rotor_plate' in normalized:
        return 'end_plate'
    if normalized == 'fem':
        return 'gear'
    if 'gear' in normalized or re.fullmatch(r'g[_-]?\d+(?:[_-]\d+)?', normalized):
        return 'gear'
    return normalized


def resolveFemSystem(control: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
    candidates = []

    def visit(system: dict[str, Any]) -> None:
        thermal = system.get('thermalComponents', {})
        for femSystemUuid, femSystem in thermal.get('fem3DThermals', {}).items():
            if femSystem.get('femRegions'):
                candidates.append(
                    (
                        str(femSystemUuid),
                        {
                            **femSystem,
                            'femNetworkInterfaces': thermal.get(
                                'femNetworkInterfaces', {}
                            ),
                        },
                        'nestedFem3DThermal',
                    )
                )
        directRegions = thermal.get('femRegions')
        if directRegions:
            candidates.append(
                (
                    str(system.get('uuid', 'rootSystem')),
                    {
                        'femRegions': directRegions,
                        'tccInterfaces': thermal.get('femFemTccInterfaces', {}),
                        'femNetworkInterfaces': thermal.get('femNetworkInterfaces', {}),
                    },
                    'directFemRegions',
                )
            )
        for child in system.get('children', {}).values():
            if isinstance(child, dict):
                visit(child)

    visit(control['rootSystem'])
    if not candidates:
        raise KeyError('control data contains neither fem3DThermals nor direct femRegions')
    if len(candidates) != 1:
        raise ValueError(f'expected exactly one 3D FEM system, found {len(candidates)}')
    return candidates[0]


def findRegion(femSystem: dict[str, Any], component: str) -> tuple[str, dict[str, Any]]:
    requested = str(component).strip().lower()
    matches = []
    for regionUuid, region in femSystem.get('femRegions', {}).items():
        name = regionName(region)
        aliases = {str(regionUuid).lower(), name.lower(), normalizedComponentName(name)}
        if requested in aliases:
            matches.append((str(regionUuid), region))
    if len(matches) != 1:
        available = sorted(
            regionName(region) for region in femSystem.get('femRegions', {}).values()
        )
        raise KeyError(
            f'component {component!r} did not match exactly one FEM region; '
            f'available={available}'
        )
    if matches[0][1].get('isActive', True) is False:
        raise ValueError(f'FEM component {component!r} is inactive and has no solved results')
    return matches[0]


def thermalAnalysisMetadata(
    control: dict[str, Any],
    *,
    snapshotTimeS: float | None = None,
) -> dict[str, Any]:
    timeDomain = control.get('solverSetting', {}).get('timeDomain', {})
    rawTransient = timeDomain.get('isTransient')
    if not isinstance(rawTransient, (bool, np.bool_)):
        return {
            'analysisType': 'unknown',
            'isTransient': None,
            'temporalSampleType': 'unknownSnapshot',
            'timeSeriesStored': False,
        }
    isTransient = bool(rawTransient)
    metadata: dict[str, Any] = {
        'analysisType': 'transient' if isTransient else 'steady',
        'isTransient': isTransient,
        'temporalSampleType': 'transientSnapshot' if isTransient else 'steadySolution',
        'timeSeriesStored': False,
    }
    if not isTransient:
        return metadata
    for sourceName, targetName in (
        ('deltaTime', 'sourceTimeStepS'),
        ('endTime', 'sourceEndTimeS'),
    ):
        rawValue = timeDomain.get(sourceName)
        if rawValue is None:
            continue
        value = float(rawValue)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f'transient solver {sourceName} must be positive and finite')
        metadata[targetName] = value
    if snapshotTimeS is not None:
        value = float(snapshotTimeS)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError('transient snapshotTimeS must be non-negative and finite')
        metadata['snapshotTimeS'] = value
    return metadata

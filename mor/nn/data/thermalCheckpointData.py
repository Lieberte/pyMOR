from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .thermalAssemblyContext import (
    THERMAL_ASSEMBLY_CONTEXT_V1,
    enrichThermalAssemblyCases,
    thermalAssemblyBoundaryFeatureNames,
    thermalAssemblyContextFeatureNames,
)
from .thermalComponentCase import thermalComponentCase, thermalComponentFeatureSchema
from .thermalDimensionless import (
    STEADY_THERMAL_DIMENSIONLESS_V2,
    steadyThermalDimensionlessCase,
    steadyThermalDimensionlessOperatorCase,
)


DEFAULT_THERMAL_OPERATING_SCALES = {
    'speedRpm': 25000.0,
    'absoluteTemperatureAnchor': 1.0,
}


@dataclass(frozen=True)
class thermalCheckpointDataContract:
    """Checkpoint-owned preprocessing contract for steady thermal operators."""

    schema: thermalComponentFeatureSchema
    operatingScales: Mapping[str, float]
    minimumTemperatureScaleK: float
    featureMode: str
    assemblyContextEnabled: bool
    componentTypes: tuple[str, ...]
    contextFeatureNames: tuple[str, ...]
    expectedVolumeFeatureDim: int | None = None
    expectedBoundaryFeatureDim: int | None = None
    expectedGlobalFeatureDim: int | None = None

    @classmethod
    def fromCheckpoint(
        cls,
        checkpoint: Mapping[str, Any],
    ) -> 'thermalCheckpointDataContract':
        metadata = dict(checkpoint.get('metadata', {}) or {})
        schemaPayload = dict(metadata.get('schema', {}) or {})
        assemblyPayload = dict(metadata.get('assemblyContext', {}) or {})
        enabled = bool(assemblyPayload.get('enabled', False))
        componentTypes = tuple(
            str(value).strip()
            for value in assemblyPayload.get('componentTypes', ())
        )
        if enabled and not componentTypes:
            raise ValueError(
                'assembly-context checkpoint does not define componentTypes'
            )
        if any(not value for value in componentTypes) or len(set(componentTypes)) != len(
            componentTypes
        ):
            raise ValueError('checkpoint componentTypes must be unique non-empty strings')

        generatedContextNames = (
            thermalAssemblyContextFeatureNames(componentTypes) if enabled else ()
        )
        storedContextNames = tuple(
            str(value) for value in assemblyPayload.get('featureNames', ())
        )
        if storedContextNames and storedContextNames != generatedContextNames:
            raise ValueError(
                'checkpoint assembly context feature order is incompatible with '
                'the current preprocessing implementation'
            )
        contextFeatureNames = storedContextNames or generatedContextNames
        defaultOperatingNames = (
            'speedRpm', *contextFeatureNames, 'absoluteTemperatureAnchor'
        )
        schema = thermalComponentFeatureSchema(
            operatingFeatureNames=tuple(
                schemaPayload.get('operatingFeatureNames', defaultOperatingNames)
            ),
            boundaryTypeCount=int(schemaPayload.get('boundaryTypeCount', 4)),
            temperatureBoundaryTypeIds=tuple(
                schemaPayload.get('temperatureBoundaryTypeIds', (1, 2))
            ),
            fixedTemperatureBoundaryTypeIds=tuple(
                schemaPayload.get('fixedTemperatureBoundaryTypeIds', (2,))
            ),
        )
        if enabled:
            required = frozenset((*contextFeatureNames, 'absoluteTemperatureAnchor'))
            missing = required.difference(schema.operatingFeatureNames)
            if missing:
                raise ValueError(
                    'checkpoint schema omits assembly context features: '
                    f'{sorted(missing)!r}'
                )

        operatingScales = {
            **DEFAULT_THERMAL_OPERATING_SCALES,
            **{
                str(name): float(value)
                for name, value in dict(metadata.get('operatingScales', {}) or {}).items()
            },
        }
        globalConfig = dict(checkpoint.get('global', {}).get('model', {}) or {})
        return cls(
            schema=schema,
            operatingScales=operatingScales,
            minimumTemperatureScaleK=float(
                metadata.get('minimumTemperatureScaleK', 1.0)
            ),
            featureMode=str(
                metadata.get('featureMode', STEADY_THERMAL_DIMENSIONLESS_V2)
            ),
            assemblyContextEnabled=enabled,
            componentTypes=componentTypes,
            contextFeatureNames=contextFeatureNames,
            expectedVolumeFeatureDim=globalConfig.get('volumeFeatureDim'),
            expectedBoundaryFeatureDim=globalConfig.get('boundaryFeatureDim'),
            expectedGlobalFeatureDim=globalConfig.get('globalFeatureDim'),
        )

    def prepareCases(
        self,
        cases: Sequence[thermalComponentCase],
    ) -> tuple[thermalComponentCase, ...]:
        resolved = tuple(cases)
        if self.assemblyContextEnabled:
            existing = tuple(
                dict(case.metadata.get('thermalAssemblyContext', {}) or {})
                for case in resolved
            )
            alreadyPrepared = tuple(bool(value) for value in existing)
            if any(alreadyPrepared):
                if not all(alreadyPrepared):
                    raise ValueError(
                        'input mixes raw and assembly-context-enriched thermal cases'
                    )
                expectedBoundaryNames = thermalAssemblyBoundaryFeatureNames(
                    self.componentTypes
                )
                if any(
                    value.get('version') != THERMAL_ASSEMBLY_CONTEXT_V1
                    or tuple(value.get('boundaryFeatureNames', ()))
                    != expectedBoundaryNames
                    for value in existing
                ):
                    raise ValueError(
                        'input assembly context version or boundary feature order '
                        'does not match the checkpoint preprocessor'
                    )
                if any(
                    tuple(value.get('componentTypes', ())) != self.componentTypes
                    for value in existing
                ):
                    raise ValueError(
                        'input assembly context component vocabulary does not '
                        'match the checkpoint'
                    )
                return resolved
            return enrichThermalAssemblyCases(
                resolved,
                componentTypes=self.componentTypes,
            )
        return resolved

    def transformCases(
        self,
        cases: Sequence[thermalComponentCase],
    ) -> tuple[steadyThermalDimensionlessCase, ...]:
        transformed = tuple(
            steadyThermalDimensionlessOperatorCase(
                case,
                schema=self.schema,
                operatingScales=self.operatingScales,
                minimumTemperatureScaleK=self.minimumTemperatureScaleK,
                featureMode=self.featureMode,
            )
            for case in cases
        )
        if transformed:
            self.validateOperatorCase(transformed[0])
        return transformed

    def prepareAndTransform(
        self,
        cases: Sequence[thermalComponentCase],
    ) -> tuple[
        tuple[thermalComponentCase, ...],
        tuple[steadyThermalDimensionlessCase, ...],
    ]:
        prepared = self.prepareCases(cases)
        return prepared, self.transformCases(prepared)

    def validateOperatorCase(
        self,
        value: steadyThermalDimensionlessCase,
    ) -> None:
        operatorCase = value.operatorCase
        dimensions = {
            'volumeFeatureDim': (
                operatorCase.volumeFeatureDim,
                self.expectedVolumeFeatureDim,
            ),
            'boundaryFeatureDim': (
                operatorCase.boundaryFeatureDim,
                self.expectedBoundaryFeatureDim,
            ),
            'globalFeatureDim': (
                operatorCase.globalFeatureDim,
                self.expectedGlobalFeatureDim,
            ),
        }
        mismatches = {
            name: {'actual': actual, 'expected': int(expected)}
            for name, (actual, expected) in dimensions.items()
            if expected is not None and actual != int(expected)
        }
        if mismatches:
            raise ValueError(
                f'thermal checkpoint input feature dimensions do not match: {mismatches}'
            )

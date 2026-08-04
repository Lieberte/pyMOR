from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mor.models import (
    thermalLoadCase,
    thermalLinearSolverOptions,
    thermalMaterialLibrary,
    thermalModel,
    thermalNonlinearOptions,
)

from .meshIr import meshIr
from .thermalAssembly import _tetraBlocks, assembleMeshThermalOperators
from .thermalFieldMapping import (
    mapBoundaryFieldByCoordinates as _mapBoundaryFieldByCoordinates,
    mapBoundaryFieldByNodeIds as _mapBoundaryFieldByNodeIds,
)


class thermalProblem:
    def __init__(
        self,
        mesh: meshIr,
        *,
        materials: thermalMaterialLibrary | Mapping[str, Mapping[str, Any]],
        regionMaterials: Mapping[str, str],
        loadCase: thermalLoadCase | Mapping[str, Any] | None = None,
        capacityMode: str = 'consistent',
        referenceTemperature: float = 293.15,
        nonlinearOptions: thermalNonlinearOptions | Mapping[str, Any] | None = None,
        linearSolverOptions: thermalLinearSolverOptions | Mapping[str, Any] | None = None,
        lengthUnit: str = 'm',
        lengthScaleToMeters: float | None = None,
        backendName: str = 'scipy',
    ):
        self.mesh = mesh
        self.materials = (
            materials
            if isinstance(materials, thermalMaterialLibrary)
            else thermalMaterialLibrary.fromDict(materials)
        )
        self.regionMaterials = dict(regionMaterials)
        self.loadCase = (
            loadCase
            if isinstance(loadCase, thermalLoadCase)
            else thermalLoadCase.fromDict(loadCase or {})
        )
        unitScales = {'m': 1.0, 'cm': 1e-2, 'mm': 1e-3}
        normalizedUnit = str(lengthUnit).lower()
        if normalizedUnit not in unitScales:
            raise ValueError("lengthUnit must be 'm', 'cm', or 'mm'")
        if lengthScaleToMeters is None:
            scale = unitScales[normalizedUnit]
        else:
            scale = float(lengthScaleToMeters)
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError('lengthScaleToMeters must be positive and finite')
            if normalizedUnit != 'm' and not np.isclose(scale, unitScales[normalizedUnit]):
                raise ValueError('lengthUnit and lengthScaleToMeters specify inconsistent scales')
        self.lengthUnit = normalizedUnit
        self.lengthScaleToMeters = scale
        self.analysisMesh = self._scaledMesh(scale)
        if capacityMode not in ('consistent', 'lumped'):
            raise ValueError("capacityMode must be 'consistent' or 'lumped'")
        self.capacityMode = capacityMode
        self.referenceTemperature = float(referenceTemperature)
        if not np.isfinite(self.referenceTemperature):
            raise ValueError('referenceTemperature must be finite')
        self.nonlinearOptions = (
            nonlinearOptions
            if isinstance(nonlinearOptions, thermalNonlinearOptions)
            else thermalNonlinearOptions.fromDict(nonlinearOptions or {})
        )
        self.linearSolverOptions = (
            linearSolverOptions
            if isinstance(linearSolverOptions, thermalLinearSolverOptions)
            else thermalLinearSolverOptions.fromDict(linearSolverOptions or {})
        )
        self.backendName = backendName
        self.materials.resolveRegionProperties(self.regionMaterials)
        self._validateBoundaryClassification()

    def _scaledMesh(self, scale: float) -> meshIr:
        return meshIr(
            nodes=np.asarray(self.mesh.nodes, dtype=float) * scale,
            boundaryNodes={
                name: np.asarray(values, dtype=float) * scale
                for name, values in self.mesh.boundaryNodes.items()
            },
            cells=self.mesh.cells,
            boundaryFaces=self.mesh.boundaryFaces,
            boundaryNodeIndices=self.mesh.boundaryNodeIndices,
            pointData=self.mesh.pointData,
            cellData=self.mesh.cellData,
            cellSets=self.mesh.cellSets,
        )

    def _validateBoundaryClassification(self) -> None:
        contactBoundaries = {
            boundary
            for interface in self.loadCase.contacts.values()
            for boundary in (interface.primaryBoundary, interface.secondaryBoundary)
        }
        externalFluxBoundaries = set(self.loadCase.convection) | set(self.loadCase.heatFlux)
        overlap = sorted(contactBoundaries & externalFluxBoundaries)
        if overlap:
            raise ValueError(
                'contact boundaries cannot also be external convection/heat-flux boundaries: '
                + ', '.join(overlap)
            )

    def _dirichletConditions(self) -> tuple[np.ndarray, np.ndarray]:
        prescribed: dict[int, float] = {}
        for boundaryName, value in self.loadCase.dirichletTemperatures.items():
            if boundaryName not in self.mesh.boundaryNodeIndices:
                raise KeyError(
                    f'boundaryNodeIndices does not contain {boundaryName!r} for Dirichlet temperature'
                )
            indices = self.mesh.boundaryNodeIndices[boundaryName]
            temperatures = np.asarray(value, dtype=float)
            if temperatures.ndim == 0:
                temperatures = np.full(indices.size, float(temperatures), dtype=float)
            else:
                temperatures = temperatures.reshape(-1)
            if temperatures.size != indices.size:
                raise ValueError(
                    f'dirichletTemperatures[{boundaryName!r}] must be scalar or contain {indices.size} values'
                )
            for nodeIndex, temperature in zip(indices, temperatures):
                node = int(nodeIndex)
                scalarTemperature = float(temperature)
                if node in prescribed and not np.isclose(prescribed[node], scalarTemperature):
                    raise ValueError(f'conflicting Dirichlet temperatures for node {node}')
                prescribed[node] = scalarTemperature
        nodes = np.array(sorted(prescribed), dtype=int)
        values = np.array([prescribed[int(node)] for node in nodes], dtype=float)
        return nodes, values

    def mapBoundaryFieldByNodeIds(self, boundaryName: str, sourceNodeIds: Any, sourceValues: Any, **kwargs):
        return _mapBoundaryFieldByNodeIds(
            self.mesh,
            boundaryName,
            sourceNodeIds,
            sourceValues,
            **kwargs,
        )

    def mapBoundaryFieldByCoordinates(
        self,
        boundaryName: str,
        sourceCoordinates: Any,
        sourceValues: Any,
        *,
        sourceLengthUnit: str = 'm',
        **kwargs,
    ):
        unitScales = {'m': 1.0, 'cm': 1e-2, 'mm': 1e-3}
        normalizedUnit = str(sourceLengthUnit).lower()
        if normalizedUnit not in unitScales:
            raise ValueError("sourceLengthUnit must be 'm', 'cm', or 'mm'")
        return _mapBoundaryFieldByCoordinates(
            self.analysisMesh,
            boundaryName,
            sourceCoordinates,
            sourceValues,
            meshLengthScaleToMeters=1.0,
            sourceLengthScaleToMeters=unitScales[normalizedUnit],
            **kwargs,
        )

    def _materialProperties(self, nodalTemperature: Any = None) -> dict[str, Any]:
        blockIndices, elements = _tetraBlocks(self.mesh)
        blockOffsets: dict[int, int] = {}
        offset = 0
        for blockIndex in blockIndices:
            blockOffsets[blockIndex] = offset
            offset += int(np.asarray(self.mesh.cells[blockIndex][1]).shape[0])
        materialNames = np.empty(elements.shape[0], dtype=object)
        materialNames[:] = None
        for regionName, materialName in self.regionMaterials.items():
            if regionName not in self.mesh.cellSets:
                raise KeyError(f'cellSets does not contain {regionName!r} for material assignment')
            for blockIndex, cellIndices in enumerate(self.mesh.cellSets[regionName]):
                if blockIndex not in blockOffsets:
                    continue
                indices = np.asarray(cellIndices, dtype=int) + blockOffsets[blockIndex]
                if any(materialNames[int(index)] is not None for index in indices):
                    raise ValueError('overlapping cell sets assign material more than once')
                materialNames[indices] = materialName
        missing = sum(value is None for value in materialNames)
        if missing:
            raise ValueError(f'material assignment leaves {missing} tetra elements unassigned')

        if nodalTemperature is None:
            nodeTemperatures = np.full(self.analysisMesh.nodes.shape[0], self.referenceTemperature, dtype=float)
        else:
            nodeTemperatures = np.asarray(nodalTemperature, dtype=float).reshape(-1)
            if nodeTemperatures.size != self.analysisMesh.nodes.shape[0]:
                raise ValueError(
                    f'nodalTemperature must contain {self.analysisMesh.nodes.shape[0]} values'
                )
            if not np.all(np.isfinite(nodeTemperatures)):
                raise ValueError('nodalTemperature must contain finite values')
        elementTemperatures = nodeTemperatures[elements].mean(axis=1)

        evaluated: list[tuple[np.ndarray, Any, np.ndarray, np.ndarray]] = []
        useTensorConductivity = False
        for materialName in self.materials.names:
            indices = np.flatnonzero(materialNames == materialName)
            if not indices.size:
                continue
            material = self.materials[materialName]
            temperatures = elementTemperatures[indices]
            conductivity = np.asarray(material.conductivityAt(temperatures), dtype=float)
            density = np.asarray(material.densityAt(temperatures), dtype=float).reshape(-1)
            specificHeat = np.asarray(material.specificHeatAt(temperatures), dtype=float).reshape(-1)
            if conductivity.shape == (indices.size, 3, 3):
                useTensorConductivity = True
            elif conductivity.shape != (indices.size,):
                raise ValueError('evaluated conductivity must be scalar or 3x3 per element')
            evaluated.append((indices, conductivity, density, specificHeat))

        densityValues = np.empty(elements.shape[0], dtype=float)
        specificHeatValues = np.empty(elements.shape[0], dtype=float)
        conductivityValues = (
            np.empty((elements.shape[0], 3, 3), dtype=float)
            if useTensorConductivity
            else np.empty(elements.shape[0], dtype=float)
        )
        for indices, conductivity, density, specificHeat in evaluated:
            if useTensorConductivity and conductivity.ndim == 1:
                conductivityValues[indices] = conductivity[:, None, None] * np.eye(3)[None, :, :]
            else:
                conductivityValues[indices] = conductivity
            densityValues[indices] = density
            specificHeatValues[indices] = specificHeat
        return {
            'conductivity': conductivityValues,
            'density': densityValues,
            'specificHeat': specificHeatValues,
        }

    def assembleOperators(self, nodalTemperature: Any = None):
        properties = self._materialProperties(nodalTemperature)
        return assembleMeshThermalOperators(
            self.analysisMesh,
            conductivity=properties['conductivity'],
            density=properties['density'],
            specificHeat=properties['specificHeat'],
            heatSource=self.loadCase.heatSources,
            capacityMode=self.capacityMode,
            convection=self.loadCase.convection,
            heatFlux=self.loadCase.heatFlux,
            contacts=self.loadCase.contacts,
            backendName=self.backendName,
        )

    def buildModel(self) -> thermalModel:
        nodes, values = self._dirichletConditions()
        return thermalModel(
            self.assembleOperators(),
            dirichletNodes=nodes,
            dirichletValues=values,
            metadata={
                'operating': dict(self.loadCase.operatingMetadata),
                'temperatureDependentMaterials': self.materials.isTemperatureDependent,
                'lengthUnit': self.lengthUnit,
                'lengthScaleToMeters': self.lengthScaleToMeters,
                'contactInterfaces': tuple(self.loadCase.contacts),
                'materialInterfaces': 'shared-node ideal contact',
            },
            operatorUpdater=self.assembleOperators if self.materials.isTemperatureDependent else None,
            referenceTemperature=self.referenceTemperature,
            nonlinearOptions=self.nonlinearOptions,
            linearSolverOptions=self.linearSolverOptions,
            backendName=self.backendName,
        )

    def toDict(self) -> dict[str, Any]:
        return {
            'materials': self.materials.toDict(),
            'regionMaterials': dict(self.regionMaterials),
            'loadCase': self.loadCase.toDict(),
            'capacityMode': self.capacityMode,
            'referenceTemperature': self.referenceTemperature,
            'nonlinearOptions': self.nonlinearOptions.toDict(),
            'linearSolverOptions': self.linearSolverOptions.toDict(),
            'lengthUnit': self.lengthUnit,
            'lengthScaleToMeters': self.lengthScaleToMeters,
            'backendName': self.backendName,
        }

    @classmethod
    def fromDict(cls, mesh: meshIr, data: Mapping[str, Any]) -> 'thermalProblem':
        return cls(
            mesh,
            materials=data['materials'],
            regionMaterials=data['regionMaterials'],
            loadCase=data.get('loadCase', {}),
            capacityMode=data.get('capacityMode', 'consistent'),
            referenceTemperature=data.get('referenceTemperature', 293.15),
            nonlinearOptions=data.get('nonlinearOptions', {}),
            linearSolverOptions=data.get('linearSolverOptions', {}),
            lengthUnit=data.get('lengthUnit', 'm'),
            lengthScaleToMeters=data.get('lengthScaleToMeters'),
            backendName=data.get('backendName', 'scipy'),
        )

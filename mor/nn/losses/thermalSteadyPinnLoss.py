from __future__ import annotations

from typing import Any

import torch

from mor.nn.registry import nnRegistry

from .thermalOperatorLoss import thermalMeanShapeEnergyLoss


def _inverseSignedLog1p(values: torch.Tensor) -> torch.Tensor:
    return torch.sign(values) * torch.expm1(torch.abs(values))


class thermalSteadyPinnLoss(thermalMeanShapeEnergyLoss):
    """Mean/shape supervision with a strong-form steady heat residual.

    The volume feature layout follows the dimensionless thermal representation:
    coordinates, a flattened 3x3 conductivity tensor, density, heat capacity,
    and signed-log volumetric heat source. Conductivity and source values are
    interpolated to collocation queries and treated as locally constant when
    evaluating ``div(K grad(T)) + q = 0``.
    """

    def __init__(
        self,
        *,
        pdeResidualWeight: float = 0.0,
        pdeResidualMode: str = 'strong',
        pdeCollocationCount: int = 16,
        pdeKernelScale: float = 0.2,
        neumannResidualWeight: float = 0.0,
        robinResidualWeight: float = 0.0,
        dirichletResidualWeight: float = 0.0,
        boundaryCollocationCount: int = 64,
        conductivityFeatureStart: int = 3,
        heatSourceFeatureIndex: int = 14,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lossName = 'thermalSteadyPinnLoss'
        self.pdeResidualWeight = float(pdeResidualWeight)
        self.pdeResidualMode = str(pdeResidualMode).strip().lower()
        self.pdeCollocationCount = int(pdeCollocationCount)
        self.pdeKernelScale = float(pdeKernelScale)
        self.neumannResidualWeight = float(neumannResidualWeight)
        self.robinResidualWeight = float(robinResidualWeight)
        self.dirichletResidualWeight = float(dirichletResidualWeight)
        self.boundaryCollocationCount = int(boundaryCollocationCount)
        self.conductivityFeatureStart = int(conductivityFeatureStart)
        self.heatSourceFeatureIndex = int(heatSourceFeatureIndex)
        if self.pdeResidualWeight < 0.0:
            raise ValueError('pdeResidualWeight must be non-negative')
        if self.pdeResidualMode not in ('strong', 'meshfree'):
            raise ValueError('pdeResidualMode must be strong or meshfree')
        if self.pdeCollocationCount <= 0:
            raise ValueError('pdeCollocationCount must be positive')
        if self.pdeKernelScale <= 0.0:
            raise ValueError('pdeKernelScale must be positive')
        boundaryWeights = (
            self.neumannResidualWeight,
            self.robinResidualWeight,
            self.dirichletResidualWeight,
        )
        if any(value < 0.0 for value in boundaryWeights):
            raise ValueError('boundary residual weights must be non-negative')
        if self.boundaryCollocationCount <= 0:
            raise ValueError('boundaryCollocationCount must be positive')
        if self.conductivityFeatureStart < 0 or self.heatSourceFeatureIndex < 0:
            raise ValueError('PDE feature indices must be non-negative')

    @property
    def requiresCoordinateGradients(self) -> bool:
        return (
            self.pdeResidualWeight > 0.0 and self.pdeResidualMode == 'strong'
        ) or self.requiresBoundaryCoordinateGradients

    @property
    def requiresBoundaryCoordinateGradients(self) -> bool:
        return self.neumannResidualWeight > 0.0 or self.robinResidualWeight > 0.0

    def _materialAtQueries(
        self,
        batch: dict[str, Any],
        queryCoordinates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        volumeFeatures = batch['volumeFeatures']
        queryDim = int(queryCoordinates.shape[-1])
        conductivityEnd = self.conductivityFeatureStart + queryDim**2
        if queryDim != 3:
            raise ValueError('steady thermal PDE residual currently requires 3D queries')
        if conductivityEnd > volumeFeatures.shape[-1]:
            raise ValueError('conductivity tensor features exceed volumeFeatureDim')
        if self.heatSourceFeatureIndex >= volumeFeatures.shape[-1]:
            raise ValueError('heat source feature exceeds volumeFeatureDim')

        volumeCoordinates = volumeFeatures[..., :queryDim]
        distanceSquared = (
            queryCoordinates.detach().unsqueeze(2)
            - volumeCoordinates.unsqueeze(1)
        ).square().sum(dim=-1)
        volumeWeights = batch['volumeWeights'].clamp_min(
            torch.finfo(volumeFeatures.dtype).tiny
        )
        logits = (
            -distanceSquared / (2.0 * self.pdeKernelScale**2)
            + torch.log(volumeWeights).unsqueeze(1)
        )
        logits = logits.masked_fill(
            ~batch['volumeMask'].unsqueeze(1),
            -torch.inf,
        )
        interpolationWeights = torch.softmax(logits, dim=-1)
        conductivity = torch.einsum(
            'bqn,bnf->bqf',
            interpolationWeights,
            volumeFeatures[..., self.conductivityFeatureStart : conductivityEnd],
        ).reshape(*queryCoordinates.shape[:2], queryDim, queryDim)
        source = torch.einsum(
            'bqn,bn->bq',
            interpolationWeights,
            _inverseSignedLog1p(
                volumeFeatures[..., self.heatSourceFeatureIndex]
            ),
        )
        return conductivity.detach(), source.detach()

    def _strongPdeResidual(
        self,
        model: Any,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        count = min(self.pdeCollocationCount, batch['queryCoordinates'].shape[1])
        collocationBatch = dict(batch)
        queryCoordinates = (
            batch['queryCoordinates'][:, :count]
            .detach()
            .clone()
            .requires_grad_(True)
        )
        collocationBatch['queryCoordinates'] = queryCoordinates
        collocationBatch['queryWeights'] = batch['queryWeights'][:, :count]
        collocationBatch['queryMask'] = batch['queryMask'][:, :count]
        prediction = model(collocationBatch).squeeze(-1)
        temperatureGradient = torch.autograd.grad(
            prediction,
            queryCoordinates,
            grad_outputs=torch.ones_like(prediction),
            create_graph=True,
            retain_graph=True,
        )[0]
        hessianRows = []
        for coordinateIndex in range(queryCoordinates.shape[-1]):
            hessianRows.append(
                torch.autograd.grad(
                    temperatureGradient[..., coordinateIndex],
                    queryCoordinates,
                    grad_outputs=torch.ones_like(
                        temperatureGradient[..., coordinateIndex]
                    ),
                    create_graph=True,
                    retain_graph=True,
                )[0]
            )
        hessian = torch.stack(hessianRows, dim=-2)
        conductivity, source = self._materialAtQueries(
            collocationBatch,
            queryCoordinates,
        )
        residual = torch.sum(conductivity * hessian, dim=(-2, -1)) + source
        normalizedResidual = residual / (1.0 + torch.abs(source))
        weights = collocationBatch['queryWeights'] * collocationBatch[
            'queryMask'
        ].to(dtype=collocationBatch['queryWeights'].dtype)
        caseWeight = weights.sum(dim=1)
        validCases = caseWeight > torch.finfo(weights.dtype).eps
        if not torch.any(validCases):
            return prediction.new_zeros(())
        caseLoss = (
            torch.sum(weights * normalizedResidual.square(), dim=1)
            / caseWeight.clamp_min(torch.finfo(weights.dtype).eps)
        )
        return torch.mean(caseLoss[validCases])

    def _meshfreePdeResidual(
        self,
        model: Any,
        batch: dict[str, Any],
        prediction: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate a differentiable local least-squares FEM-like residual.

        This operates on the discrete query cloud and differentiates only with
        respect to model values. It avoids the expensive coordinate Hessian of
        the strong PINN residual and uses the conductivity trace in a stable
        kernel-graph approximation of ``div(K grad(T)) + q``.
        """
        if prediction is None:
            prediction = model(batch)
        count = min(
            batch['queryCoordinates'].shape[1],
            max(batch['queryCoordinates'].shape[-1] + 1, self.pdeCollocationCount),
        )
        temperature = prediction[:, :count].squeeze(-1)
        coordinates = batch['queryCoordinates'][:, :count]
        mask = batch['queryMask'][:, :count]
        queryCount = coordinates.shape[1]
        if queryCount <= coordinates.shape[-1]:
            return temperature.new_zeros(())

        relative = coordinates.unsqueeze(1) - coordinates.unsqueeze(2)
        distanceSquared = relative.square().sum(dim=-1)
        pairMask = mask.unsqueeze(1) & mask.unsqueeze(2)
        diagonal = torch.eye(
            queryCount,
            dtype=torch.bool,
            device=coordinates.device,
        ).unsqueeze(0)
        pairMask &= ~diagonal
        neighborWeights = torch.exp(
            -distanceSquared / (2.0 * self.pdeKernelScale**2)
        ) * pairMask.to(dtype=coordinates.dtype)
        neighborTotal = neighborWeights.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(coordinates.dtype).eps
        )
        neighborWeights = neighborWeights / neighborTotal

        temperatureDifference = temperature.unsqueeze(1) - temperature.unsqueeze(2)
        conductivity, source = self._materialAtQueries(batch, coordinates)
        dimension = coordinates.shape[-1]
        isotropicConductivity = conductivity.diagonal(
            dim1=-2, dim2=-1
        ).mean(dim=-1)
        meanSquaredRadius = torch.sum(
            neighborWeights * distanceSquared,
            dim=-1,
        ).clamp_min(1e-5)
        graphLaplacian = (
            2.0
            * float(dimension)
            * torch.sum(neighborWeights * temperatureDifference, dim=-1)
            / meanSquaredRadius
        )
        divergence = isotropicConductivity * graphLaplacian
        residual = divergence + source
        normalizedResidual = residual / (1.0 + torch.abs(source))
        weights = batch['queryWeights'][:, :count] * mask.to(
            dtype=batch['queryWeights'].dtype
        )
        caseWeight = weights.sum(dim=1)
        enoughNeighbors = torch.all(
            (pairMask.sum(dim=-1) >= coordinates.shape[-1]) | ~mask,
            dim=1,
        )
        validCases = enoughNeighbors & (
            caseWeight > torch.finfo(weights.dtype).eps
        )
        if not torch.any(validCases):
            return temperature.new_zeros(())
        caseLoss = (
            torch.sum(weights * normalizedResidual.square(), dim=1)
            / caseWeight.clamp_min(torch.finfo(weights.dtype).eps)
        )
        return torch.mean(caseLoss[validCases])

    def computePdeResidual(
        self,
        model: Any,
        batch: dict[str, Any],
        prediction: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.pdeResidualMode == 'meshfree':
            return self._meshfreePdeResidual(model, batch, prediction)
        return self._strongPdeResidual(model, batch)

    @staticmethod
    def _weightedBoundaryResidual(
        residual: torch.Tensor,
        active: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        effectiveWeights = weights * active.to(dtype=weights.dtype)
        caseWeight = effectiveWeights.sum(dim=1)
        validCases = caseWeight > torch.finfo(weights.dtype).eps
        if not torch.any(validCases):
            return residual.new_zeros(())
        caseLoss = (
            torch.sum(effectiveWeights * residual.square(), dim=1)
            / caseWeight.clamp_min(torch.finfo(weights.dtype).eps)
        )
        return torch.mean(caseLoss[validCases])

    def _boundaryConditionResiduals(
        self,
        model: Any,
        batch: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        count = min(
            self.boundaryCollocationCount,
            batch['boundaryFeatures'].shape[1],
        )
        boundaryFeatures = batch['boundaryFeatures'][:, :count]
        boundaryMask = batch['boundaryMask'][:, :count]
        boundaryWeights = batch['boundaryWeights'][:, :count]
        boundaryBatch = dict(batch)
        boundaryCoordinates = (
            boundaryFeatures[..., :3].detach().clone().requires_grad_(True)
        )
        boundaryBatch['queryCoordinates'] = boundaryCoordinates
        boundaryBatch['queryWeights'] = boundaryWeights
        boundaryBatch['queryMask'] = boundaryMask
        prediction = model(boundaryBatch).squeeze(-1)

        fixed = torch.zeros_like(boundaryMask)
        for boundaryType in self.fixedTemperatureBoundaryTypeIds:
            fixed |= (
                boundaryFeatures[
                    ...,
                    self.boundaryTypeFeatureStart + boundaryType,
                ]
                > 0.5
            )
        fixed &= boundaryMask
        biot = torch.expm1(
            boundaryFeatures[..., self.boundaryBiotFeatureIndex]
        ).clamp_min(0.0)
        ambient = boundaryFeatures[..., self.boundaryTemperatureFeatureIndex]
        prescribedFlux = _inverseSignedLog1p(
            boundaryFeatures[..., self.boundaryFluxFeatureIndex]
        )
        convection = (biot > 1e-8) & boundaryMask & ~fixed
        neumann = boundaryMask & ~fixed & ~convection

        zero = prediction.new_zeros(())
        neumannLoss = zero
        robinLoss = zero
        if self.requiresBoundaryCoordinateGradients:
            temperatureGradient = torch.autograd.grad(
                prediction,
                boundaryCoordinates,
                grad_outputs=torch.ones_like(prediction),
                create_graph=True,
                retain_graph=True,
            )[0]
            conductivity, _ = self._materialAtQueries(
                boundaryBatch,
                boundaryCoordinates,
            )
            normals = boundaryFeatures[..., 3:6]
            conductiveFlux = torch.einsum(
                'bqi,bqij,bqj->bq',
                normals,
                conductivity,
                temperatureGradient,
            )
            neumannResidual = conductiveFlux - prescribedFlux
            robinResidual = (
                conductiveFlux
                - prescribedFlux
                + biot * (prediction - ambient)
            )
            neumannLoss = self._weightedBoundaryResidual(
                neumannResidual,
                neumann,
                boundaryWeights,
            )
            robinLoss = self._weightedBoundaryResidual(
                robinResidual,
                convection,
                boundaryWeights,
            )
        dirichletLoss = self._weightedBoundaryResidual(
            prediction - ambient,
            fixed,
            boundaryWeights,
        )
        return {
            'neumannResidual': neumannLoss,
            'robinResidual': robinLoss,
            'dirichletResidual': dirichletLoss,
        }

    def lossTerms(
        self,
        model: Any,
        modelOutput: Any,
        batch: dict[str, Any],
        *,
        boundaryPredictions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        dataAndEnergy = super().computeBatch(
            modelOutput,
            batch,
            boundaryPredictions=boundaryPredictions,
        )
        prediction = (
            modelOutput['prediction']
            if isinstance(modelOutput, dict)
            else modelOutput
        )
        pdeResidual = (
            self.computePdeResidual(model, batch, prediction)
            if self.pdeResidualWeight > 0.0
            else dataAndEnergy.new_zeros(())
        )
        boundaryResiduals = (
            self._boundaryConditionResiduals(model, batch)
            if (
                self.requiresBoundaryCoordinateGradients
                or self.dirichletResidualWeight > 0.0
            )
            else {
                'neumannResidual': dataAndEnergy.new_zeros(()),
                'robinResidual': dataAndEnergy.new_zeros(()),
                'dirichletResidual': dataAndEnergy.new_zeros(()),
            }
        )
        terms = {
            'dataAndEnergy': dataAndEnergy,
            'pdeResidual': pdeResidual,
            **boundaryResiduals,
        }
        for name, value in terms.items():
            if not torch.isfinite(value):
                raise FloatingPointError(f'non-finite thermal PINN loss term: {name}')
        return terms

    def computePinnBatch(
        self,
        model: Any,
        modelOutput: Any,
        batch: dict[str, Any],
        *,
        boundaryPredictions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        terms = self.lossTerms(
            model,
            modelOutput,
            batch,
            boundaryPredictions=boundaryPredictions,
        )
        return (
            terms['dataAndEnergy']
            + self.pdeResidualWeight * terms['pdeResidual']
            + self.neumannResidualWeight * terms['neumannResidual']
            + self.robinResidualWeight * terms['robinResidual']
            + self.dirichletResidualWeight * terms['dirichletResidual']
        )


nnRegistry.register('losses', 'thermalSteadyPinnLoss', thermalSteadyPinnLoss)

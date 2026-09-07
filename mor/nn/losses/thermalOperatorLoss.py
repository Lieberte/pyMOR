import torch

from mor.nn.registry import nnRegistry

from .baseLoss import baseLoss


class thermalOperatorLoss(baseLoss):
    def __init__(self, hotspotWeight: float = 0.0, **kwargs):
        super().__init__(lossName='thermalOperatorLoss', **kwargs)
        self.hotspotWeight = float(hotspotWeight)
        if self.hotspotWeight < 0.0:
            raise ValueError('hotspotWeight must be non-negative')

    def compute(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        queryWeights: torch.Tensor,
        queryMask: torch.Tensor,
    ) -> torch.Tensor:
        weights = queryWeights * queryMask.to(dtype=queryWeights.dtype)
        if self.hotspotWeight:
            targetMinimum = targets.masked_fill(
                ~queryMask.unsqueeze(-1), torch.inf
            ).amin(dim=1, keepdim=True)
            targetMaximum = targets.masked_fill(
                ~queryMask.unsqueeze(-1), -torch.inf
            ).amax(dim=1, keepdim=True)
            normalized = (targets - targetMinimum) / (
                targetMaximum - targetMinimum
            ).clamp_min(1e-6)
            weights = weights * (1.0 + self.hotspotWeight * normalized.squeeze(-1))
        squaredError = (predictions - targets).square().squeeze(-1)
        return (squaredError * weights).sum() / weights.sum().clamp_min(1e-12)


def _inverseSignedLog1p(values: torch.Tensor) -> torch.Tensor:
    return torch.sign(values) * torch.expm1(torch.abs(values))


class thermalMeanShapeEnergyLoss(baseLoss):
    def __init__(
        self,
        *,
        shapeWeight: float = 1.0,
        meanWeight: float = 1.0,
        gaugeWeight: float = 0.1,
        reconstructionWeight: float = 0.1,
        hotspotWeight: float = 0.0,
        energyBalanceWeight: float = 0.0,
        anchorFeatureIndex: int = -1,
        volumeSourceFeatureIndex: int = 14,
        boundaryBiotFeatureIndex: int = 6,
        boundaryTemperatureFeatureIndex: int = 7,
        boundaryFluxFeatureIndex: int = 8,
        boundaryTypeFeatureStart: int = 9,
        fixedTemperatureBoundaryTypeIds=(),
        **kwargs,
    ):
        super().__init__(lossName='thermalMeanShapeEnergyLoss', **kwargs)
        self.shapeWeight = float(shapeWeight)
        self.meanWeight = float(meanWeight)
        self.gaugeWeight = float(gaugeWeight)
        self.reconstructionWeight = float(reconstructionWeight)
        self.hotspotWeight = float(hotspotWeight)
        self.energyBalanceWeight = float(energyBalanceWeight)
        self.anchorFeatureIndex = int(anchorFeatureIndex)
        self.volumeSourceFeatureIndex = int(volumeSourceFeatureIndex)
        self.boundaryBiotFeatureIndex = int(boundaryBiotFeatureIndex)
        self.boundaryTemperatureFeatureIndex = int(
            boundaryTemperatureFeatureIndex
        )
        self.boundaryFluxFeatureIndex = int(boundaryFluxFeatureIndex)
        self.boundaryTypeFeatureStart = int(boundaryTypeFeatureStart)
        self.fixedTemperatureBoundaryTypeIds = tuple(
            int(value) for value in fixedTemperatureBoundaryTypeIds
        )
        weights = (
            self.shapeWeight,
            self.meanWeight,
            self.gaugeWeight,
            self.reconstructionWeight,
            self.hotspotWeight,
            self.energyBalanceWeight,
        )
        if any(value < 0.0 for value in weights):
            raise ValueError('thermal mean-shape loss weights must be non-negative')

    @property
    def requiresBoundaryPredictions(self) -> bool:
        return self.energyBalanceWeight > 0.0

    def compute(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        queryWeights: torch.Tensor,
        queryMask: torch.Tensor,
    ) -> torch.Tensor:
        weights = queryWeights * queryMask.to(dtype=queryWeights.dtype)
        squaredError = (predictions - targets).square().squeeze(-1)
        return (squaredError * weights).sum() / weights.sum().clamp_min(1e-12)

    def _normalizedQueryWeights(
        self,
        batch: dict,
        targets: torch.Tensor,
        *,
        applyHotspotWeight: bool,
    ) -> torch.Tensor:
        weights = batch['queryWeights'] * batch['queryMask'].to(
            dtype=batch['queryWeights'].dtype
        )
        if self.hotspotWeight and applyHotspotWeight:
            targetMinimum = targets.masked_fill(
                ~batch['queryMask'].unsqueeze(-1), torch.inf
            ).amin(dim=1, keepdim=True)
            targetMaximum = targets.masked_fill(
                ~batch['queryMask'].unsqueeze(-1), -torch.inf
            ).amax(dim=1, keepdim=True)
            normalized = (targets - targetMinimum) / (
                targetMaximum - targetMinimum
            ).clamp_min(1e-6)
            weights = weights * (
                1.0 + self.hotspotWeight * normalized.squeeze(-1)
            )
        return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)

    def _energyBalanceLoss(
        self,
        batch: dict,
        boundaryPredictions: torch.Tensor | None,
    ) -> torch.Tensor:
        if boundaryPredictions is None or not self.requiresBoundaryPredictions:
            return batch['targets'].new_zeros(())
        volumeMask = batch['volumeMask'].to(dtype=batch['volumeWeights'].dtype)
        boundaryMask = batch['boundaryMask'].to(
            dtype=batch['boundaryWeights'].dtype
        )
        volumeSource = _inverseSignedLog1p(
            batch['volumeFeatures'][..., self.volumeSourceFeatureIndex]
        )
        boundaryFlux = _inverseSignedLog1p(
            batch['boundaryFeatures'][..., self.boundaryFluxFeatureIndex]
        )
        biot = torch.expm1(
            batch['boundaryFeatures'][..., self.boundaryBiotFeatureIndex]
        ).clamp_min(0.0)
        ambient = batch['boundaryFeatures'][
            ..., self.boundaryTemperatureFeatureIndex
        ]
        volumePower = (
            volumeSource * batch['volumeWeights'] * volumeMask
        ).sum(dim=1)
        boundaryPower = (
            boundaryFlux * batch['boundaryWeights'] * boundaryMask
        ).sum(dim=1)
        convectionPower = (
            biot
            * (boundaryPredictions.squeeze(-1) - ambient)
            * batch['boundaryWeights']
            * boundaryMask
        ).sum(dim=1)
        fixed = torch.zeros_like(batch['boundaryMask'])
        for boundaryType in self.fixedTemperatureBoundaryTypeIds:
            fixed |= (
                batch['boundaryFeatures'][
                    ...,
                    self.boundaryTypeFeatureStart + boundaryType,
                ]
                > 0.5
            )
        hasFixed = (fixed & batch['boundaryMask']).any(dim=1)
        convectionConductance = (
            biot * batch['boundaryWeights'] * boundaryMask
        ).sum(dim=1)
        valid = (~hasFixed) & (convectionConductance > 1e-8)
        if not torch.any(valid):
            return batch['targets'].new_zeros(())
        residual = volumePower + boundaryPower - convectionPower
        normalization = (
            torch.abs(volumePower)
            + torch.abs(boundaryPower)
            + torch.abs(convectionPower)
        ).clamp_min(1e-3)
        return torch.mean(torch.square(residual[valid] / normalization[valid]))

    def computeBatch(
        self,
        modelOutput,
        batch: dict,
        *,
        boundaryPredictions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if isinstance(modelOutput, dict):
            predictions = modelOutput['prediction']
            rawShape = modelOutput['shape']
            meanOffset = modelOutput['meanOffset']
            decomposedOutput = True
        else:
            predictions = modelOutput
            rawShape = modelOutput
            meanOffset = None
            decomposedOutput = False
        targets = batch['targets']
        measureWeights = self._normalizedQueryWeights(
            batch,
            targets,
            applyHotspotWeight=False,
        )
        lossWeights = self._normalizedQueryWeights(
            batch,
            targets,
            applyHotspotWeight=True,
        )
        targetMean = torch.sum(
            measureWeights.unsqueeze(-1) * targets, dim=1, keepdim=True
        )
        targetShape = targets - targetMean
        rawShapeMean = torch.sum(
            measureWeights.unsqueeze(-1) * rawShape, dim=1, keepdim=True
        )
        if meanOffset is None:
            meanOffset = torch.sum(
                measureWeights.unsqueeze(-1) * predictions,
                dim=1,
                keepdim=True,
            )
        centeredShape = rawShape - rawShapeMean
        shapeError = (centeredShape - targetShape).square().squeeze(-1)
        shapeLoss = torch.mean(torch.sum(lossWeights * shapeError, dim=1))
        gaugeLoss = (
            torch.mean(rawShapeMean.square())
            if decomposedOutput
            else predictions.new_zeros(())
        )
        anchored = batch['globalFeatures'][:, self.anchorFeatureIndex] > 0.5
        if torch.any(anchored):
            meanError = meanOffset[anchored] - targetMean[anchored]
            meanLoss = torch.mean(meanError.square())
        else:
            meanLoss = predictions.new_zeros(())
        reconstructionError = (predictions - targets).square().squeeze(-1)
        reconstructionLoss = torch.mean(
            torch.sum(lossWeights * reconstructionError, dim=1)
        )
        energyLoss = self._energyBalanceLoss(batch, boundaryPredictions)
        return (
            self.shapeWeight * shapeLoss
            + self.meanWeight * meanLoss
            + self.gaugeWeight * gaugeLoss
            + self.reconstructionWeight * reconstructionLoss
            + self.energyBalanceWeight * energyLoss
        )


nnRegistry.register('losses', 'thermalOperatorLoss', thermalOperatorLoss)
nnRegistry.register(
    'losses', 'thermalMeanShapeEnergyLoss', thermalMeanShapeEnergyLoss
)

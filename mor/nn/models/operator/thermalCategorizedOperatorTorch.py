from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from typing import Any, Mapping, Sequence
from urllib.parse import quote

import torch
from torch import nn

from mor.nn.models.torchModelBase import torchModelBase
from mor.nn.registry import nnRegistry

from .thermalQuerySetOperatorTorch import thermalQuerySetOperatorTorch
from .thermalSetAttentionOperatorTorch import thermalSetAttentionOperatorTorch
from .thermalSetDeepONetTorch import thermalSetDeepONetTorch


THERMAL_CATEGORIZED_OPERATOR_FORMAT = 'thermal-categorized-pinn-v1'


def thermalCategoryKey(
    labels: Mapping[str, Any],
    categoryFields: Sequence[str],
) -> str:
    parts = []
    for rawField in categoryFields:
        field = str(rawField).strip()
        if not field:
            raise ValueError('category fields must be non-empty')
        value = labels.get(field)
        if value is None or not str(value).strip():
            raise KeyError(f'category labels do not contain {field!r}')
        parts.append(
            f'{quote(field, safe="-_")}={quote(str(value).strip(), safe="-_")}'
        )
    if not parts:
        raise ValueError('at least one category field is required')
    return '|'.join(parts)


def thermalCaseCategoryLabels(case: Any) -> dict[str, Any]:
    metadata = dict(getattr(case, 'metadata', {}) or {})
    return {
        **metadata,
        'caseId': getattr(case, 'caseId', None),
        'geometryId': getattr(case, 'geometryId', None),
        'componentType': getattr(case, 'componentType', None),
    }


def _cpuState(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _modelFromPayload(payload: Mapping[str, Any]) -> nn.Module:
    modelName = str(payload.get('modelName', 'thermalQuerySetOperatorTorch'))
    factories = {
        'thermalQuerySetOperatorTorch': thermalQuerySetOperatorTorch,
        'thermalSetAttentionOperatorTorch': thermalSetAttentionOperatorTorch,
        'thermalSetDeepONetTorch': thermalSetDeepONetTorch,
    }
    try:
        factory = factories[modelName]
    except KeyError as error:
        raise ValueError(
            f'unsupported categorized thermal model type: {modelName!r}'
        ) from error
    return factory(**payload['model'])


class thermalCategorizedOperatorTorch(torchModelBase):
    """Routes known categories to experts and all other cases to a global model."""

    def __init__(
        self,
        globalModel: nn.Module,
        expertModels: Mapping[str, nn.Module] | None = None,
        *,
        categoryFields: Sequence[str] = ('componentType',),
        expertBlendWeights: Mapping[str, float] | None = None,
        **kwargs,
    ):
        super().__init__(modelName='thermalCategorizedOperatorTorch', **kwargs)
        fields = tuple(str(value).strip() for value in categoryFields)
        if not fields or any(not value for value in fields) or len(set(fields)) != len(fields):
            raise ValueError('categoryFields must contain unique non-empty names')
        self.categoryFields = fields
        self.globalModel = globalModel
        self.expertModels = nn.ModuleDict()
        self._categoryToModule: dict[str, str] = {}
        self._expertBlendWeights: dict[str, float] = {}
        requestedBlendWeights = dict(expertBlendWeights or {})
        for categoryKey, model in (expertModels or {}).items():
            self.addExpert(
                categoryKey,
                model,
                blendWeight=requestedBlendWeights.pop(categoryKey, 1.0),
            )
        if requestedBlendWeights:
            raise KeyError(
                'expertBlendWeights contain unknown categories: '
                f'{sorted(requestedBlendWeights)!r}'
            )

    @property
    def categoryKeys(self) -> tuple[str, ...]:
        return tuple(sorted(self._categoryToModule))

    @property
    def expertBlendWeights(self) -> dict[str, float]:
        return {
            key: self._expertBlendWeights[key]
            for key in self.categoryKeys
        }

    def addExpert(
        self,
        categoryKey: str,
        model: nn.Module,
        *,
        blendWeight: float = 1.0,
    ) -> None:
        key = str(categoryKey).strip()
        if not key:
            raise ValueError('expert category key must be non-empty')
        if key in self._categoryToModule:
            raise ValueError(f'duplicate expert category: {key!r}')
        moduleKey = f'expert_{sha256(key.encode("utf-8")).hexdigest()[:24]}'
        if moduleKey in self.expertModels:
            raise ValueError(f'expert category hash collision: {key!r}')
        blendWeight = float(blendWeight)
        if not 0.0 <= blendWeight <= 1.0:
            raise ValueError('expert blend weight must be in [0, 1]')
        self.expertModels[moduleKey] = model
        self._categoryToModule[key] = moduleKey
        self._expertBlendWeights[key] = blendWeight

    def categoryKey(self, labels: Mapping[str, Any]) -> str:
        return thermalCategoryKey(labels, self.categoryFields)

    def _normalizedCategory(self, category: Any) -> str | None:
        if category is None:
            return None
        if isinstance(category, Mapping):
            try:
                return self.categoryKey(category)
            except KeyError:
                return None
        value = str(category).strip()
        return value or None

    def resolveModel(self, category: Any) -> tuple[nn.Module, dict[str, Any]]:
        requested = self._normalizedCategory(category)
        moduleKey = self._categoryToModule.get(requested or '')
        if moduleKey is None:
            return self.globalModel, {
                'requestedCategoryKey': requested,
                'selectedCategoryKey': '__global__',
                'usedGlobalFallback': True,
                'expertBlendWeight': 0.0,
            }
        blendWeight = self._expertBlendWeights[requested]
        return self.expertModels[moduleKey], {
            'requestedCategoryKey': requested,
            'selectedCategoryKey': requested,
            'usedGlobalFallback': blendWeight == 0.0,
            'expertBlendWeight': blendWeight,
        }

    @staticmethod
    def _sliceBatch(
        batch: dict[str, Any],
        indices: torch.Tensor,
        batchSize: int,
    ) -> dict[str, Any]:
        selected = {}
        cpuIndices = indices.detach().cpu().tolist()
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == batchSize:
                selected[key] = value.index_select(0, indices.to(value.device))
            elif isinstance(value, list) and len(value) == batchSize:
                selected[key] = [value[index] for index in cpuIndices]
            elif isinstance(value, tuple) and len(value) == batchSize:
                selected[key] = tuple(value[index] for index in cpuIndices)
            else:
                selected[key] = value
        return selected

    def _categories(self, batch: dict[str, Any], category: Any) -> list[Any]:
        batchSize = int(batch['queryCoordinates'].shape[0])
        resolved = batch.get('categoryKeys') if category is None else category
        if resolved is None or isinstance(resolved, (str, Mapping)):
            return [resolved] * batchSize
        values = list(resolved)
        if len(values) != batchSize:
            raise ValueError('category sequence must match the operator batch size')
        return values

    def forwardWithRouting(
        self,
        batch: dict[str, Any],
        category: Any = None,
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        categories = self._categories(batch, category)
        groupedIndices: dict[str | None, list[int]] = defaultdict(list)
        routing = []
        for index, value in enumerate(categories):
            _, route = self.resolveModel(value)
            groupedIndices[route['selectedCategoryKey']].append(index)
            routing.append(route)

        predictions = []
        positions = []
        batchSize = len(categories)
        device = batch['queryCoordinates'].device
        for selectedKey, rawIndices in groupedIndices.items():
            indices = torch.as_tensor(rawIndices, dtype=torch.long, device=device)
            selectedBatch = self._sliceBatch(batch, indices, batchSize)
            if selectedKey == '__global__':
                prediction = self.globalModel(selectedBatch)
            else:
                blendWeight = self._expertBlendWeights[selectedKey]
                if blendWeight == 0.0:
                    prediction = self.globalModel(selectedBatch)
                else:
                    prediction = self.expertModels[
                        self._categoryToModule[selectedKey]
                    ](selectedBatch)
                    if blendWeight < 1.0:
                        globalPrediction = self.globalModel(selectedBatch)
                        prediction = torch.lerp(
                            globalPrediction,
                            prediction,
                            blendWeight,
                        )
            predictions.append(prediction)
            positions.append(indices)
        combinedPrediction = torch.cat(predictions, dim=0)
        combinedPositions = torch.cat(positions, dim=0)
        originalOrder = torch.argsort(combinedPositions)
        return combinedPrediction.index_select(0, originalOrder), routing

    def forward(self, batch: dict[str, Any], category: Any = None) -> torch.Tensor:
        return self.forwardWithRouting(batch, category)[0]

    def toCheckpoint(
        self,
        *,
        globalModelConfig: Mapping[str, Any],
        expertModelConfigs: Mapping[str, Mapping[str, Any]] | None = None,
        expertMetadata: Mapping[str, Mapping[str, Any]] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        expertModelConfigs = expertModelConfigs or {}
        expertMetadata = expertMetadata or {}
        defaultConfig = dict(globalModelConfig)
        experts = {}
        for categoryKey in self.categoryKeys:
            model = self.expertModels[self._categoryToModule[categoryKey]]
            experts[categoryKey] = {
                'modelName': str(
                    getattr(model, 'modelName', 'thermalQuerySetOperatorTorch')
                ),
                'model': dict(expertModelConfigs.get(categoryKey, defaultConfig)),
                'stateDict': _cpuState(model),
                'blendWeight': self._expertBlendWeights[categoryKey],
                'metadata': dict(expertMetadata.get(categoryKey, {})),
            }
        return {
            'formatVersion': THERMAL_CATEGORIZED_OPERATOR_FORMAT,
            'categoryFields': list(self.categoryFields),
            'routingPolicy': 'exact-category-then-global',
            'global': {
                'modelName': str(
                    getattr(
                        self.globalModel,
                        'modelName',
                        'thermalQuerySetOperatorTorch',
                    )
                ),
                'model': defaultConfig,
                'stateDict': _cpuState(self.globalModel),
            },
            'experts': experts,
            'metadata': dict(metadata or {}),
        }

    @classmethod
    def fromCheckpoint(cls, checkpoint: Mapping[str, Any]) -> 'thermalCategorizedOperatorTorch':
        if checkpoint.get('formatVersion') != THERMAL_CATEGORIZED_OPERATOR_FORMAT:
            raise ValueError(
                f'unsupported categorized thermal checkpoint: '
                f'{checkpoint.get("formatVersion")!r}'
            )
        globalPayload = checkpoint['global']
        globalModel = _modelFromPayload(globalPayload)
        globalModel.load_state_dict(globalPayload['stateDict'])
        experts = {}
        blendWeights = {}
        for categoryKey, payload in checkpoint.get('experts', {}).items():
            model = _modelFromPayload(payload)
            model.load_state_dict(payload['stateDict'])
            experts[str(categoryKey)] = model
            blendWeights[str(categoryKey)] = float(payload.get('blendWeight', 1.0))
        return cls(
            globalModel,
            experts,
            categoryFields=checkpoint['categoryFields'],
            expertBlendWeights=blendWeights,
        )


nnRegistry.register(
    'models',
    'thermalCategorizedOperatorTorch',
    thermalCategorizedOperatorTorch,
)

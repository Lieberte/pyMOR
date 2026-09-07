from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from mor.nn.data import (
    STEADY_THERMAL_DIMENSIONLESS_V2,
    collateThermalOperatorCases,
    enrichThermalAssemblyCases,
    loadSubsampledThermalDataset,
    randomSimilarityAugmentations,
    steadyThermalDimensionlessOperatorCase,
    thermalComponentFeatureSchema,
    thermalAssemblyContextFeatureNames,
)
from mor.nn.losses import thermalSteadyPinnLoss
from mor.nn.models import (
    thermalCaseCategoryLabels,
    thermalCategorizedOperatorTorch,
    thermalCategoryKey,
    thermalQuerySetOperatorTorch,
    thermalSetAttentionOperatorTorch,
    thermalSetDeepONetTorch,
)
from mor.nn.trainers import thermalOperatorTrainerTorch


OPERATING_SCALES = {
    'speedRpm': 25000.0,
    'absoluteTemperatureAnchor': 1.0,
}
def _featureSchema(contextFeatureNames: Sequence[str] = ()):
    return thermalComponentFeatureSchema(
        operatingFeatureNames=(
            'speedRpm',
            *tuple(contextFeatureNames),
            'absoluteTemperatureAnchor',
        ),
        boundaryTypeCount=4,
        temperatureBoundaryTypeIds=(1, 2),
        fixedTemperatureBoundaryTypeIds=(2,),
    )


SCHEMA = _featureSchema()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Train categorized steady thermal PINN experts plus an all-case '
            'global fallback from a standardized discrete component dataset.'
        )
    )
    parser.add_argument('source', type=Path)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument(
        '--initial-global-checkpoint',
        type=Path,
        help=(
            'Reuse the global model from a categorized thermal checkpoint and '
            'train only the requested experts.'
        ),
    )
    parser.add_argument('--analysis-type', default='steady')
    parser.add_argument(
        '--category-fields',
        nargs='+',
        default=('componentType',),
        help=(
            'Hierarchy labels forming an expert key, for example componentType '
            'or systemType assemblyType componentType.'
        ),
    )
    parser.add_argument('--components', nargs='*')
    parser.add_argument('--minimum-category-cases', type=int, default=3)
    parser.add_argument(
        '--expert-balance-fields',
        nargs='+',
        help=(
            'Optional hierarchy fields used only to oversample minority groups '
            'inside each expert; they do not change the routing key.'
        ),
    )
    parser.add_argument(
        '--expert-balance-components',
        nargs='*',
        help='Limit expert balancing to these component types (default: all).',
    )
    parser.add_argument('--validation-fraction', type=float, default=0.2)
    parser.add_argument('--maximum-volume-samples', type=int, default=256)
    parser.add_argument('--maximum-boundary-samples', type=int, default=512)
    parser.add_argument('--maximum-query-samples', type=int, default=256)
    parser.add_argument('--global-epochs', type=int, default=300)
    parser.add_argument('--expert-epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=2e-3)
    parser.add_argument('--expert-learning-rate', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument(
        '--model-family',
        choices=('query-set', 'set-deeponet', 'set-attention'),
        default='query-set',
    )
    parser.add_argument(
        '--residual-block-count',
        type=int,
        default=0,
        help='Number of smooth residual blocks in the decoder (0 keeps the legacy MLP).',
    )
    parser.add_argument('--residual-scale', type=float, default=1.0)
    parser.add_argument(
        '--decoder-architecture',
        choices=('mlp', 'resnet', 'unet', 'dense'),
        help=(
            'Decoder family. If omitted, residual blocks select resnet and '
            'zero residual blocks preserve the legacy MLP.'
        ),
    )
    parser.add_argument('--kernel-scales', nargs='+', type=float, default=(0.12, 0.35))
    parser.add_argument('--query-fourier-frequencies', type=int, default=4)
    parser.add_argument(
        '--context-architecture',
        choices=('rbf', 'global', 'deeponet', 'attention', 'hybrid'),
        default='rbf',
    )
    parser.add_argument('--attention-head-count', type=int, default=4)
    parser.add_argument(
        '--assembly-context',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Add component identity, parent-assembly and interface input features.',
    )
    parser.add_argument('--minimum-temperature-scale-k', type=float, default=1.0)
    parser.add_argument('--similarity-augmentations-per-case', type=int, default=0)
    parser.add_argument('--energy-balance-weight', type=float, default=0.02)
    parser.add_argument('--pde-residual-weight', type=float, default=0.01)
    parser.add_argument(
        '--pde-residual-mode',
        choices=('meshfree', 'strong'),
        default='meshfree',
    )
    parser.add_argument('--pde-collocation-count', type=int, default=16)
    parser.add_argument('--pde-kernel-scale', type=float, default=0.2)
    parser.add_argument('--neumann-residual-weight', type=float, default=0.0)
    parser.add_argument('--robin-residual-weight', type=float, default=0.0)
    parser.add_argument('--dirichlet-residual-weight', type=float, default=0.0)
    parser.add_argument('--boundary-collocation-count', type=int, default=64)
    parser.add_argument(
        '--physics-in-validation',
        action='store_true',
        help='Also compute the second-order PDE residual on every validation batch.',
    )
    parser.add_argument('--shape-loss-weight', type=float, default=1.0)
    parser.add_argument('--mean-loss-weight', type=float, default=1.0)
    parser.add_argument('--reconstruction-loss-weight', type=float, default=0.1)
    parser.add_argument('--hotspot-loss-weight', type=float, default=0.25)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=7)
    return parser.parse_args()


class _SplitDataModule:
    def __init__(self, trainCases, validationCases, *, batchSize: int, seed: int):
        self.trainCases = tuple(trainCases)
        self.validationCases = tuple(validationCases) or self.trainCases
        self.batchSize = int(batchSize)
        self.seed = int(seed)
        if not self.trainCases:
            raise ValueError('training split must contain at least one case')

    def _loader(self, cases, shuffle: bool):
        return DataLoader(
            cases,
            batch_size=min(self.batchSize, len(cases)),
            shuffle=shuffle,
            collate_fn=collateThermalOperatorCases,
            generator=torch.Generator().manual_seed(self.seed),
        )

    def getTrainLoader(self):
        return self._loader(self.trainCases, True)

    def getValidationLoader(self):
        return self._loader(self.validationCases, False)


def _caseCategory(case, categoryFields: Sequence[str]) -> str:
    return thermalCategoryKey(thermalCaseCategoryLabels(case), categoryFields)


def _balancedExpertCases(
    entries,
    *,
    balanceFields: Sequence[str],
    seed: int,
):
    grouped = defaultdict(list)
    for sourceCase, operatorCase in entries:
        grouped[_caseCategory(sourceCase, balanceFields)].append(operatorCase)
    if len(grouped) < 2:
        return [operatorCase for _, operatorCase in entries], {
            key: len(values) for key, values in sorted(grouped.items())
        }
    targetCount = max(len(values) for values in grouped.values())
    generator = np.random.default_rng(seed)
    balanced = []
    counts = {}
    for key, values in sorted(grouped.items()):
        indices = list(range(len(values)))
        while len(indices) < targetCount:
            indices.extend(
                generator.permutation(len(values)).tolist()[
                    : min(len(values), targetCount - len(indices))
                ]
            )
        balanced.extend(values[index] for index in indices)
        counts[key] = len(indices)
    generator.shuffle(balanced)
    return balanced, counts


def _caseGroup(case) -> str:
    labels = thermalCaseCategoryLabels(case)
    source = str(labels.get('sourceDataset', ''))
    assembly = str(labels.get('assemblyId') or labels.get('motorId') or '')
    parent = str(labels.get('parentCaseId') or case.caseId)
    return '|'.join((source, assembly, parent))


def _splitCases(cases, validationFraction: float, seed: int):
    if not 0.0 <= validationFraction < 1.0:
        raise ValueError('validationFraction must be in [0, 1)')
    groups = sorted({_caseGroup(case) for case in cases})
    if validationFraction == 0.0 or len(groups) < 2:
        return tuple(cases), ()
    generator = np.random.default_rng(seed)
    generator.shuffle(groups)
    validationCount = min(
        len(groups) - 1,
        max(1, int(round(validationFraction * len(groups)))),
    )
    validationGroups = frozenset(groups[:validationCount])
    return (
        tuple(case for case in cases if _caseGroup(case) not in validationGroups),
        tuple(case for case in cases if _caseGroup(case) in validationGroups),
    )


def _dimensionless(
    cases,
    minimumTemperatureScaleK: float,
    schema: thermalComponentFeatureSchema = SCHEMA,
):
    return tuple(
        steadyThermalDimensionlessOperatorCase(
            case,
            schema=schema,
            operatingScales=OPERATING_SCALES,
            minimumTemperatureScaleK=minimumTemperatureScaleK,
            featureMode=STEADY_THERMAL_DIMENSIONLESS_V2,
        )
        for case in cases
    )


def _modelConfig(
    example,
    arguments: argparse.Namespace,
    schema: thermalComponentFeatureSchema = SCHEMA,
) -> dict[str, Any]:
    common = {
        'volumeFeatureDim': example.volumeFeatureDim,
        'boundaryFeatureDim': example.boundaryFeatureDim,
        'globalFeatureDim': example.globalFeatureDim,
        'queryDim': example.queryDim,
    }
    modelFamily = getattr(arguments, 'model_family', 'query-set')
    if modelFamily == 'set-deeponet':
        return {
            **common,
            'pointHiddenDim': arguments.hidden_dim,
            'globalHiddenDim': arguments.hidden_dim,
            'latentDim': 96,
            'poolingMode': 'mean_max',
            'queryFourierFrequencies': arguments.query_fourier_frequencies,
        }
    if modelFamily == 'set-attention':
        return {
            **common,
            'hiddenDim': arguments.hidden_dim,
            'latentCount': 8,
            'attentionHeads': arguments.attention_head_count,
            'queryFourierFrequencies': arguments.query_fourier_frequencies,
        }
    config = {
        **common,
        'hiddenDim': arguments.hidden_dim,
        'kernelScales': list(arguments.kernel_scales),
        'queryFourierFrequencies': arguments.query_fourier_frequencies,
        'includeRelativeKernelFeatures': True,
        'separateBoundaryTypeContexts': True,
        'boundaryTypeFeatureStart': 9,
        'boundaryTypeCount': schema.boundaryTypeCount,
        'decomposeMeanAndShape': True,
        'anchorFeatureIndex': -1,
        'residualBlockCount': arguments.residual_block_count,
        'residualScale': arguments.residual_scale,
        'contextArchitecture': arguments.context_architecture,
        'attentionHeadCount': arguments.attention_head_count,
    }
    if arguments.decoder_architecture is not None:
        config['decoderArchitecture'] = arguments.decoder_architecture
    return config


def _newModel(modelConfig: Mapping[str, Any], modelFamily: str):
    factories = {
        'query-set': thermalQuerySetOperatorTorch,
        'set-deeponet': thermalSetDeepONetTorch,
        'set-attention': thermalSetAttentionOperatorTorch,
    }
    return factories[modelFamily](**modelConfig)


def _loss(
    arguments: argparse.Namespace,
    schema: thermalComponentFeatureSchema = SCHEMA,
) -> thermalSteadyPinnLoss:
    return thermalSteadyPinnLoss(
        shapeWeight=arguments.shape_loss_weight,
        meanWeight=arguments.mean_loss_weight,
        gaugeWeight=0.1,
        reconstructionWeight=arguments.reconstruction_loss_weight,
        hotspotWeight=arguments.hotspot_loss_weight,
        energyBalanceWeight=arguments.energy_balance_weight,
        fixedTemperatureBoundaryTypeIds=schema.fixedTemperatureBoundaryTypeIds,
        pdeResidualWeight=arguments.pde_residual_weight,
        pdeResidualMode=arguments.pde_residual_mode,
        pdeCollocationCount=arguments.pde_collocation_count,
        pdeKernelScale=arguments.pde_kernel_scale,
        neumannResidualWeight=getattr(arguments, 'neumann_residual_weight', 0.0),
        robinResidualWeight=getattr(arguments, 'robin_residual_weight', 0.0),
        dirichletResidualWeight=getattr(
            arguments, 'dirichlet_residual_weight', 0.0
        ),
        boundaryCollocationCount=getattr(
            arguments, 'boundary_collocation_count', 64
        ),
    )


def _fit(
    model,
    trainCases,
    validationCases,
    *,
    arguments: argparse.Namespace,
    epochs: int,
    learningRate: float,
    seed: int,
    schema: thermalComponentFeatureSchema = SCHEMA,
):
    trainer = thermalOperatorTrainerTorch(
        _loss(arguments, schema),
        epochs=epochs,
        learningRate=learningRate,
        weightDecay=arguments.weight_decay,
        deviceName=arguments.device,
        physicsInValidation=arguments.physics_in_validation,
    )
    result = trainer.fit(
        model,
        _SplitDataModule(
            trainCases,
            validationCases,
            batchSize=arguments.batch_size,
            seed=seed,
        ),
    )
    result.pop('bestModelState', None)
    return result


def _loadInitialGlobal(
    checkpointPath: Path,
    modelConfig: dict[str, Any],
    modelFamily: str,
):
    checkpoint = torch.load(checkpointPath, map_location='cpu', weights_only=False)
    if checkpoint.get('formatVersion') is None or 'global' not in checkpoint:
        raise ValueError(
            'initial global checkpoint must use the categorized thermal format'
        )
    storedConfig = dict(checkpoint['global']['model'])
    if modelFamily == 'query-set':
        storedConfig.setdefault('residualBlockCount', 0)
        storedConfig.setdefault('residualScale', 1.0)
        storedConfig.setdefault('contextArchitecture', 'rbf')
        storedConfig.setdefault('attentionHeadCount', 4)
    normalizedStored = {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in storedConfig.items()
    }
    normalizedRequested = {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in modelConfig.items()
    }
    if normalizedStored != normalizedRequested:
        raise ValueError(
            'initial global checkpoint model configuration does not match the '
            f'requested configuration: {normalizedStored!r} != '
            f'{normalizedRequested!r}'
        )
    expectedModelName = {
        'query-set': 'thermalQuerySetOperatorTorch',
        'set-deeponet': 'thermalSetDeepONetTorch',
        'set-attention': 'thermalSetAttentionOperatorTorch',
    }[modelFamily]
    if checkpoint['global'].get('modelName', 'thermalQuerySetOperatorTorch') != expectedModelName:
        raise ValueError('initial global checkpoint model family does not match')
    model = _newModel(modelConfig, modelFamily)
    model.load_state_dict(checkpoint['global']['stateDict'])
    return model


def _evaluate(
    model,
    cases,
    dimensionlessCases,
    device: str,
    *,
    expertModel=None,
    expertBlendWeight: float = 0.0,
):
    records = []
    model.eval()
    with torch.no_grad():
        for case, dimensionlessCase in zip(cases, dimensionlessCases):
            operatorCase = dimensionlessCase.operatorCase
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in collateThermalOperatorCases([operatorCase]).items()
            }
            prediction = model(batch)[0, :, 0]
            if expertModel is not None and expertBlendWeight > 0.0:
                expertPrediction = expertModel(batch)[0, :, 0]
                prediction = torch.lerp(
                    prediction,
                    expertPrediction,
                    float(expertBlendWeight),
                )
            target = batch['targets'][0, :, 0]
            mask = batch['queryMask'][0]
            weights = batch['queryWeights'][0, mask]
            if not dimensionlessCase.scales.absoluteTemperatureAnchorAvailable:
                weightTotal = weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
                prediction = prediction.clone()
                target = target.clone()
                prediction[mask] -= torch.sum(weights * prediction[mask]) / weightTotal
                target[mask] -= torch.sum(weights * target[mask]) / weightTotal
            errorK = (
                prediction[mask] - target[mask]
            ) * dimensionlessCase.scales.temperatureScaleK
            squaredErrorIntegral = float(torch.sum(weights * errorK.square()).cpu())
            weightSum = float(weights.sum().cpu())
            records.append(
                {
                    'caseId': case.caseId,
                    'categoryKey': None,
                    'componentType': case.componentType,
                    'rmseK': math.sqrt(squaredErrorIntegral / weightSum),
                    'maximumErrorK': float(torch.max(torch.abs(errorK)).cpu()),
                    '_squaredErrorIntegral': squaredErrorIntegral,
                    '_weightSum': weightSum,
                }
            )
    return records


def _calibrateExpertBlend(
    globalModel,
    expertModel,
    cases,
    dimensionlessCases,
    device: str,
):
    if not cases:
        return 0.0, [], {'reason': 'noCategoryValidationCases'}
    candidates = []
    for alpha in np.linspace(0.0, 1.0, 11):
        records = _evaluate(
            globalModel,
            cases,
            dimensionlessCases,
            device,
            expertModel=expertModel,
            expertBlendWeight=float(alpha),
        )
        summary = _summary(records)
        candidates.append(
            {
                'blendWeight': float(alpha),
                'evaluation': summary,
                '_records': records,
            }
        )
    selected = min(
        candidates,
        key=lambda value: (
            value['evaluation']['volumeWeightedRmseK'],
            value['evaluation']['macroCaseRmseK'],
            value['evaluation']['maximumErrorK'],
        ),
    )
    publicCandidates = [
        {key: value for key, value in candidate.items() if key != '_records'}
        for candidate in candidates
    ]
    return (
        float(selected['blendWeight']),
        selected['_records'],
        {
            'strategy': 'heldOutGridSearch',
            'objective': 'volumeWeightedRmseK',
            'candidates': publicCandidates,
        },
    )


def _summary(records) -> dict[str, Any] | None:
    if not records:
        return None
    squaredError = sum(record['_squaredErrorIntegral'] for record in records)
    weight = sum(record['_weightSum'] for record in records)
    return {
        'caseCount': len(records),
        'volumeWeightedRmseK': math.sqrt(squaredError / weight),
        'macroCaseRmseK': float(np.mean([record['rmseK'] for record in records])),
        'maximumErrorK': max(record['maximumErrorK'] for record in records),
    }


def _public(records):
    return [
        {key: value for key, value in record.items() if not key.startswith('_')}
        for record in records
    ]


def _byCategorySummary(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[record['categoryKey']].append(record)
    return {key: _summary(values) for key, values in sorted(grouped.items())}


def main() -> None:
    arguments = _arguments()
    if arguments.minimum_category_cases <= 0:
        raise ValueError('minimum-category-cases must be positive')
    dataset = loadSubsampledThermalDataset(
        arguments.source,
        analysisType=arguments.analysis_type,
        componentTypes=arguments.components,
        maximumVolumeSamples=arguments.maximum_volume_samples,
        maximumBoundarySamples=arguments.maximum_boundary_samples,
        maximumQuerySamples=arguments.maximum_query_samples,
        randomSeed=arguments.seed,
    )
    if not dataset.cases:
        raise ValueError('no cases match the requested analysis type and components')
    componentTypes = tuple(sorted({case.componentType for case in dataset.cases}))
    contextFeatureNames = (
        thermalAssemblyContextFeatureNames(componentTypes)
        if arguments.assembly_context
        else ()
    )
    schema = _featureSchema(contextFeatureNames)
    cases = (
        enrichThermalAssemblyCases(
            dataset.cases,
            componentTypes=componentTypes,
        )
        if arguments.assembly_context
        else dataset.cases
    )
    categoryFields = tuple(arguments.category_fields)
    categoryByCaseId = {
        case.caseId: _caseCategory(case, categoryFields) for case in cases
    }
    trainCases, validationCases = _splitCases(
        cases,
        arguments.validation_fraction,
        arguments.seed,
    )
    augmentedTrainCases = randomSimilarityAugmentations(
        trainCases,
        augmentationsPerCase=arguments.similarity_augmentations_per_case,
        schema=schema,
        randomSeed=arguments.seed,
    )
    trainDimensionless = _dimensionless(
        augmentedTrainCases,
        arguments.minimum_temperature_scale_k,
        schema,
    )
    validationDimensionless = _dimensionless(
        validationCases,
        arguments.minimum_temperature_scale_k,
        schema,
    )
    trainOperatorCases = tuple(value.operatorCase for value in trainDimensionless)
    validationOperatorCases = tuple(
        value.operatorCase for value in validationDimensionless
    )
    modelConfig = _modelConfig(trainOperatorCases[0], arguments, schema)
    torch.manual_seed(arguments.seed)
    if arguments.initial_global_checkpoint is None:
        globalModel = _newModel(modelConfig, arguments.model_family)
        print(
            f'training global fallback on {len(trainOperatorCases)} cases...',
            file=sys.stderr,
        )
        globalTraining = _fit(
            globalModel,
            trainOperatorCases,
            validationOperatorCases,
            arguments=arguments,
            epochs=arguments.global_epochs,
            learningRate=arguments.learning_rate,
            seed=arguments.seed,
            schema=schema,
        )
        resolvedDevice = globalTraining['resolvedDeviceName']
    else:
        print(
            f'reusing global fallback from {arguments.initial_global_checkpoint}...',
            file=sys.stderr,
        )
        globalModel = _loadInitialGlobal(
            arguments.initial_global_checkpoint,
            modelConfig,
            arguments.model_family,
        )
        resolvedDevice = globalModel.toDevice(
            arguments.device,
            allowAutoFallback=True,
        )
        globalModel.evalMode()
        globalTraining = {
            'reusedCheckpoint': str(arguments.initial_global_checkpoint),
            'trainLossHistory': [],
            'validationLossHistory': [],
            'bestValidationLoss': None,
            'bestEpochIndex': None,
            'resolvedDeviceName': resolvedDevice,
        }
    evaluationCases = validationCases or trainCases
    evaluationDimensionless = (
        validationDimensionless
        if validationCases
        else _dimensionless(
            trainCases,
            arguments.minimum_temperature_scale_k,
            schema,
        )
    )
    globalRecords = _evaluate(
        globalModel,
        evaluationCases,
        evaluationDimensionless,
        resolvedDevice,
    )
    for record, case in zip(globalRecords, evaluationCases):
        record['categoryKey'] = categoryByCaseId[case.caseId]

    originalTrainByCategory = defaultdict(list)
    augmentedTrainByCategory = defaultdict(list)
    augmentedTrainEntriesByCategory = defaultdict(list)
    validationByCategory = defaultdict(list)
    evaluationByCategory = defaultdict(list)
    for case in trainCases:
        originalTrainByCategory[categoryByCaseId[case.caseId]].append(case)
    for case, value in zip(augmentedTrainCases, trainDimensionless):
        sourceCaseId = case.metadata.get('similarityAugmentation', {}).get(
            'sourceCaseId', case.caseId
        )
        augmentedTrainByCategory[categoryByCaseId[sourceCaseId]].append(
            value.operatorCase
        )
        augmentedTrainEntriesByCategory[categoryByCaseId[sourceCaseId]].append(
            (next(case for case in trainCases if case.caseId == sourceCaseId), value.operatorCase)
        )
    for case, value in zip(validationCases, validationDimensionless):
        validationByCategory[categoryByCaseId[case.caseId]].append(
            value.operatorCase
        )
    for case, value in zip(evaluationCases, evaluationDimensionless):
        evaluationByCategory[categoryByCaseId[case.caseId]].append((case, value))

    experts = {}
    expertReports = {}
    expertMetadata = {}
    expertRecordsByCaseId = {}
    routedExpertRecordsByCaseId = {}
    expertBlendWeights = {}
    skippedCategories = {}
    balancingReports = {}
    balanceFields = tuple(arguments.expert_balance_fields or ())
    balanceComponents = (
        None
        if arguments.expert_balance_components is None
        else frozenset(arguments.expert_balance_components)
    )
    for categoryIndex, categoryKey in enumerate(sorted(originalTrainByCategory)):
        originalCount = len(originalTrainByCategory[categoryKey])
        if originalCount < arguments.minimum_category_cases:
            skippedCategories[categoryKey] = {
                'reason': 'insufficientTrainingCases',
                'trainingCaseCount': originalCount,
                'minimumCategoryCases': arguments.minimum_category_cases,
            }
            continue
        expertTrainCases = augmentedTrainByCategory[categoryKey]
        componentType = originalTrainByCategory[categoryKey][0].componentType
        if balanceFields and (
            balanceComponents is None or componentType in balanceComponents
        ):
            expertTrainCases, balancedCounts = _balancedExpertCases(
                augmentedTrainEntriesByCategory[categoryKey],
                balanceFields=balanceFields,
                seed=arguments.seed + categoryIndex + 1000,
            )
            balancingReports[categoryKey] = {
                'fields': list(balanceFields),
                'originalCount': len(augmentedTrainByCategory[categoryKey]),
                'balancedCount': len(expertTrainCases),
                'balancedGroupCounts': balancedCounts,
            }
        print(
            f'training expert {categoryKey} on '
            f'{len(expertTrainCases)} cases...',
            file=sys.stderr,
        )
        expert = copy.deepcopy(globalModel)
        training = _fit(
            expert,
            expertTrainCases,
            validationByCategory[categoryKey],
            arguments=arguments,
            epochs=arguments.expert_epochs,
            learningRate=arguments.expert_learning_rate,
            seed=arguments.seed + categoryIndex + 1,
            schema=schema,
        )
        pairs = evaluationByCategory[categoryKey]
        records = _evaluate(
            expert,
            [pair[0] for pair in pairs],
            [pair[1] for pair in pairs],
            training['resolvedDeviceName'],
        )
        for record in records:
            record['categoryKey'] = categoryKey
            expertRecordsByCaseId[record['caseId']] = record
        blendWeight, blendedRecords, calibration = _calibrateExpertBlend(
            globalModel,
            expert,
            [pair[0] for pair in pairs],
            [pair[1] for pair in pairs],
            training['resolvedDeviceName'],
        )
        for record in blendedRecords:
            record['categoryKey'] = categoryKey
            routedExpertRecordsByCaseId[record['caseId']] = record
        expertBlendWeights[categoryKey] = blendWeight
        experts[categoryKey] = expert.cpu()
        expertReports[categoryKey] = {
            'training': training,
            'evaluation': _summary(records),
            'cases': _public(records),
            'blendWeight': blendWeight,
            'calibration': calibration,
            'blendedEvaluation': _summary(blendedRecords),
        }
        expertMetadata[categoryKey] = {
            'trainingCaseCount': originalCount,
            'effectiveTrainingCaseCount': len(expertTrainCases),
            'blendWeight': blendWeight,
        }
        if resolvedDevice.startswith('cuda'):
            torch.cuda.empty_cache()

    routedRecords = []
    routingCounts = Counter()
    for globalRecord in globalRecords:
        categoryKey = globalRecord['categoryKey']
        expertRecord = routedExpertRecordsByCaseId.get(globalRecord['caseId'])
        if expertRecord is None:
            routedRecords.append(globalRecord)
            routingCounts['globalFallback'] += 1
        else:
            routedRecords.append(expertRecord)
            blendWeight = expertBlendWeights[categoryKey]
            routeName = (
                'globalFallback'
                if blendWeight == 0.0
                else (
                    'categoryExpert'
                    if blendWeight == 1.0
                    else 'globalExpertBlend'
                )
            )
            routingCounts[routeName] += 1

    bundle = thermalCategorizedOperatorTorch(
        globalModel.cpu(),
        experts,
        categoryFields=categoryFields,
        expertBlendWeights=expertBlendWeights,
    )
    physics = {
        'dimensionlessPde': 'div(K*grad(theta)) + q = 0',
        'pdeResidualWeight': arguments.pde_residual_weight,
        'pdeResidualMode': arguments.pde_residual_mode,
        'pdeCollocationCount': arguments.pde_collocation_count,
        'pdeKernelScale': arguments.pde_kernel_scale,
        'physicsInValidation': arguments.physics_in_validation,
        'energyBalanceWeight': arguments.energy_balance_weight,
        'meanShapeDecomposition': True,
        'similarityAugmentationsPerCase': (
            arguments.similarity_augmentations_per_case
        ),
    }
    report = {
        'source': str(dataset.sourcePath),
        'analysisType': arguments.analysis_type,
        'modelFamily': arguments.model_family,
        'categoryFields': list(categoryFields),
        'routingPolicy': 'validation-calibrated-category-blend-then-global',
        'caseCounts': {
            'selected': len(cases),
            'train': len(trainCases),
            'effectiveTrain': len(augmentedTrainCases),
            'validation': len(validationCases),
            'categoryCounts': dict(
                sorted(Counter(categoryByCaseId.values()).items())
            ),
        },
        'physics': physics,
        'globalFallback': {
            'training': globalTraining,
            'evaluation': _summary(globalRecords),
            'byCategory': _byCategorySummary(globalRecords),
            'cases': _public(globalRecords),
        },
        'experts': expertReports,
        'skippedCategories': skippedCategories,
        'expertBalancing': balancingReports,
        'routedModel': {
            'evaluation': _summary(routedRecords),
            'byCategory': _byCategorySummary(routedRecords),
            'routingCounts': dict(routingCounts),
            'cases': _public(routedRecords),
        },
    }
    checkpoint = bundle.toCheckpoint(
        globalModelConfig=modelConfig,
        expertMetadata=expertMetadata,
        metadata={
            'source': str(dataset.sourcePath),
            'analysisType': arguments.analysis_type,
            'modelFamily': arguments.model_family,
            'featureMode': STEADY_THERMAL_DIMENSIONLESS_V2,
            'schema': {
                'operatingFeatureNames': list(schema.operatingFeatureNames),
                'boundaryTypeCount': schema.boundaryTypeCount,
                'temperatureBoundaryTypeIds': list(
                    schema.temperatureBoundaryTypeIds
                ),
                'fixedTemperatureBoundaryTypeIds': list(
                    schema.fixedTemperatureBoundaryTypeIds
                ),
            },
            'operatingScales': OPERATING_SCALES,
            'minimumTemperatureScaleK': arguments.minimum_temperature_scale_k,
            'physics': physics,
            'assemblyContext': {
                'enabled': arguments.assembly_context,
                'componentTypes': list(componentTypes),
                'featureNames': list(contextFeatureNames),
            },
        },
    )
    arguments.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, arguments.checkpoint)
    report['checkpoint'] = str(arguments.checkpoint)
    output = arguments.output or arguments.checkpoint.with_suffix('.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

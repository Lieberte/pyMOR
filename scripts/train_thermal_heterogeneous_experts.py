from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch

from mor.nn.data import (
    STEADY_THERMAL_DIMENSIONLESS_V2,
    loadSubsampledThermalDataset,
    thermalCheckpointDataContract,
)
from mor.nn.models import (
    thermalCategorizedOperatorTorch,
    thermalCategoryKey,
)

try:
    from scripts.train_thermal_discrete_pinn import (
        _calibrateExpertBlend,
        _evaluate,
        _fit,
        _splitCases,
        _summary,
    )
except ModuleNotFoundError:
    from train_thermal_discrete_pinn import (
        _calibrateExpertBlend,
        _evaluate,
        _fit,
        _splitCases,
        _summary,
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Fine-tune heterogeneous per-component experts over a global fallback.'
    )
    parser.add_argument('source', type=Path)
    parser.add_argument('--global-checkpoint', type=Path, required=True)
    parser.add_argument('--sweep', type=Path, action='append', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expert-epochs', type=int, default=40)
    parser.add_argument('--expert-learning-rate', type=float, default=2e-4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--minimum-category-cases', type=int, default=3)
    parser.add_argument('--pde-residual-weight', type=float, default=0.002)
    parser.add_argument('--pde-collocation-count', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=23)
    return parser.parse_args()


def _trainingArguments(arguments):
    return SimpleNamespace(
        shape_loss_weight=1.0,
        mean_loss_weight=1.0,
        reconstruction_loss_weight=0.1,
        hotspot_loss_weight=0.25,
        energy_balance_weight=0.02,
        pde_residual_weight=arguments.pde_residual_weight,
        pde_residual_mode='meshfree',
        pde_collocation_count=arguments.pde_collocation_count,
        pde_kernel_scale=0.2,
        weight_decay=0.0,
        device=arguments.device,
        physics_in_validation=False,
        batch_size=arguments.batch_size,
    )


def _candidateIndex(paths):
    candidates = []
    for path in paths:
        report = json.loads(path.read_text(encoding='utf-8'))
        for candidate in report['candidates']:
            candidates.append(
                {
                    **candidate,
                    'sweep': str(path),
                }
            )
    return candidates


def _selectedCandidates(candidates, components):
    result = {}
    for component in components:
        eligible = [
            candidate
            for candidate in candidates
            if component in candidate.get('byComponent', {})
        ]
        if not eligible:
            continue
        result[component] = min(
            eligible,
            key=lambda candidate: (
                candidate['byComponent'][component]['volumeWeightedRmseK'],
                candidate['byComponent'][component]['maximumErrorK'],
            ),
        )
    return result


def _public(records):
    return [
        {key: value for key, value in record.items() if not key.startswith('_')}
        for record in records
    ]


def main() -> None:
    arguments = _arguments()
    dataset = loadSubsampledThermalDataset(
        arguments.source,
        analysisType='steady',
        maximumVolumeSamples=128,
        maximumBoundarySamples=256,
        maximumQuerySamples=128,
        randomSeed=arguments.seed,
    )
    globalCheckpoint = torch.load(
        arguments.global_checkpoint,
        map_location='cpu',
        weights_only=False,
    )
    dataContract = thermalCheckpointDataContract.fromCheckpoint(globalCheckpoint)
    componentTypes = dataContract.componentTypes
    contextNames = dataContract.contextFeatureNames
    schema = dataContract.schema
    cases = dataContract.prepareCases(dataset.cases)
    trainCases, validationCases = _splitCases(cases, 0.2, arguments.seed)
    trainDimensionless = dataContract.transformCases(trainCases)
    validationDimensionless = dataContract.transformCases(validationCases)
    trainByComponent = defaultdict(list)
    validationByComponent = defaultdict(list)
    validationPairsByComponent = defaultdict(list)
    for case, value in zip(trainCases, trainDimensionless):
        trainByComponent[case.componentType].append(value.operatorCase)
    for case, value in zip(validationCases, validationDimensionless):
        validationByComponent[case.componentType].append(value.operatorCase)
        validationPairsByComponent[case.componentType].append((case, value))

    globalBundle = thermalCategorizedOperatorTorch.fromCheckpoint(globalCheckpoint)
    globalModel = globalBundle.globalModel
    globalConfig = dict(globalCheckpoint['global']['model'])
    globalDevice = globalModel.toDevice(arguments.device, allowAutoFallback=True)
    globalRecords = _evaluate(
        globalModel,
        validationCases,
        validationDimensionless,
        globalDevice,
    )
    for record in globalRecords:
        record['categoryKey'] = thermalCategoryKey(
            {'componentType': record['componentType']}, ('componentType',)
        )

    candidates = _candidateIndex(arguments.sweep)
    selectedCandidates = _selectedCandidates(candidates, componentTypes)
    trainingArguments = _trainingArguments(arguments)
    experts = {}
    expertConfigs = {}
    expertBlendWeights = {}
    expertMetadata = {}
    reports = {}
    routedByCaseId = {}
    skipped = {}
    for index, component in enumerate(componentTypes):
        trainOperatorCases = trainByComponent.get(component, ())
        validationOperatorCases = validationByComponent.get(component, ())
        if len(trainOperatorCases) < arguments.minimum_category_cases:
            skipped[component] = {
                'reason': 'insufficientTrainingCases',
                'trainingCaseCount': len(trainOperatorCases),
            }
            continue
        candidate = selectedCandidates.get(component)
        if candidate is None:
            skipped[component] = {'reason': 'noValidatedArchitectureCandidate'}
            continue
        print(
            f'[{index + 1}/{len(componentTypes)}] {component}: '
            f'{candidate["name"]} -> expert',
            file=sys.stderr,
            flush=True,
        )
        candidateCheckpoint = torch.load(
            candidate['checkpoint'],
            map_location='cpu',
            weights_only=False,
        )
        candidateBundle = thermalCategorizedOperatorTorch.fromCheckpoint(
            candidateCheckpoint
        )
        expert = copy.deepcopy(candidateBundle.globalModel)
        training = _fit(
            expert,
            trainOperatorCases,
            validationOperatorCases,
            arguments=trainingArguments,
            epochs=arguments.expert_epochs,
            learningRate=arguments.expert_learning_rate,
            seed=arguments.seed + index + 1,
            schema=schema,
        )
        pairs = validationPairsByComponent.get(component, ())
        validationComponentCases = [pair[0] for pair in pairs]
        validationComponentDimensionless = [pair[1] for pair in pairs]
        expertRecords = _evaluate(
            expert,
            validationComponentCases,
            validationComponentDimensionless,
            training['resolvedDeviceName'],
        )
        blendWeight, blendedRecords, calibration = _calibrateExpertBlend(
            globalModel,
            expert,
            validationComponentCases,
            validationComponentDimensionless,
            training['resolvedDeviceName'],
        )
        categoryKey = thermalCategoryKey(
            {'componentType': component}, ('componentType',)
        )
        for record in blendedRecords:
            record['categoryKey'] = categoryKey
            routedByCaseId[record['caseId']] = record
        experts[categoryKey] = expert.cpu()
        expertConfigs[categoryKey] = dict(candidateCheckpoint['global']['model'])
        expertBlendWeights[categoryKey] = blendWeight
        expertMetadata[categoryKey] = {
            'componentType': component,
            'initializationCandidate': candidate['name'],
            'initializationCheckpoint': candidate['checkpoint'],
            'trainingCaseCount': len(trainOperatorCases),
            'validationCaseCount': len(validationOperatorCases),
            'blendWeight': blendWeight,
        }
        reports[component] = {
            'initializationCandidate': candidate['name'],
            'training': training,
            'expertEvaluation': _summary(expertRecords),
            'blendWeight': blendWeight,
            'calibration': calibration,
            'blendedEvaluation': _summary(blendedRecords),
            'cases': _public(blendedRecords),
        }
        if globalDevice.startswith('cuda'):
            torch.cuda.empty_cache()

    routedRecords = [
        routedByCaseId.get(record['caseId'], record) for record in globalRecords
    ]
    routingCounts = Counter()
    for record in routedRecords:
        alpha = expertBlendWeights.get(record['categoryKey'], 0.0)
        routingCounts[
            'globalFallback'
            if alpha == 0.0
            else ('categoryExpert' if alpha == 1.0 else 'globalExpertBlend')
        ] += 1
    bundle = thermalCategorizedOperatorTorch(
        globalModel.cpu(),
        experts,
        categoryFields=('componentType',),
        expertBlendWeights=expertBlendWeights,
    )
    checkpoint = bundle.toCheckpoint(
        globalModelConfig=globalConfig,
        expertModelConfigs=expertConfigs,
        expertMetadata=expertMetadata,
        metadata={
            **dict(globalCheckpoint.get('metadata', {})),
            'source': str(dataset.sourcePath),
            'analysisType': 'steady',
            'featureMode': STEADY_THERMAL_DIMENSIONLESS_V2,
            'routingPolicy': 'validation-calibrated-heterogeneous-experts',
            'globalCheckpoint': str(arguments.global_checkpoint),
            'schema': {
                'operatingFeatureNames': list(schema.operatingFeatureNames),
                'boundaryTypeCount': schema.boundaryTypeCount,
                'temperatureBoundaryTypeIds': list(schema.temperatureBoundaryTypeIds),
                'fixedTemperatureBoundaryTypeIds': list(
                    schema.fixedTemperatureBoundaryTypeIds
                ),
            },
            'assemblyContext': {
                'enabled': True,
                'componentTypes': list(componentTypes),
                'featureNames': list(contextNames),
            },
            'physics': {
                'energyBalanceWeight': 0.02,
                'pdeResidualMode': 'meshfree',
                'pdeResidualWeight': arguments.pde_residual_weight,
                'pdeCollocationCount': arguments.pde_collocation_count,
            },
        },
    )
    arguments.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, arguments.checkpoint)
    report = {
        'source': str(dataset.sourcePath),
        'checkpoint': str(arguments.checkpoint),
        'globalCheckpoint': str(arguments.global_checkpoint),
        'caseCounts': {
            'selected': len(cases),
            'train': len(trainCases),
            'validation': len(validationCases),
        },
        'globalFallback': {
            'evaluation': _summary(globalRecords),
            'cases': _public(globalRecords),
        },
        'experts': reports,
        'skippedComponents': skipped,
        'routedModel': {
            'evaluation': _summary(routedRecords),
            'routingCounts': dict(routingCounts),
            'cases': _public(routedRecords),
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from mor.nn.data import (
    loadSubsampledThermalDataset,
    thermalCheckpointDataContract,
)
from mor.nn.models import thermalCategorizedOperatorTorch

try:
    from scripts.sweep_thermal_all_models import _meshfreeResidual
    from scripts.train_thermal_discrete_pinn import _splitCases
except ModuleNotFoundError:
    from sweep_thermal_all_models import _meshfreeResidual
    from train_thermal_discrete_pinn import _splitCases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('sweep', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--pde-collocation-count', type=int, default=8)
    arguments = parser.parse_args()
    report = json.loads(arguments.sweep.read_text(encoding='utf-8'))
    source = Path(report['dataset']['source'])
    dataset = loadSubsampledThermalDataset(
        source,
        analysisType='steady',
        maximumVolumeSamples=128,
        maximumBoundarySamples=256,
        maximumQuerySamples=128,
        randomSeed=int(report['dataset']['seed']),
    )
    options = SimpleNamespace(
        pde_collocation_count=arguments.pde_collocation_count,
        pde_kernel_scale=0.2,
        batch_size=arguments.batch_size,
    )
    values = {}
    for candidate in report['candidates']:
        checkpoint = torch.load(
            candidate['checkpoint'], map_location='cpu', weights_only=False
        )
        dataContract = thermalCheckpointDataContract.fromCheckpoint(checkpoint)
        cases = dataContract.prepareCases(dataset.cases)
        _, validationCases = _splitCases(
            cases, 0.2, int(report['dataset']['seed'])
        )
        operatorCases = tuple(
            value.operatorCase
            for value in dataContract.transformCases(validationCases)
        )
        model = thermalCategorizedOperatorTorch.fromCheckpoint(
            checkpoint
        ).globalModel
        values[candidate['name']] = _meshfreeResidual(
            model,
            operatorCases,
            options,
            model.toDevice(arguments.device, allowAutoFallback=True),
        )
    output = {
        'sourceSweep': str(arguments.sweep),
        'residual': 'stable kernel graph Laplacian V2',
        'values': values,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()

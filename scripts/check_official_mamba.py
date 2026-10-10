import argparse
import json
import platform
from pathlib import Path

import torch

from mor.nn.models.mambaDynamicsTorch import mambaDynamicsTorch


def main():
    parser = argparse.ArgumentParser(description='Validate actual mamba-ssm CUDA kernels against the PyTorch model.')
    parser.add_argument('--output', type=Path, default=Path('reports/official_mamba/report.json'))
    args = parser.parse_args()
    if platform.system() != 'Linux' or not torch.cuda.is_available():
        parser.error('official Mamba kernel validation requires Linux with NVIDIA CUDA')
    import importlib.metadata
    import selective_scan_cuda
    version = importlib.metadata.version('mamba-ssm')
    if version != '2.2.6.post3': parser.error('this validation targets mamba-ssm==2.2.6.post3')
    cases = []
    for length in (1, 17, 128):
        torch.manual_seed(7)
        options = dict(inputDim=3, outputDim=2, dModel=8, stateDim=4, convWidth=4, numLayers=1)
        reference = mambaDynamicsTorch(**options, engine='reference').cuda()
        official = mambaDynamicsTorch(**options, engine='mamba-ssm').cuda()
        official.loadState(reference.saveState(toCpu=False))
        values = torch.randn(2, length, 3, device='cuda')
        expected, actual = reference(values), official(values)
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
        expected.square().mean().backward()
        actual.square().mean().backward()
        gradientError = 0.
        referenceParameters = dict(reference.named_parameters())
        officialParameters = dict(official.named_parameters())
        if referenceParameters.keys() != officialParameters.keys():
            raise AssertionError('official/reference parameter names differ')
        for name, baseline in referenceParameters.items():
            optimized = officialParameters[name]
            torch.testing.assert_close(optimized.grad, baseline.grad, rtol=2e-3, atol=2e-5)
            gradientError = max(gradientError, float((baseline.grad - optimized.grad).abs().max()))
        optimizer = torch.optim.Adam(official.parameters(), lr=1e-3)
        optimizer.step()
        restored = mambaDynamicsTorch(**options, engine='mamba-ssm').cuda()
        restored.loadState(official.saveState())
        torch.testing.assert_close(restored(values), official(values))
        torch.cuda.synchronize()
        cases.append({'length': length, 'maxOutputDifference': float((actual - expected).abs().max().detach()),
                      'maxGradientDifference': gradientError, 'passed': True})
    payload = {'engine': 'mamba-ssm', 'version': version, 'compiledModule': selective_scan_cuda.__file__,
               'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(), 'cases': cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()

import argparse
import ast
import copy
import json
import platform
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from mor.nn.models.mambaDynamicsTorch import _ReferenceMamba


def loadReference(path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'selective_scan_ref')
    namespace = {'torch': torch, 'F': F, 'rearrange': rearrange, 'repeat': repeat}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['selective_scan_ref']


def referenceForward(block, inputs, scan):
    x, gate = block.in_proj(inputs).chunk(2, dim=-1)
    x = F.silu(block.conv1d(x.transpose(1, 2))[..., :inputs.shape[1]])
    delta, b, c = block.x_proj(x.transpose(1, 2)).split([block.dt_rank, block.d_state, block.d_state], dim=-1)
    delta = F.linear(delta, block.dt_proj.weight).transpose(1, 2)
    result = scan(x, delta, -block.A_log.float().exp(), b.transpose(1, 2), c.transpose(1, 2),
                  D=block.D.float(), z=gate.transpose(1, 2), delta_bias=block.dt_proj.bias.float(), delta_softplus=True)
    return block.out_proj(result.transpose(1, 2))


def main():
    parser = argparse.ArgumentParser(description='Compare native Mamba with the official local selective_scan_ref source.')
    parser.add_argument('--source', type=Path, required=True, help='Trusted local selective_scan_interface.py from mamba-ssm 2.2.6.post3')
    parser.add_argument('--output', type=Path, default=Path('reports/mamba_reference/reference.json'))
    args = parser.parse_args()
    reference = loadReference(args.source)
    rows = []
    for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
        for length in [1, 17, 64]:
            torch.manual_seed(7)
            current = _ReferenceMamba(8, d_state=4, d_conv=4, expand=2).to(device)
            previous = copy.deepcopy(current)
            inputs = torch.randn(2, length, 8, device=device)
            result, expected = current(inputs), referenceForward(previous, inputs, reference)
            torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)
            result.square().mean().backward()
            expected.square().mean().backward()
            gradientDifference = 0.
            for parameter, baseline in zip(current.parameters(), previous.parameters()):
                torch.testing.assert_close(parameter.grad, baseline.grad, rtol=1e-4, atol=1e-7)
                gradientDifference = max(gradientDifference, float((parameter.grad - baseline.grad).abs().max()))
            rows.append({'device': device, 'length': length, 'maxOutputDifference': float((result - expected).abs().max().detach()),
                         'maxGradientDifference': gradientDifference, 'passed': True})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {'source': str(args.source.resolve()), 'sourceVersion': '2.2.6.post3',
               'torch': torch.__version__, 'platform': platform.platform(), 'cases': rows,
               'scope': 'Official Python reference function; does not validate the compiled CUDA kernels'}
    args.output.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()

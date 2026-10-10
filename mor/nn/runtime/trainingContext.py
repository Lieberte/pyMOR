from contextlib import contextmanager
from functools import wraps
from threading import RLock
import os

import torch

from mor.runtime import resolveDeviceName

_trainingLock = RLock()


@contextmanager
def trainingContext(runtime):
    device = resolveDeviceName(runtime.deviceName)
    devices = list(range(torch.cuda.device_count())) if device == 'cuda' else []
    with _trainingLock, torch.random.fork_rng(devices=devices):
        deterministic = torch.are_deterministic_algorithms_enabled()
        warnOnly = torch.is_deterministic_algorithms_warn_only_enabled()
        cudnnDeterministic = torch.backends.cudnn.deterministic
        cudnnBenchmark = torch.backends.cudnn.benchmark
        workspace = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
        try:
            if devices and runtime.deterministic and workspace is None:
                os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            torch.use_deterministic_algorithms(runtime.deterministic)
            torch.backends.cudnn.deterministic = runtime.deterministic
            if runtime.deterministic:
                torch.backends.cudnn.benchmark = False
            if runtime.randomSeed is not None:
                torch.random.default_generator.manual_seed(runtime.randomSeed)
                if devices:
                    torch.cuda.manual_seed_all(runtime.randomSeed)
            yield
        finally:
            torch.use_deterministic_algorithms(deterministic, warn_only=warnOnly)
            torch.backends.cudnn.deterministic = cudnnDeterministic
            torch.backends.cudnn.benchmark = cudnnBenchmark
            if workspace is None:
                os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
            else:
                os.environ['CUBLAS_WORKSPACE_CONFIG'] = workspace


def configuredTraining(operation):
    @wraps(operation)
    def run(config, *args, **kwargs):
        with trainingContext(config.runtime):
            return operation(config, *args, **kwargs)
    return run

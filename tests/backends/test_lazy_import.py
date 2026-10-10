import os
import subprocess
import sys
from pathlib import Path


def runFreshPython(source):
    result = subprocess.run([sys.executable, '-c', source],
                            cwd=Path(__file__).resolve().parents[2],
                            env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'},
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_classical_import_and_backend_discovery_do_not_load_torch():
    runFreshPython('''
import sys
import mor
from mor.backends import backendRegistry
assert {'scipy', 'torch'} <= set(backendRegistry.list())
assert backendRegistry.getDefaultBackendName() == 'scipy'
assert backendRegistry.get().name == 'scipy'
assert 'torch' not in sys.modules
assert 'mor.nn' not in sys.modules
''')


def test_torch_backend_loads_on_first_request():
    runFreshPython('''
import sys
import mor
from mor.backends import backendRegistry
assert 'torch' in backendRegistry.list()
backendRegistry.setDefault('torch')
assert 'torch' not in sys.modules
backend = backendRegistry.get(device='cpu')
assert backend.name == 'torch'
assert backend.array.zeros((1,)).device.type == 'cpu'
assert 'torch' in sys.modules
assert backendRegistry.list().count('torch') == 1
''')


def test_backend_registration_and_classical_pod_do_not_probe_unused_dependencies():
    runFreshPython('''
import importlib.abc
import sys
class NoDependencyDiscovery(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'h5py', 'meshio', 'sklearn', 'onnx', 'onnxruntime'}:
            raise AssertionError('unused dependency was probed: ' + fullname)
sys.meta_path.insert(0, NoDependencyDiscovery())
import mor
from mor.reductors import podReductor
result = podReductor('scipy').reduce(mor.matrixOperator([[2., 0.], [0., 1.]], 'scipy'), rank=1)
assert result.order == 1
assert 'torch' in mor.backendRegistry.list()
''')


def test_torch_backend_import_errors_propagate_on_request():
    runFreshPython('''
import importlib.abc
import sys
import mor
failure = ModuleNotFoundError('packaged Torch dependency is missing', name='torch')
class BrokenTorchImport(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch':
            raise failure
sys.meta_path.insert(0, BrokenTorchImport())
assert 'torch' in mor.backendRegistry.list()
try:
    mor.backendRegistry.get('torch', device='cpu')
except ModuleNotFoundError as error:
    assert error is failure
else:
    raise AssertionError('backend dependency failure was hidden')
''')

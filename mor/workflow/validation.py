import math
import re
from pathlib import Path

from .storage import jsonCopy

API_VERSION = 1
MAX_INPUT_BYTES = 256 * 1024 * 1024
MAX_ESTIMATED_PEAK_BYTES = 2 * 1024 * 1024 * 1024
MAX_HEADER_BYTES = 64 * 1024
IDENTIFIER = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$')
INPUT_METHODS = {'snapshots': ('pod', 'autoencoder'), 'timeSeries': ('pod', 'dmd', 'pod-dmd', 'autoencoder'),
                 'ltiSystem': ('bt',)}
CLASSICAL_PARAMETERS = {
    'dmd': {'rank': None, 'tol': None, 'verbose': True},
    'pod-dmd': {'podRank': None, 'podTol': None, 'dmdRank': None, 'dmdTol': None,
                'centerSnapshots': False, 'verbose': True},
    'bt': {'order': None, 'maxError': None, 'verbose': True},
}
AUTOENCODER_PARAMETERS = {
    'latentDim': {'type': 'integer', 'default': 2, 'minimum': 1, 'maximum': 4096},
    'hiddenDim': {'type': 'integer', 'default': 64, 'minimum': 1, 'maximum': 4096},
    'epochs': {'type': 'integer', 'default': 100, 'minimum': 1, 'maximum': 10000},
    'batchSize': {'type': 'integer', 'default': 32, 'minimum': 1, 'maximum': 65536},
    'learningRate': {'type': 'number', 'default': 0.001, 'exclusiveMinimum': 0, 'maximum': 1},
    'seed': {'type': 'integer', 'default': 42, 'minimum': 0, 'maximum': 2147483647},
    'verbose': {'type': 'boolean', 'default': True},
}


def _parameterDefaults(method):
    if isinstance(method, str) and method in CLASSICAL_PARAMETERS:
        return dict(CLASSICAL_PARAMETERS[method])
    if method == 'autoencoder':
        return {name: schema['default'] for name, schema in AUTOENCODER_PARAMETERS.items()}
    return {'verbose': True, **({'steps': 20, 'delayMs': 50, 'failAtStep': None}
                               if method == 'mock' else
                               {'rank': None, 'tol': None, 'centerSnapshots': False})}


def getCapabilities():
    return {
        'apiVersion': API_VERSION,
        'methods': [
            {
                'id': 'mock', 'available': True, 'reason': None,
                'parameters': {
                    'steps': {'type': 'integer', 'default': 20, 'minimum': 1, 'maximum': 10000},
                    'delayMs': {'type': 'integer', 'default': 50, 'minimum': 0, 'maximum': 60000},
                    'failAtStep': {'type': 'integer', 'nullable': True},
                    'verbose': {'type': 'boolean', 'default': True},
                },
            },
            {
                'id': 'pod', 'available': True,
                'reason': None,
                'availabilityCheck': 'packaged-dependencies',
                'parameters': {
                    'rank': {'type': 'integer', 'nullable': True, 'minimum': 1},
                    'tol': {'type': 'number', 'nullable': True, 'minimum': 0, 'exclusiveMaximum': 1,
                            'meaning': 'discarded squared singular-value energy fraction'},
                    'centerSnapshots': {'type': 'boolean', 'default': False},
                    'verbose': {'type': 'boolean', 'default': True},
                },
                'exclusiveParameters': [['rank', 'tol']],
                'input': {'format': 'npy', 'axes': ['state', 'snapshot'], 'realOnly': True},
                'backend': 'scipy', 'device': 'cpu', 'weighted': False,
            },
            {
                'id': 'autoencoder', 'label': 'NN / Autoencoder', 'available': True,
                'reason': None,
                'availabilityCheck': 'packaged-dependencies',
                'parameters': jsonCopy(AUTOENCODER_PARAMETERS),
                'input': {'format': 'npy', 'axes': ['state', 'snapshot'], 'realOnly': True},
                'backend': 'torch', 'device': 'cpu', 'dtype': 'float32',
                'architecture': 'fully-connected-ReLU', 'optimizer': 'adam',
                'preprocessing': 'none; supply already prepared training snapshots',
            },
        ] + [
            {'id': method, 'available': True,
             'reason': None,
             'inputKinds': [kind for kind, methods in INPUT_METHODS.items() if method in methods],
             'parameters': {key: {'default': value, 'nullable': value is None,
                                  'type': 'boolean' if isinstance(value, bool) else
                                  'integer' if key in ('rank', 'podRank', 'dmdRank', 'order') else 'number'}
                            for key, value in defaults.items()},
             'backend': 'scipy', 'device': 'cpu'}
            for method, defaults in CLASSICAL_PARAMETERS.items()
        ],
        'inputKinds': {key: list(value) for key, value in INPUT_METHODS.items()},
        'maxConcurrentTasks': 1,
        'cancellation': 'cooperative-stage-checkpoints',
        'limits': {'maxInputBytes': MAX_INPUT_BYTES, 'maxEstimatedPeakBytes': MAX_ESTIMATED_PEAK_BYTES},
    }


def validateWorkflow(request):
    errors = []

    def reject(field, message, code='invalid_request'):
        errors.append({'field': field, 'code': code, 'message': message})

    if not isinstance(request, dict):
        reject('request', 'request must be a JSON object')
        return {'valid': False, 'request': None, 'errors': errors}
    try:
        normalized = jsonCopy(request)
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        reject('request', f'request must contain finite JSON values: {error}')
        return {'valid': False, 'request': None, 'errors': errors}
    if type(normalized.get('apiVersion')) is not int or normalized['apiVersion'] != API_VERSION:
        reject('apiVersion', 'only apiVersion 1 is supported')
    for key in ('taskId', 'nodeId'):
        if not isinstance(normalized.get(key), str) or not IDENTIFIER.fullmatch(normalized[key]):
            reject(key, 'must be a nonempty identifier of at most 128 ASCII letters, digits, _, ., :, or -')
    caseId = normalized.get('caseId')
    if not isinstance(caseId, str) or not caseId.strip() or len(caseId) > 2048 or '\x00' in caseId:
        reject('caseId', 'must be a nonempty case identifier of at most 2048 characters')
    method = normalized.get('method')
    if method not in ('mock', 'pod', 'autoencoder', 'dmd', 'pod-dmd', 'bt'):
        reject('method', 'unsupported reduction method', 'unsupported_method')
    inputKind = normalized.get('inputKind', 'snapshots')
    if method != 'mock' and (not isinstance(inputKind, str) or method not in INPUT_METHODS.get(inputKind, ())):
        reject('inputKind', 'method is incompatible with the selected mathematical input kind')
    if normalized.get('eventPolicy', 'stream') not in ('stream', 'finalOnly'):
        reject('eventPolicy', 'must be stream or finalOnly')

    parameters = normalized.setdefault('parameters', {})
    if not isinstance(parameters, dict):
        reject('parameters', 'parameters must be an object')
        parameters = {}
    allowed = set(_parameterDefaults(method))
    for key in parameters.keys() - allowed:
        reject(f'parameters.{key}', 'unsupported parameter')
    parameters.setdefault('verbose', True)
    if type(parameters['verbose']) is not bool:
        reject('parameters.verbose', 'must be a boolean')
    if method == 'mock':
        for key, default, minimum, maximum in [('steps', 20, 1, 10000), ('delayMs', 50, 0, 60000)]:
            value = parameters.setdefault(key, default)
            if type(value) is not int or not minimum <= value <= maximum:
                reject(f'parameters.{key}', f'must be an integer between {minimum} and {maximum}')
        failAtStep = parameters.setdefault('failAtStep', None)
        if failAtStep is not None and (type(failAtStep) is not int or failAtStep < 1 or
                                     type(parameters['steps']) is int and failAtStep > parameters['steps']):
            reject('parameters.failAtStep', 'must be null or a 1-based step within steps')
    if method == 'pod':
        rank = parameters.setdefault('rank', None)
        tol = parameters.setdefault('tol', None)
        parameters.setdefault('centerSnapshots', False)
        if rank is not None and (type(rank) is not int or rank < 1):
            reject('parameters.rank', 'must be null or a positive integer')
        if tol is not None and (type(tol) not in (int, float) or not 0 <= tol < 1 or not math.isfinite(tol)):
            reject('parameters.tol', 'must be null or a finite discarded energy fraction in [0, 1)')
        if rank is not None and tol is not None:
            reject('parameters', 'rank and tol are mutually exclusive')
        if type(parameters['centerSnapshots']) is not bool:
            reject('parameters.centerSnapshots', 'must be a boolean')
    if method == 'autoencoder':
        for key, schema in AUTOENCODER_PARAMETERS.items():
            value = parameters.setdefault(key, schema['default'])
            if schema['type'] == 'integer' and (type(value) is not int or
                                               not schema['minimum'] <= value <= schema['maximum']):
                reject(f'parameters.{key}', f'must be an integer between {schema["minimum"]} and {schema["maximum"]}')
            elif schema['type'] == 'number' and (type(value) not in (int, float) or not 0 < value <= 1):
                reject(f'parameters.{key}', 'must be a finite number in (0, 1]')

    if isinstance(method, str) and method in CLASSICAL_PARAMETERS:
        for key, default in CLASSICAL_PARAMETERS[method].items():
            value = parameters.setdefault(key, default)
            if key in ('verbose', 'centerSnapshots'):
                if type(value) is not bool:
                    reject(f'parameters.{key}', 'must be a boolean')
            elif value is not None:
                if key in ('rank', 'podRank', 'dmdRank', 'order'):
                    if type(value) is not int or not 1 <= value <= 2147483647:
                        reject(f'parameters.{key}', 'must be a positive integer')
                elif type(value) not in (int, float) or not math.isfinite(value) or (
                        value <= 0 if key == 'maxError' else not 0 <= value < 1):
                    reject(f'parameters.{key}', 'must be positive (maxError) or in [0, 1) (energy tolerance)')
        for rank, tolerance in (('rank', 'tol'), ('podRank', 'podTol'), ('dmdRank', 'dmdTol'), ('order', 'maxError')):
            if parameters.get(rank) is not None and parameters.get(tolerance) is not None:
                reject('parameters', f'{rank} and {tolerance} are mutually exclusive')

    inputs = normalized.setdefault('inputs', {})
    if not isinstance(inputs, dict):
        reject('inputs', 'inputs must be an object')
        inputs = {}
    if method in ('pod', 'autoencoder', 'dmd', 'pod-dmd'):
        reference = inputs.get('snapshots')
        if not isinstance(reference, dict):
            reject('inputs.snapshots', 'snapshots reference is required')
        else:
            for key in reference.keys() - {'path', 'format', 'axes', 'shape', 'dtype', 'sha256'}:
                reject(f'inputs.snapshots.{key}', 'unsupported input reference field')
            if reference.get('format') != 'npy':
                reject('inputs.snapshots.format', 'only npy is supported')
            if reference.get('axes') != ['state', 'snapshot']:
                reject('inputs.snapshots.axes', 'must be ["state", "snapshot"]')
            if 'shape' in reference and (not isinstance(reference['shape'], list) or len(reference['shape']) != 2 or
                                         any(type(size) is not int or size < 1 for size in reference['shape'])):
                reject('inputs.snapshots.shape', 'must be two positive integer dimensions')
            if 'dtype' in reference and (not isinstance(reference['dtype'], str) or not reference['dtype']):
                reject('inputs.snapshots.dtype', 'must be a NumPy dtype name')
            if 'sha256' in reference and (not isinstance(reference['sha256'], str) or
                                          not re.fullmatch('[a-fA-F0-9]{64}', reference['sha256'])):
                reject('inputs.snapshots.sha256', 'must be a SHA-256 hexadecimal digest')
            source = _absolutePath(reference.get('path'), 'inputs.snapshots.path', reject)
            if source is not None:
                try:
                    if not source.is_file() or source.suffix.lower() != '.npy':
                        reject('inputs.snapshots.path', 'must reference an existing .npy file')
                    elif source.stat().st_size > MAX_INPUT_BYTES + MAX_HEADER_BYTES:
                        reject('inputs.snapshots.path', 'input file exceeds the configured size limit', 'input_too_large')
                    reference['path'] = str(source)
                except OSError as error:
                    reject('inputs.snapshots.path', str(error), 'input_unavailable')

    if method in ('dmd', 'pod-dmd'):
        sequence = inputs.get('sequence')
        if not isinstance(sequence, dict):
            reject('inputs.sequence', 'time-series metadata with positive dt is required')
        else:
            dt = sequence.get('dt')
            if type(dt) not in (int, float) or not math.isfinite(dt) or dt <= 0:
                reject('inputs.sequence.dt', 'must be a positive finite sampling interval')
            if set(sequence) - {'dt', 'pairs'}:
                reject('inputs.sequence', 'unsupported time-series metadata')
            if 'pairs' in sequence:
                pairs = sequence['pairs']
                if not isinstance(pairs, list) or not pairs or any(
                    not isinstance(pair, list) or len(pair) != 2 or any(type(i) is not int or i < 0 for i in pair)
                    or pair[1] != pair[0] + 1 for pair in pairs):
                    reject('inputs.sequence.pairs', 'must contain consecutive nonnegative index pairs')
                elif any(pairs[i][0] >= pairs[i + 1][0] for i in range(len(pairs) - 1)):
                    reject('inputs.sequence.pairs', 'pairs must be unique and ordered')
    if method == 'bt':
        _validateSystem(inputs.get('system'), reject)

    output = _absolutePath(normalized.get('outputDirectory'), 'outputDirectory', reject)
    if output is not None:
        try:
            if output.exists() or output.is_symlink():
                reject('outputDirectory', 'task output directory must not already exist', 'output_exists')
            if not output.parent.is_dir():
                reject('outputDirectory', 'parent directory must already exist')
            normalized['outputDirectory'] = str(output)
        except OSError as error:
            reject('outputDirectory', str(error), 'output_unavailable')
    if 'workflow' in normalized:
        _validateWorkflowSnapshot(normalized, reject)
    return {'valid': not errors, 'request': normalized if not errors else None, 'errors': errors}


def _validateSystem(system, reject):
    if not isinstance(system, dict) or set(system) - {'matrices', 'isContinuous', 'dt'}:
        reject('inputs.system', 'system requires matrices and a time domain')
        return
    continuous = system.setdefault('isContinuous', True)
    if type(continuous) is not bool:
        reject('inputs.system.isContinuous', 'must be a boolean')
    if 'dt' in system or continuous is False:
        dt = system.get('dt')
        if type(dt) not in (int, float) or not math.isfinite(dt) or dt <= 0:
            reject('inputs.system.dt', 'must be a positive sampling interval')
    matrices = system.get('matrices')
    if not isinstance(matrices, dict) or not {'A', 'B', 'C'} <= matrices.keys() or matrices.keys() - {'A', 'B', 'C', 'D', 'E'}:
        reject('inputs.system.matrices', 'supply A, B, C and optional D, E')
        return
    for key, reference in matrices.items():
        field = f'inputs.system.matrices.{key}'
        if not isinstance(reference, dict):
            reject(field, 'matrix reference must be an object')
            continue
        if set(reference) - {'path', 'format', 'axes', 'shape', 'dtype', 'sha256'}:
            reject(field, 'unsupported matrix reference field')
        if reference.get('format') != 'npy' or reference.get('axes') != ['row', 'column']:
            reject(field, 'matrix must be NPY with axes [row, column]')
        if 'sha256' in reference and (not isinstance(reference['sha256'], str) or not re.fullmatch('[a-fA-F0-9]{64}', reference['sha256'])):
            reject(field, 'invalid SHA-256 digest')
        source = _absolutePath(reference.get('path'), field + '.path', reject)
        if source is not None:
            try:
                if not source.is_file() or source.suffix.lower() != '.npy' or source.stat().st_size > MAX_INPUT_BYTES + MAX_HEADER_BYTES:
                    reject(field, 'matrix must be an existing bounded NPY file')
                reference['path'] = str(source)
            except OSError as error:
                reject(field, str(error))


def _validateWorkflowSnapshot(request, reject):
    workflow = request['workflow']
    if not isinstance(workflow, dict):
        reject('workflow', 'workflow snapshot must be an object')
        return
    if workflow.get('executionScope') != 'selectedNode' or workflow.get('selectedNodeId') != request.get('nodeId'):
        reject('workflow.executionScope', 'only selectedNode execution with matching selectedNodeId is supported')
    if type(workflow.get('schemaVersion')) is not int or workflow['schemaVersion'] != 1:
        reject('workflow.schemaVersion', 'only workflow schemaVersion 1 is supported')
    graph = workflow.get('graph')
    if not isinstance(graph, dict) or not isinstance(graph.get('nodes'), list) or not isinstance(graph.get('connections'), list):
        reject('workflow.graph', 'graph must contain nodes and connections arrays')
        return
    nodes = {}
    for index, node in enumerate(graph['nodes']):
        if not isinstance(node, dict) or not isinstance(node.get('uuid'), str) or not node['uuid'] or node['uuid'] in nodes:
            reject(f'workflow.graph.nodes[{index}]', 'nodes must have unique nonempty uuid strings')
            return
        nodes[node['uuid']] = node
    selected = nodes.get(request.get('nodeId')) if isinstance(request.get('nodeId'), str) else None
    if selected is None or selected.get('typeId') != 'rom.reductionMethod':
        reject('workflow.selectedNodeId', 'selected node must be a rom.reductionMethod node in graph')
        return
    metadata = selected.get('romMetadata')
    if not isinstance(metadata, dict) or not isinstance(metadata.get('methodName'), str) or metadata['methodName'].strip().lower() != request.get('method'):
        reject('workflow.graph.nodes', 'selected node methodName must match the requested method')
    elif not isinstance(metadata.get('parameters'), dict):
        reject('workflow.graph.nodes', 'selected node parameters must be an object')
    else:
        defaults = _parameterDefaults(request.get('method'))
        graphParameters = {**defaults, **metadata['parameters']}
        if not isinstance(request.get('parameters'), dict) or graphParameters != request['parameters'] or any(
                type(value) is not type(request['parameters'][key]) for key, value in graphParameters.items()
                if key in request['parameters'] and value is not None):
            reject('workflow.graph.nodes', 'selected node parameters must match the normalized task parameters')
    incoming = {}
    for index, connection in enumerate(graph['connections']):
        if not isinstance(connection, dict) or not isinstance(connection.get('from'), dict) or not isinstance(connection.get('to'), dict):
            reject(f'workflow.graph.connections[{index}]', 'connection requires from and to endpoints')
            return
        source = connection['from'].get('nodeUuid')
        target = connection['to'].get('nodeUuid')
        if any(not isinstance(connection[end].get('portId'), str) or not connection[end]['portId'].strip()
               for end in ('from', 'to')):
            reject(f'workflow.graph.connections[{index}]', 'connection endpoints require nonempty portId strings')
            return
        if not isinstance(source, str) or not isinstance(target, str) or source not in nodes or target not in nodes or source == target:
            reject(f'workflow.graph.connections[{index}]', 'connection endpoints must reference distinct existing nodes')
            return
        if target == request.get('nodeId') and (
                connection['to']['portId'] != 'trainingData' or
                (nodes[source].get('typeId') == 'rom.dataInput' and connection['from']['portId'] != 'data')):
            reject(f'workflow.graph.connections[{index}]', 'direct dataInput must connect data to trainingData')
            return
        incoming.setdefault(target, []).append(source)
    parents = incoming.get(request.get('nodeId'), [])
    if len(parents) > 1:
        reject('workflow.graph.connections', 'selectedNode supports at most one direct data input')
    for parent in parents:
        if nodes[parent].get('typeId') != 'rom.dataInput' or incoming.get(parent):
            reject('workflow.graph.connections', 'only a direct dataInput is supported; upstream transforms and split nodes are not executed')


def _absolutePath(value, field, reject):
    if not isinstance(value, str) or not value or '\x00' in value:
        reject(field, 'must be an absolute filesystem path')
        return None
    path = Path(value)
    if not path.is_absolute() or '..' in path.parts:
        reject(field, 'must be an absolute path without parent traversal')
        return None
    try:
        if field == 'outputDirectory' and path.is_symlink():
            reject(field, 'output directory must not be a symbolic link')
            return None
        return path.resolve()
    except (OSError, ValueError, RuntimeError) as error:
        reject(field, f'invalid path: {error}')
        return None

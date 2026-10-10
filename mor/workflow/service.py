import json
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .storage import atomicJson, jsonCopy
from .validation import API_VERSION, getCapabilities, validateWorkflow

_taskLock = threading.Lock()
_activeTask = None
_usedTaskIds = set()


class _Cancelled(Exception):
    pass


class _ObserverError(Exception):
    pass


class _Run:
    def __init__(self, request, onEvent, isCancelled):
        self.request = request
        self.onEvent = onEvent
        self.isCancelled = isCancelled
        self.directory = None
        self.sequence = 0
        self.stage = 'validation'

    def emit(self, eventType, payload):
        if self.request.get('eventPolicy') == 'finalOnly' and not (eventType == 'state' and payload.get('state') in ('succeeded', 'failed', 'cancelled')):
            return
        self.sequence += 1
        event = {
            'apiVersion': API_VERSION, 'taskId': self.request.get('taskId', ''),
            'nodeId': self.request.get('nodeId', ''), 'sequence': self.sequence,
            'timestamp': datetime.now(timezone.utc).isoformat(), 'type': eventType, 'payload': payload,
        }
        if self.directory is not None:
            with (self.directory / 'events.jsonl').open('a', encoding='utf-8', newline='\n') as stream:
                stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
        if self.onEvent is not None:
            try:
                self.onEvent(jsonCopy(event))
            except Exception as error:
                self.onEvent = None
                raise _ObserverError(f'event observer failed: {error}') from error

    def setStage(self, stage):
        self.stage = stage
        self.emit('state', {'state': 'running', 'stage': stage})

    def log(self, message, level='info'):
        if self.request['parameters']['verbose']:
            if self.request.get('eventPolicy') == 'finalOnly' and self.directory is not None:
                with (self.directory / 'calculation.log').open('a', encoding='utf-8') as stream:
                    stream.write(f'[{level}] {self.stage}: {message}\n')
                return
            self.emit('log', {'level': level, 'message': message, 'stage': self.stage})

    def checkpoint(self):
        if self.isCancelled is not None:
            try:
                cancelled = bool(self.isCancelled())
            except Exception as error:
                raise _ObserverError(f'cancellation observer failed: {error}') from error
            if cancelled:
                raise _Cancelled('Task cancellation was acknowledged at a cooperative checkpoint.')

    def terminal(self, state, error=None, summary=None, modelReference=None):
        return {
            'apiVersion': API_VERSION, 'taskId': self.request.get('taskId', ''),
            'nodeId': self.request.get('nodeId', ''), 'state': state,
            'modelReference': modelReference, 'files': self.files(),
            'error': error, 'summary': summary or {},
        }

    def files(self):
        if self.directory is None:
            return []
        try:
            return sorted(str(path) for path in self.directory.iterdir() if path.is_file() and not path.name.startswith('.'))
        except OSError:
            return []

    def finish(self, result):
        warnings = []
        try:
            self.emit('state', {'state': result['state'], 'stage': self.stage, 'error': result['error']})
        except Exception as error:
            warnings.append(f'Terminal event could not be delivered or persisted: {error}')
        if self.directory is not None:
            try:
                result['files'] = sorted(set(self.files() + [str(self.directory / 'result.json')]))
            except OSError as error:
                warnings.append(f'Could not enumerate task artifacts: {error}')
        if warnings:
            result['summary']['warnings'] = result['summary'].get('warnings', []) + warnings
        if self.directory is not None:
            try:
                atomicJson(self.directory / 'result.json', result)
            except Exception as error:
                result['summary'].setdefault('warnings', []).append(f'Could not persist final diagnostics: {error}')
        return result


def executeWorkflow(request, onEvent=None, isCancelled=None):
    global _activeTask

    validation = validateWorkflow(request)
    identity = {key: request.get(key, '') if isinstance(request, dict) and isinstance(request.get(key), str) else ''
                for key in ('taskId', 'nodeId')}
    run = _Run(validation['request'] or identity, onEvent if callable(onEvent) else None,
               isCancelled if callable(isCancelled) else None)
    if onEvent is not None and not callable(onEvent) or isCancelled is not None and not callable(isCancelled):
        return run.finish(run.terminal('failed', _error('invalid_callback', 'callbacks must be callable or None', run.stage)))
    if not validation['valid']:
        error = _error('invalid_request', '; '.join(item['message'] for item in validation['errors']), run.stage)
        error['details'] = validation['errors']
        return run.finish(run.terminal('failed', error))

    taskId = run.request['taskId']
    run.stage = 'admission'
    admissionError = None
    with _taskLock:
        if taskId in _usedTaskIds:
            admissionError = _error('duplicate_task', 'Every execution must use a new taskId.', 'admission')
        elif _activeTask is not None:
            admissionError = _error('task_busy', 'Another workflow task is already running.', 'admission')
        else:
            _activeTask = taskId
    if admissionError is not None:
        return run.finish(run.terminal('failed', admissionError))

    try:
        return _executeAccepted(run, taskId)
    finally:
        with _taskLock:
            _activeTask = None


def _executeAccepted(run, taskId):
    try:
        directory = Path(run.request['outputDirectory'])
        directory.mkdir(exist_ok=False)
        run.directory = directory
        with _taskLock:
            _usedTaskIds.add(taskId)
        atomicJson(directory / 'submittedRequest.json', run.request)
        atomicJson(directory / 'request.json', run.request)
        run.setStage('preparing')
        run.checkpoint()
        if run.request['method'] == 'mock':
            manifest, summary = _executeMock(run)
        elif run.request['method'] == 'bt':
            from .dynamics import prepareSystem, reduceBalanced

            run.setStage('loading')
            matrices = prepareSystem(run.request, directory, run.checkpoint)
            atomicJson(directory / 'request.json', run.request)
            manifest, summary = reduceBalanced(run.request, matrices, directory, run.checkpoint, run.setStage, run.log)
        else:
            from .pod import prepareSnapshots

            run.setStage('loading')
            data, frozenInput = prepareSnapshots(run.request, directory, run.checkpoint)
            run.request['inputs']['snapshots'] = frozenInput
            atomicJson(directory / 'request.json', run.request)
            if run.request['method'] == 'pod':
                from .pod import reducePod

                manifest, summary = reducePod(run.request, data, directory, run.checkpoint, run.setStage, run.log)
            elif run.request['method'] in ('dmd', 'pod-dmd'):
                from .dynamics import reduceDynamics

                manifest, summary = reduceDynamics(run.request, data, directory, run.checkpoint, run.setStage, run.log)
            else:
                from .autoencoder import reduceAutoencoder

                manifest, summary = reduceAutoencoder(
                    run.request, data, directory, run.checkpoint, run.setStage, run.log,
                    lambda metrics: run.emit('metrics', metrics),
                )
            if summary['reconstructionError'] is not None:
                run.emit('metrics', {'reconstructionError': summary['reconstructionError'],
                                     'actualRank': summary['actualRank']})
        run.setStage('committing')
        manifest['inputKind'] = run.request.get('inputKind', 'snapshots')
        if 'datasetProvenance' in run.request:
            manifest['datasetProvenance'] = run.request['datasetProvenance']
        if run.request['method'] != 'mock':
            run.setStage('exporting')
            from .onnxExport import exportInference

            try:
                manifest['inference'] = exportInference(manifest, directory)
            except ValueError as error:
                manifest['inference'] = {'available': False, 'reason': str(error)}
                summary.setdefault('warnings', []).append(f'ONNX validation unavailable: {error}')
        run.checkpoint()
        result = run.terminal('succeeded', summary=summary, modelReference=str(directory / 'manifest.json'))
        result['files'] = sorted(set(result['files'] + [str(directory / 'manifest.json'),
                                                      str(directory / 'result.json')]))
        atomicJson(directory / 'result.json', result)
        # The manifest is the commit marker. Cancellation after this checkpoint cannot revoke a model.
        atomicJson(directory / 'manifest.json', manifest)
    except _Cancelled as error:
        result = run.terminal('cancelled', _error('cancelled', str(error), run.stage))
    except Exception as error:
        code = 'observer_error' if isinstance(error, _ObserverError) else 'storage_error' if isinstance(error, OSError) else 'execution_error'
        result = run.terminal('failed', _error(code, str(error), run.stage, traceback.format_exc()))
    return run.finish(result)


def _executeMock(run):
    parameters = run.request['parameters']
    total = parameters['steps']
    run.setStage('running')
    for step in range(1, total + 1):
        run.checkpoint()
        deadline = time.monotonic() + parameters['delayMs'] / 1000
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(remaining, 0.02))
            run.checkpoint()
        if parameters['failAtStep'] == step:
            raise RuntimeError(f'Mock failure requested at step {step}.')
        run.log(f'Mock step {step}/{total}')
        run.emit('metrics', {'step': step, 'total': total, 'progress': step / total,
                             'epoch': step, 'loss': 1.0 / step, 'learningRate': 0.01})
    run.checkpoint()
    run.setStage('saving')
    summary = {'method': 'mock', 'completedSteps': total}
    return {
        'apiVersion': API_VERSION, 'modelVersion': 1, 'kind': 'mock',
        'taskId': run.request['taskId'], 'nodeId': run.request['nodeId'], 'summary': summary,
    }, summary


def _error(code, message, stage, trace=None):
    return {'code': code, 'message': message, 'stage': stage, 'traceback': trace}

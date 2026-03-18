import torch
from typing import Any
from mor.nn.trainers.baseTrainer import baseTrainer
from mor.nn.registry import nnRegistry

class autoEncoderTrainerTorch(baseTrainer):
    def __init__(self, lossFunction: Any, config: Any | None = None, **kwargs):
        super().__init__(trainerName='autoEncoderTrainerTorch', **kwargs)
        self.lossFunction, self.config, self._initOverrides = lossFunction, config, dict(kwargs)
        self.epochs = self._read('epochs', 100)
        self.opt = self._loadOptimizerConfig()
        self.sched = self._loadSchedulerConfig()
        self.run = self._loadRuntimeConfig()
        self.early = self._loadEarlyStoppingConfig()
        self.ckpt = self._loadCheckpointConfig()
        self.log = self._loadLoggingConfig()

    def _read(self, path: str, default: Any, overrideKey: str | None = None) -> Any:
        key = overrideKey or path.split('.')[-1]
        if key in self._initOverrides and self._initOverrides[key] is not None: return self._initOverrides[key]
        if self.config is None: return default
        val = self.config
        for part in path.split('.'):
            if not hasattr(val, part): return default
            val = getattr(val, part)
        return val

    def _loadOptimizerConfig(self) -> dict:
        name = self._read('optimizer.optimizerName', 'adam').lower()
        cfg = {'name': name, 'lr': self._read('optimizer.learningRate', 1e-3, 'learningRate'), 'weightDecay': self._read('optimizer.weightDecay', 0.0, 'weightDecay'), 'clip': self._read('optimizer.gradientClip', None, 'gradientClip'), 'params': {}}
        if name in ['adam', 'adamw']:
            cfg['params'] = {'betas': (self._read(f'optimizer.{name}.beta1', 0.9), self._read(f'optimizer.{name}.beta2', 0.999)), 'eps': self._read(f'optimizer.{name}.epsilon', 1e-8), 'amsgrad': self._read(f'optimizer.{name}.amsgrad', False)}
        elif name == 'sgd':
            cfg['params'] = {'momentum': self._read('optimizer.sgd.momentum', 0.9), 'dampening': self._read('optimizer.sgd.dampening', 0.0), 'nesterov': self._read('optimizer.sgd.nesterov', False)}
        elif name == 'rmsprop':
            cfg['params'] = {'alpha': self._read('optimizer.rmsprop.alpha', 0.99), 'eps': self._read('optimizer.rmsprop.epsilon', 1e-8), 'momentum': self._read('optimizer.rmsprop.momentum', 0.0), 'centered': self._read('optimizer.rmsprop.centered', False)}
        elif name == 'adagrad':
            cfg['params'] = {'lr_decay': self._read('optimizer.adagrad.lrDecay', 0.0), 'eps': self._read('optimizer.adagrad.epsilon', 1e-10), 'initial_accumulator_value': self._read('optimizer.adagrad.initialAccumulatorValue', 0.0)}
        return cfg

    def _loadSchedulerConfig(self) -> dict:
        return {'name': self._read('scheduler.schedulerName', 'none').lower(), 'stepSize': self._read('scheduler.stepSize', 10), 'gamma': self._read('scheduler.gamma', 0.1), 'milestones': self._read('scheduler.milestones', []), 'tMax': self._read('scheduler.tMax', 100), 'etaMin': self._read('scheduler.etaMin', 0.0)}

    def _loadRuntimeConfig(self) -> dict:
        return {'device': self._read('runtime.deviceName', 'cpu', 'deviceName'), 'fallback': self._read('runtime.deviceAutoFallback', True), 'dtype': self._read('runtime.dtypeName', 'float32')}

    def _loadEarlyStoppingConfig(self) -> dict:
        return {'enabled': self._read('earlyStopping.enabled', False, 'earlyStoppingEnabled'), 'patience': self._read('earlyStopping.patience', 10, 'earlyStoppingPatience'), 'delta': self._read('earlyStopping.delta', 0.0, 'earlyStoppingDelta'), 'metric': self._read('earlyStopping.monitorMetric', 'validationLoss'), 'mode': self._read('earlyStopping.modeName', 'min').lower(), 'warmup': self._read('earlyStopping.warmupEpochs', 0)}

    def _loadCheckpointConfig(self) -> dict:
        return {'metric': self._read('checkpoint.monitorMetric', 'validationLoss', 'checkpointMonitorMetric'), 'delta': self._read('checkpoint.monitorDelta', 0.0), 'mode': self._read('checkpoint.monitorModeName', 'min').lower(), 'saveBest': self._read('checkpoint.saveBestModel', True, 'saveBestModel'), 'saveLast': self._read('checkpoint.saveLastModel', False, 'saveLastModel')}

    def _loadLoggingConfig(self) -> dict:
        return {'verbose': self._read('logging.verbose', True), 'interval': self._read('logging.logInterval', 10), 'reportTrain': self._read('logging.reportTrainLoss', True), 'reportVal': self._read('logging.reportValidationLoss', True)}

    def _buildOptimizer(self, model: Any) -> Any:
        classes = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW, 'sgd': torch.optim.SGD, 'rmsprop': torch.optim.RMSprop, 'adagrad': torch.optim.Adagrad}
        if self.opt['name'] not in classes: raise ValueError(f"Unsupported optimizer: {self.opt['name']}")
        return classes[self.opt['name']](model.parameters(), lr=self.opt['lr'], weight_decay=self.opt['weightDecay'], **self.opt['params'])

    def _buildScheduler(self, optimizer: Any) -> Any:
        name = self.sched['name']
        if name == 'none': return None
        if name == 'step': return torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.sched['stepSize'], gamma=self.sched['gamma'])
        if name == 'multistep': return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.sched['milestones'] or [self.sched['stepSize']], gamma=self.sched['gamma'])
        if name == 'cosine': return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.sched['tMax'], eta_min=self.sched['etaMin'])
        return None

    def _resolveDtype(self) -> torch.dtype:
        return {'float64': torch.float64, 'float16': torch.float16, 'bfloat16': torch.bfloat16}.get(self.run['dtype'], torch.float32)

    def _toRuntimeTensor(self, data: Any, device: str, dtype: torch.dtype) -> torch.Tensor:
        return (data if isinstance(data, torch.Tensor) else torch.as_tensor(data)).to(device=device, dtype=dtype)

    def _runBatch(self, model: Any, inputs: Any, targets: Any, device: str, dtype: torch.dtype, optimizer: Any | None = None) -> float:
        inputs, targets = self._toRuntimeTensor(inputs, device, dtype), self._toRuntimeTensor(targets, device, dtype)
        if optimizer: optimizer.zero_grad()
        preds = model.forward(inputs)
        loss = self.lossFunction.compute(preds, targets)
        if optimizer:
            loss.backward()
            if self.opt['clip']: torch.nn.utils.clip_grad_norm_(model.parameters(), self.opt['clip'])
            optimizer.step()
        return float(loss.item())

    def _runEpoch(self, model: Any, loader: Any, dataModule: Any, device: str, dtype: torch.dtype, optimizer: Any | None = None, globalStep: int = 0) -> tuple[float, int]:
        if optimizer: model.trainMode()
        else: model.evalMode()
        if loader:
            losses = []
            for x, y in loader:
                losses.append(self._runBatch(model, x, y, device, dtype, optimizer))
                if optimizer:
                    globalStep += 1
                    if hasattr(model, 'updateTrainingState'): model.updateTrainingState(currentStep=globalStep)
            return sum(losses) / max(len(losses), 1), globalStep
        x, y = dataModule.getTrainData() if optimizer else dataModule.getValidationData()
        loss = self._runBatch(model, x, y, device, dtype, optimizer)
        if optimizer:
            globalStep += 1
            if hasattr(model, 'updateTrainingState'): model.updateTrainingState(currentStep=globalStep)
        return loss, globalStep

    def fit(self, model: Any, dataModule: Any) -> dict:
        device = model.toDevice(self.run['device'], allowAutoFallback=self.run['fallback']) if hasattr(model, 'toDevice') else self.run['device']
        dtype, optimizer = self._resolveDtype(), self._buildOptimizer(model)
        scheduler, globalStep = self._buildScheduler(optimizer), 0
        trainLoader, valLoader = dataModule.getTrainLoader(), dataModule.getValidationLoader()
        history = {'train': [], 'val': [], 'lr': []}
        best = {'val': None, 'epoch': None, 'state': None}
        es = {'best': None, 'patience': 0}
        if hasattr(model, 'updateTrainingState'): model.updateTrainingState(currentEpoch=0, currentStep=0, isTraining=False)
        for epoch in range(self.epochs):
            if hasattr(model, 'updateTrainingState'): model.updateTrainingState(currentEpoch=epoch, isTraining=True)
            trainLoss, globalStep = self._runEpoch(model, trainLoader, dataModule, device, dtype, optimizer, globalStep)
            with torch.no_grad(): valLoss, _ = self._runEpoch(model, valLoader, dataModule, device, dtype)
            if scheduler: scheduler.step()
            history['train'].append(trainLoss); history['val'].append(valLoss); history['lr'].append(float(optimizer.param_groups[0]['lr']))
            metric = valLoss if self.ckpt['metric'] == 'validationLoss' else trainLoss
            if self._isBetter(metric, best['val'], self.ckpt['delta'], self.ckpt['mode']):
                best.update({'val': metric, 'epoch': epoch})
                if self.ckpt['saveBest']: best['state'] = model.saveState(toCpu=True)
            self._logEpoch(epoch, trainLoss, valLoss, history['lr'][-1])
            esMetric = valLoss if self.early['metric'] == 'validationLoss' else trainLoss
            if self._isBetter(esMetric, es['best'], self.early['delta'], self.early['mode']): es.update({'best': esMetric, 'patience': 0})
            elif epoch >= self.early['warmup']: es['patience'] += 1
            if self.early['enabled'] and es['patience'] >= self.early['patience']: break
        if best['state'] and self.ckpt['saveBest']: model.loadState(best['state'])
        lastState = model.saveState(toCpu=True) if self.ckpt['saveLast'] else None
        if hasattr(model, 'updateTrainingState'): model.updateTrainingState(currentEpoch=epoch, currentStep=globalStep, isTraining=False)
        return {'trainLossHistory': history['train'], 'validationLossHistory': history['val'], 'learningRateHistory': history['lr'], 'bestMetricValue': best['val'], 'bestEpochIndex': best['epoch'], 'bestModelState': best['state'], 'lastModelState': lastState, 'resolvedDeviceName': device, 'totalSteps': globalStep}

    def _isBetter(self, current: float, best: float | None, delta: float, mode: str) -> bool:
        if best is None: return True
        return current < (best - delta) if mode == 'min' else current > (best + delta)

    def _logEpoch(self, epoch: int, trainLoss: float, valLoss: float, lr: float):
        if not self.log['verbose']: return
        if (epoch + 1) % self.log['interval'] == 0 or epoch == 0 or epoch + 1 == self.epochs:
            print(f"epoch={epoch+1}/{self.epochs} | lr={lr:.6g} | train={trainLoss:.6g} | val={valLoss:.6g}")

nnRegistry.register('trainers.stateReconstruction', 'autoEncoderTrainerTorch', autoEncoderTrainerTorch)

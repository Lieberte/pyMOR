from typing import Any
import torch
from dataclasses import asdict, is_dataclass
from mor.nn.configs import representationLearningConfig
import mor.nn.data.representationLearning
import mor.nn.models.representationLearning
import mor.nn.losses.representationLearning
import mor.nn.trainers.representationLearning
import mor.nn.validation.representationLearning
from mor.nn.registry import nnRegistry

def runRepresentationLearningTorch(config: representationLearningConfig, inputs: Any, targets: Any | None = None, validationInputs: Any | None = None, validationTargets: Any | None = None, initialModelState: dict | None = None, returnModelState: bool = False) -> dict:
    if config.runtime.randomSeed is not None:
        torch.manual_seed(config.runtime.randomSeed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.runtime.randomSeed)
    if config.runtime.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    dataModuleClass = nnRegistry.get('data.representationLearning', config.dataModuleName)
    modelClass = nnRegistry.get('models.representationLearning', config.modelName)
    lossClass = nnRegistry.get('losses.representationLearning', config.lossFunction)
    trainerClass = nnRegistry.get('trainers.representationLearning', config.trainerName)
    validationClass = nnRegistry.get('validation.representationLearning', config.validationName)
    dataLoaderParams = asdict(config.dataLoader) if is_dataclass(config.dataLoader) else {}
    valSplit = dataLoaderParams.pop('validationSplit', 0.2)
    if validationInputs is None:
        dataModule = dataModuleClass.fromSnapshots(
            inputs=inputs,
            targets=targets,
            validationSplit=valSplit,
            **dataLoaderParams
        )
    else:
        dataModule = dataModuleClass(
            trainInputs=inputs,
            validationInputs=validationInputs,
            trainTargets=targets,
            validationTargets=validationTargets,
            **dataLoaderParams
        )
    baseConfigFields = {'name', 'options', 'epochs', 'earlyStopping', 'logging', 'optimizer', 'scheduler', 'dataLoader', 'runtime', 'checkpoint', 'modelName', 'trainerName', 'lossFunction', 'validationName', 'dataModuleName'}
    modelParams = {k: v for k, v in asdict(config).items() if k not in baseConfigFields}
    model = modelClass(**modelParams)
    resolvedDeviceName = model.toDevice(config.runtime.deviceName, allowAutoFallback=config.runtime.deviceAutoFallback) if hasattr(model, 'toDevice') else config.runtime.deviceName
    if initialModelState is not None and hasattr(model, 'loadState'):
        model.loadState(initialModelState)
    lossFunction = lossClass(**config.options) if config.options else lossClass()
    trainer = trainerClass(lossFunction=lossFunction, config=config)
    trainingResult = trainer.fit(model=model, dataModule=dataModule)
    validator = validationClass(config=config, lossFunction=lossFunction)
    validationResult = validator.evaluate(model=model, dataModule=dataModule)
    result = {
        'config': config,
        'model': model,
        'dataModule': dataModule,
        'trainingResult': trainingResult,
        'validationResult': validationResult,
        'runtimeInfo': {
            'backendName': config.runtime.backendName,
            'requestedDeviceName': config.runtime.deviceName,
            'resolvedDeviceName': trainingResult.get('resolvedDeviceName', resolvedDeviceName),
            'dtypeName': config.runtime.dtypeName,
            'deterministic': config.runtime.deterministic,
            'deviceAutoFallback': config.runtime.deviceAutoFallback
        },
        'trainingState': model.getTrainingState() if hasattr(model, 'getTrainingState') else {}
    }
    if returnModelState and hasattr(model, 'saveState'):
        result['modelState'] = model.saveState(toCpu=True)
    return result

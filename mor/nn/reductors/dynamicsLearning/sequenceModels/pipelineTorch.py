from typing import Any
import torch
from mor.nn.configs import dynamicsLearningConfig
from mor.nn.registry import nnRegistry

def runDynamicsLearningTorch(config: dynamicsLearningConfig, inputs: Any, targets: Any, validationInputs: Any | None = None, validationTargets: Any | None = None, initialModelState: dict | None = None, returnModelState: bool = False) -> dict:
    if config.runtime.randomSeed is not None:
        torch.manual_seed(config.runtime.randomSeed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.runtime.randomSeed)
    dataModuleClass = nnRegistry.get('data.dynamicsLearning', config.dataModuleName)
    modelClass = nnRegistry.get('models.dynamicsLearning', config.modelName)
    lossClass = nnRegistry.get('losses.dynamicsLearning', config.lossFunction)
    trainerClass = nnRegistry.get('trainers.dynamicsLearning', config.trainerName)
    validationClass = nnRegistry.get('validation.dynamicsLearning', config.validationName)
    if validationInputs is None:
        dataModule = dataModuleClass.fromSnapshots(inputs=inputs, targets=targets, **config.dataParams) if hasattr(dataModuleClass, 'fromSnapshots') else dataModuleClass(inputs, inputs, targets, targets, **config.dataParams)
    else:
        dataModule = dataModuleClass(trainInputs=inputs, validationInputs=validationInputs, trainTargets=targets, validationTargets=validationTargets, **config.dataParams)
    model = modelClass(**config.modelParams)
    resolvedDeviceName = model.toDevice(config.runtime.deviceName, allowAutoFallback=config.runtime.deviceAutoFallback) if hasattr(model, 'toDevice') else config.runtime.deviceName
    if initialModelState is not None and hasattr(model, 'loadState'):
        model.loadState(initialModelState)
    lossFunction = lossClass(**config.lossParams) if config.lossParams else lossClass()
    trainer = trainerClass(lossFunction=lossFunction, config=config)
    trainingResult = trainer.fit(model=model, dataModule=dataModule)
    validator = validationClass(config=config, lossFunction=lossFunction, **config.validationParams)
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

from typing import Any
import torch
from mor.nn.configs import stateReconstructionConfig
import mor.nn.data.stateReconstruction
import mor.nn.models.stateReconstruction
import mor.nn.losses.stateReconstruction
import mor.nn.trainers.stateReconstruction
import mor.nn.validation.stateReconstruction
from mor.nn.registry import nnRegistry

def runStateReconstructionTorch(config: stateReconstructionConfig, inputs: Any, targets: Any | None = None, validationInputs: Any | None = None, validationTargets: Any | None = None, initialModelState: dict | None = None, returnModelState: bool = False) -> dict:
    if config.runtime.randomSeed is not None:
        torch.manual_seed(config.runtime.randomSeed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.runtime.randomSeed)
    if config.runtime.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    dataModuleClass = nnRegistry.get('data.stateReconstruction', config.dataModuleName)
    modelClass = nnRegistry.get('models.stateReconstruction', config.modelName)
    lossClass = nnRegistry.get('losses.stateReconstruction', config.lossFunction)
    trainerClass = nnRegistry.get('trainers.stateReconstruction', config.trainerName)
    validationClass = nnRegistry.get('validation.stateReconstruction', config.validationName)
    if validationInputs is None:
        dataModule = dataModuleClass.fromSnapshots(
            inputs=inputs,
            targets=targets,
            validationSplit=config.dataLoader.validationSplit,
            shuffle=config.dataLoader.shuffle,
            randomSeed=config.runtime.randomSeed,
            batchSize=config.dataLoader.batchSize,
            numWorkers=config.dataLoader.numWorkers,
            pinMemory=config.dataLoader.pinMemory,
            dropLast=config.dataLoader.dropLast,
            persistentWorkers=config.dataLoader.persistentWorkers
        )
    else:
        dataModule = dataModuleClass(
            trainInputs=inputs,
            validationInputs=validationInputs,
            trainTargets=targets,
            validationTargets=validationTargets,
            batchSize=config.dataLoader.batchSize,
            shuffle=config.dataLoader.shuffle,
            numWorkers=config.dataLoader.numWorkers,
            pinMemory=config.dataLoader.pinMemory,
            dropLast=config.dataLoader.dropLast,
            persistentWorkers=config.dataLoader.persistentWorkers
        )
    model = modelClass(
        inputDim=config.inputDim,
        latentDim=config.latentDim,
        hiddenDims=config.hiddenDims
    )
    resolvedDeviceName = model.toDevice(config.runtime.deviceName, allowAutoFallback=config.runtime.deviceAutoFallback) if hasattr(model, 'toDevice') else config.runtime.deviceName
    if initialModelState is not None and hasattr(model, 'loadState'):
        model.loadState(initialModelState)
    lossFunction = lossClass()
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

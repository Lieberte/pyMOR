from typing import Any
from mor.nn.configs import representationLearningConfig
import mor.nn.data
import mor.nn.models
import mor.nn.losses
import mor.nn.trainers
import mor.nn.validation
from mor.nn.registry import nnRegistry
from mor.nn.runtime.trainingContext import configuredTraining
from .pipelineConfig import resolveTrainerName as _resolveTrainerName
from .pipelineConfig import validateTrainingModeConfig as _validateTrainingModeConfig


@configuredTraining
def runRepresentationLearningTorch(config: representationLearningConfig, inputs: Any, targets: Any | None = None, validationInputs: Any | None = None, validationTargets: Any | None = None, initialModelState: dict | None = None, returnModelState: bool = False) -> dict:
    _validateTrainingModeConfig(config)
    dataModuleClass = nnRegistry.get('data', config.dataModuleName)
    modelClass = nnRegistry.get('models', config.modelName)
    lossClass = nnRegistry.get('losses', config.lossFunction)
    trainerClass = nnRegistry.get('trainers', _resolveTrainerName(config, defaultTrainerName='autoEncoderTrainerTorch'))
    validationClass = nnRegistry.get('validation', config.validationName)

    if validationInputs is None:
        dataModule = dataModuleClass.fromSnapshots(inputs=inputs, targets=targets, **config.dataParams)
    else:
        dataModule = dataModuleClass(trainInputs=inputs, validationInputs=validationInputs, trainTargets=targets, validationTargets=validationTargets, **config.dataParams)

    model = modelClass(**config.modelParams)
    
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
            'resolvedDeviceName': trainingResult.get('resolvedDeviceName'),
            'dtypeName': config.runtime.dtypeName,
            'deterministic': config.runtime.deterministic,
            'deviceAutoFallback': config.runtime.deviceAutoFallback,
        },
        'trainingState': model.getTrainingState() if hasattr(model, 'getTrainingState') else {}
    }
    
    if returnModelState and hasattr(model, 'saveState'):
        result['modelState'] = model.saveState(toCpu=True)
    return result

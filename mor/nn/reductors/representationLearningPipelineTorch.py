from typing import Any
import torch
from mor.nn.configs import representationLearningConfig
import mor.nn.data
import mor.nn.models
import mor.nn.losses
import mor.nn.trainers
import mor.nn.validation
from mor.nn.registry import nnRegistry

def runRepresentationLearningTorch(config: representationLearningConfig, inputs: Any, targets: Any | None = None, validationInputs: Any | None = None, validationTargets: Any | None = None, initialModelState: dict | None = None, returnModelState: bool = False) -> dict:
    if config.runtime.randomSeed is not None:
        torch.manual_seed(config.runtime.randomSeed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.runtime.randomSeed)
            
    dataModuleClass = nnRegistry.get('data', config.dataModuleName)
    modelClass = nnRegistry.get('models', config.modelName)
    lossClass = nnRegistry.get('losses', config.lossFunction)
    trainerClass = nnRegistry.get('trainers', config.trainerName)
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
        'trainingState': model.getTrainingState() if hasattr(model, 'getTrainingState') else {}
    }
    
    if returnModelState and hasattr(model, 'saveState'):
        result['modelState'] = model.saveState(toCpu=True)
    return result

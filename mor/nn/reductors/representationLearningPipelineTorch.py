from typing import Any
import torch
from mor.nn.configs import representationLearningConfig
import mor.nn.data
import mor.nn.models
import mor.nn.losses
import mor.nn.trainers
import mor.nn.validation
from mor.nn.registry import nnRegistry


def _resolveTrainerName(config: representationLearningConfig, defaultTrainerName: str) -> str:
    if config.trainerName:
        return config.trainerName
    return 'operatorTrainerTorch' if config.trainingMode == 'operator' else defaultTrainerName


def _validateTrainingModeConfig(config: representationLearningConfig) -> None:
    if config.trainingMode not in ('point', 'operator'):
        raise ValueError(f'unsupported trainingMode: {config.trainingMode}')
    if config.trainingMode != 'operator':
        return
    requiredCellTypes = list(config.options.get('requiredCellTypes', []))
    supportedCellTypes = set(config.options.get('supportedCellTypes', []))
    if requiredCellTypes and not supportedCellTypes:
        raise ValueError('operator mode requires options.supportedCellTypes when requiredCellTypes is set')
    missingCellTypes = [cellType for cellType in requiredCellTypes if cellType not in supportedCellTypes]
    if missingCellTypes:
        raise ValueError(f'unsupported operator cell types: {missingCellTypes}')


def runRepresentationLearningTorch(config: representationLearningConfig, inputs: Any, targets: Any | None = None, validationInputs: Any | None = None, validationTargets: Any | None = None, initialModelState: dict | None = None, returnModelState: bool = False) -> dict:
    _validateTrainingModeConfig(config)
    if config.runtime.randomSeed is not None:
        torch.manual_seed(config.runtime.randomSeed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.runtime.randomSeed)
            
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
        'trainingState': model.getTrainingState() if hasattr(model, 'getTrainingState') else {}
    }
    
    if returnModelState and hasattr(model, 'saveState'):
        result['modelState'] = model.saveState(toCpu=True)
    return result

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
            
    dataModuleClass = nnRegistry.get('data.representationLearning', config.dataModuleName)
    modelClass = nnRegistry.get('models.representationLearning', config.modelName)
    lossClass = nnRegistry.get('losses.representationLearning', config.lossFunction)
    trainerClass = nnRegistry.get('trainers.representationLearning', config.trainerName)
    validationClass = nnRegistry.get('validation.representationLearning', config.validationName)

    # 1. 实例化数据模块 (透传 dataParams)
    if validationInputs is None:
        dataModule = dataModuleClass.fromSnapshots(inputs=inputs, targets=targets, **config.dataParams)
    else:
        dataModule = dataModuleClass(trainInputs=inputs, validationInputs=validationInputs, trainTargets=targets, validationTargets=validationTargets, **config.dataParams)

    # 2. 实例化模型 (透传 modelParams)
    model = modelClass(**config.modelParams)
    
    if initialModelState is not None and hasattr(model, 'loadState'):
        model.loadState(initialModelState)
        
    # 3. 实例化损失函数 (透传 lossParams)
    lossFunction = lossClass(**config.lossParams) if config.lossParams else lossClass()
    
    # 4. 实例化训练器
    trainer = trainerClass(lossFunction=lossFunction, config=config)
    trainingResult = trainer.fit(model=model, dataModule=dataModule)
    
    # 5. 实例化验证器 (透传 validationParams)
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

from mor.nn.configs.base import baseConfig


def resolveTrainerName(config: baseConfig, defaultTrainerName: str) -> str:
    if config.trainerName:
        return config.trainerName
    return 'operatorTrainerTorch' if config.trainingMode == 'operator' else defaultTrainerName


def validateTrainingModeConfig(config: baseConfig) -> None:
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

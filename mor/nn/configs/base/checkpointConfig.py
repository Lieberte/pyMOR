from dataclasses import dataclass

@dataclass
class checkpointConfig:
    saveBestModel: bool = True
    saveLastModel: bool = False
    monitorMetric: str = 'validationLoss'
    monitorDelta: float = 0.0
    monitorModeName: str = 'min'
    saveDirName: str = 'checkpoints'
    filePrefix: str = 'stateReconstruction'

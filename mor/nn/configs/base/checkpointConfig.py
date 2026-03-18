from dataclasses import dataclass

@dataclass
class checkpointConfig:
    saveBestModel: bool = True
    saveLastModel: bool = False
    monitorMetric: str = 'validationLoss'

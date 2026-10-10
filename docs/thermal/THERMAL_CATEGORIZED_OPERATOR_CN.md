# 分类热算子与推理契约

`thermalCategorizedOperatorTorch` 面向变长 FEM 点集，使用 Global 模型和按 `componentType` 路由的专家。网络结构、专家集合和融合权重由 checkpoint 提供；源码不附带已训练模型，也不存在固定的专家数量或默认实验精度。

## 模型与路由

```text
同一父工况的元件点集
  → 装配级热源、边界、温度锚点与接口上下文
  → Global 模型与 componentType 专家
  → checkpoint 中的融合权重
  → 查询点温度或温差
```

路由键为 `componentType`，目录或实例名称不参与分类。无可用专家时使用 Global。checkpoint 可以声明 `supportedComponentTypes`；runtime 先利用全部输入准备上下文，再只输出支持的元件，输入中没有任何支持类型时明确报错。

Set-Attention、Set-DeepONet 和 Query-Set 是可选择的点集结构。它们不要求规则网格，不能把点集 decoder 称作规则网格卷积 U-Net，也不能用人为近邻图冒充原始 FEM 连接。

## 输入与温度参考

Case 物理数组见 [thermal_component_case_format.md](../data/thermal_component_case_format.md)。装配上下文模型还需要 `assemblyId/motorId`、`parentCaseId`、`regionUuid` 与 `surfacePatches` 中的接口和对侧 region 信息。使用此类 checkpoint 时，按完整父工况批量提供可获得的元件。

输入包括体域材料、热源、体积权重，边界法向、HTC、环境温度、热流、边界类型、面积权重、patch，以及工况和查询坐标。推理只使用当时可知的物理量，禁止把温度目标作为模型特征。

查询位置由 `temperature.coordinatesMeters` 承载。有绝对边界锚点时 `temperatureK` 标签不进入输入，可填占位值。参考温度优先使用固定温度边界，否则使用 HTC×面积加权环境温度，并可由完整装配传播到局部元件。

没有绝对温度锚点时，`predictCases` 默认拒绝绝对温度输出。显式 `requireAbsoluteTemperature=False` 可请求温差；此时无锚点结果的 `temperatureK/temperatureReferenceK` 为 None。尺度、边界编号和特征顺序从 checkpoint 恢复，维度不匹配直接报错。无量纲规则见 [thermal_dimensionless_v2.md](thermal_dimensionless_v2.md)。

## 使用方法

示例数据和模型需自行生成或提供，路径以仓库根目录为基准。

```python
from mor.nn.data import loadSubsampledThermalDataset
from mor.nn.runtime import thermalCategorizedPredictor

dataset = loadSubsampledThermalDataset(
    'dataset/thermal_components/dataset.json',
    analysisType='steady',
    maximumVolumeSamples=128,
    maximumBoundarySamples=256,
    maximumQuerySamples=128,
    randomSeed=23,
)
predictor = thermalCategorizedPredictor.fromCheckpoint(
    'reports/thermal/model.pt', device='cpu',
)
predictions = predictor.predictCases(dataset.cases)
```

返回项包含 caseId、componentType、查询坐标、温度/温差、参考温度、温标、绝对温度可用性和实际路由信息。

```powershell
python scripts/evaluate_thermal_heterogeneous_checkpoint.py dataset/thermal_components/dataset.json --checkpoint reports/thermal/model.pt --output reports/thermal/evaluation.json --device cpu --seed 23
```

## 评价边界

专家选择、融合权重标定与最终评估应分开，按父工况或独立几何划分。训练验证集同时用于选模和报分会使结果偏乐观；历史小样本最优结构与权重不应直接固化到接口说明。

求解结果导出的 Neumann 热流属于条件输入，评价的是给定边界后的场重构。未知工况、未知接口整机预测和未见几何泛化需要独立证据。FEM 物理组装与求解见 [thermal_physical_operator.md](thermal_physical_operator.md)。

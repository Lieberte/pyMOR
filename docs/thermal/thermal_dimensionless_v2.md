# 稳态热算子无量纲 V2

V2 保留原始 FEM 物理数组和 V1 checkpoint 读取能力，新增面向混合边界与跨几何训练的表示和训练链路。

## 数据与参考温度

- `thermalComponentFeatureSchema.fixedTemperatureBoundaryTypeIds` 显式声明 Dirichlet 类型。
- 存在 Dirichlet 时，`T_ref` 只由 Dirichlet 节点按面积加权确定。
- 不存在 Dirichlet、但存在 Robin 时，`T_ref` 使用 `hA` 加权环境温度。
- 纯 Neumann case 只监督体积加权零均值相对温度。

训练新数据集时应统一边界类型编号。例如：

```python
schema = thermalComponentFeatureSchema(
    operatingFeatureNames=('speedRpm', 'absoluteTemperatureAnchor'),
    boundaryTypeCount=3,
    temperatureBoundaryTypeIds=(1, 2),
    fixedTemperatureBoundaryTypeIds=(2,),
)
```

## V2 全局物理特征

V2 在逐点体域/边界特征之外增加：

- 无量纲体积、表面积和长宽高比例；
- 体热源总功率、边界热流总功率、净功率；
- 无量纲对流热导；
- 三个温差候选尺度相对于 `DeltaT_c` 的比例；
- 热源与边界热流的无量纲中心和空间扩展；
- 每种边界类型的面积占比和面积中心；
- Dirichlet/Robin 面积占比及边界温度统计量；
- 工况特征。

特征名称可通过 `steadyThermalDimensionlessV2GlobalFeatureNames(schema)` 获取。

## 网络与目标

`thermalQuerySetOperatorTorch` 的 V2 配置启用：

- 查询点到体域/边界点的相对位置和距离特征；
- 按边界类型分别计算局部核上下文；
- 标量平均温升头与局部温度形状头。

有温度锚点时预测：

```text
dimensionless temperature = zero-mean shape + mean offset
```

纯 Neumann case 不添加平均温升头，只保留相对场。

## 损失

`thermalMeanShapeEnergyLoss` 包含：

- 零均值场形状误差；
- 有锚点 case 的平均温升误差；
- 形状头零均值 gauge 约束；
- 完整温度重构误差；
- 可选热点加权；
- 无 Dirichlet、存在 Robin 时的整体能量守恒误差。

能量项不用于带 Dirichlet 的 case，因为固定温度边界反力热流在当前数据接口中未知。

推理查询点与 Dirichlet 边界点重合时，可使用 `enforceMatchingDirichletQueries` 进行硬回填。

## 物理相似性增强

`similarityAugmentedThermalComponentCase` 和 `randomSimilarityAugmentations` 按热方程同步缩放几何、导热率、温差、体热源、边界热流和 HTC。该增强保持无量纲 case 不变，可强化跨尺寸和整体材料尺度稳定性，但不能生成新的边界拓扑。

## 训练入口

下面的输入目录和附加数据集由调用方提供，输出为运行生成产物。

```powershell
python scripts/run_motor_fem_dimensionless_holdout.py `
  "input\motor_cases" `
  --additional-datasets dataset\additional_components `
  --similarity-augmentations-per-case 1 `
  --energy-balance-weight 0.02 `
  --checkpoint reports\thermal\model.pt `
  --output reports\thermal\evaluation.json
```

`--additional-datasets` 会自动识别以下输入，不需要再手工展开：

- 标准 `thermalComponentDataset`（根目录含 `dataset.json`）；
- `build_motor_directory_dataset.py` 或 `build_motor_sta_dataset.py` 的输出根目录；
- 直接传入 `whole_motor` 目录或其 `dataset.json`。

整机视图会展开为逐 FEM 元件训练样本，同时保留 `motorId` 和
`parentCaseId` 分组。数据集里声明的 `resultHeatFlux`、`convection`、
`fixedTemperature` 会按物理名称映射到统一边界编号，因此不同转换器的
局部编号可以不同，但物理名称必须完整且明确。

新增数据集应重点扩展 `Bi`、无量纲热源/热流、边界面积占比、边界空间位置和锚点类型的联合分布。无量纲化不会替代这些物理组合的训练覆盖。

若 case 未填写 `speedRpm`，适配器默认使用 0；若未填写
`absoluteTemperatureAnchor`，会根据 Dirichlet/Robin 边界自动判断。

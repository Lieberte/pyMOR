# 元件级热神经算子数据格式

## 目的

该格式用于把已有求解器的元件热结果转换成跨网格神经算子训练数据。原始 Case 保存物理量，不保存预先拼接或归一化后的神经网络特征，因此后续可以独立调整网络、归一化和采样策略。

一个 Case 表示：

\[
\{\text{元件几何},\text{材料场},\text{热源场},\text{边界场}\}
\rightarrow \{\mathbf{x}_q,T_q\}.
\]

网络训练不要求四面体连接。求解器或前处理程序可以使用单元连接计算节点代表体积、边界面积和法向，但导出的神经算子 Case 只需要点、物理场和测度权重。

## 目录结构

单个 Case：

```text
shaft_case/
├── manifest.json
└── fields.npz
```

数据集：

```text
component_dataset/
├── dataset.json
└── cases/
    ├── shaft_coarse/
    │   ├── manifest.json
    │   └── fields.npz
    ├── shaft_fine/
    └── housing_case/
```

`manifest.json` 保存：

- `formatVersion`
- `caseId`
- `geometryId`
- `componentType`
- 可选工况标量
- 求解器、网格和来源等元数据

`geometryId` 表示同一个物理几何。相同几何的粗网格和细网格应使用同一个 `geometryId`，从而保证划分训练/验证集时不会发生重网格信息泄漏。

## 物理数组

### 体域

- `volumeCoordinatesMeters`：体域采样点物理坐标，单位 m。
- `volumeConductivityWPerMK`：逐点 `3x3` 导热张量。
- `volumeDensityKgPerM3`：密度。
- `volumeSpecificHeatJPerKgK`：比热。
- `volumeHeatSourceWPerM3`：体积热源。
- `volumeWeightsM3`：每个点代表的体积。
- `volumeExtraFeatures`：可选的固定维度附加特征。

### 边界

- `boundaryCoordinatesMeters`：边界点物理坐标。
- `boundaryNormals`：单位外法向。
- `boundaryHtcWPerM2K`：逐点 HTC。
- `boundaryAmbientTemperatureK`：逐点环境温度。
- `boundaryHeatFluxWPerM2`：逐点法向热流密度；正负号沿数据源约定统一。
- `boundaryWeightsM2`：每个边界点代表的面积。
- `boundaryTypeIds`：边界类别编号，例如对流、绝热、接触或给定温度。
- `boundaryExtraFeatures`：可选附加特征。

### 温度标签

- `queryCoordinatesMeters`：温度查询坐标。
- `queryTemperatureK`：求解器温度标签。
- `queryWeightsM3`：误差积分所用的体积权重。

## 自动几何处理

用户不需要手工输入长、宽、高。坐标框架直接由体域点计算：

\[
\mathbf c=\frac{\mathbf{x}_{\max}+\mathbf{x}_{\min}}{2},\qquad
\mathbf s=\frac{\mathbf{x}_{\max}-\mathbf{x}_{\min}}{2}.
\]

网络使用的局部坐标为：

\[
\hat{\mathbf{x}}=\frac{\mathbf{x}-\mathbf c}{\mathbf s}.
\]

点云分布保存具体几何形状；`halfExtentMeters`、总体积和总边界面积作为自动派生的尺度信息进入 Case 级特征。它们不是用户手工定义的边界条件。

## Python 接口

```python
from mor.nn.data import (
    thermalBoundarySamples,
    thermalComponentCase,
    thermalComponentFeatureSchema,
    thermalTemperatureSamples,
    thermalVolumeSamples,
    saveThermalComponentCase,
)

case = thermalComponentCase(
    caseId='shaft_speed_12000',
    geometryId='shaft_geometry_v1',
    componentType='shaft',
    volume=thermalVolumeSamples(
        coordinatesMeters=volume_xyz,
        conductivityWPerMK=conductivity,
        densityKgPerM3=density,
        specificHeatJPerKgK=specific_heat,
        heatSourceWPerM3=heat_source,
        weightsM3=nodal_volume,
    ),
    boundary=thermalBoundarySamples(
        coordinatesMeters=boundary_xyz,
        normals=boundary_normals,
        htcWPerM2K=htc,
        ambientTemperatureK=ambient_temperature,
        heatFluxWPerM2=heat_flux,
        weightsM2=nodal_area,
        boundaryTypeIds=boundary_type,
        patchIds=boundary_patch_id,
    ),
    temperature=thermalTemperatureSamples(
        coordinatesMeters=query_xyz,
        temperatureK=temperature,
        weightsM3=query_volume,
    ),
    operatingFeatures={'speedRpm': 12000.0},
)

saveThermalComponentCase(case, 'dataset/cases/shaft_speed_12000')

operator_case = case.toOperatorCase(
    schema=thermalComponentFeatureSchema(
        operatingFeatureNames=[],
        boundaryTypeCount=4,
    )
)
```

整个数据集保存后，可以直接建立按几何留出的训练数据模块：

```python
from mor.nn.data import thermalOperatorDataModule

data_module = thermalOperatorDataModule.fromComponentDataset(
    'component_dataset',
    validationGeometryIds=['held_out_component_geometry'],
    schema=thermalComponentFeatureSchema(boundaryTypeCount=4),
    batchSize=4,
)
```

同一个 `geometryId` 下的粗网格和细网格会一起进入训练集或验证集。

当热源和 HTC 已知时，`operatingFeatureNames` 可以为空。转速、扭矩等仍可保存在原始 Case 的元数据或工况字段中，但不必进入第一版温度算子的网络输入。

## 真实电机 FEM 转换

`motorFemComponentCase.py` 可以把已经求解的电机 FEM 区域拆成独立部件 Case：

- 每个四面体转换为体域积分点，权重为单元体积；
- 每个表面节点转换为边界点，权重为三角形面积的节点集总；
- 结果文件中的逐节点 `heatFlowRate` 转换成 `heatFluxWPerM2`；
- 当逐节点热流采用对称模型缩放口径时，使用 overview 总换热量校准其整体尺度；
- FEM 节点温度作为查询标签；
- 四面体和三角形连接只在转换阶段使用，不进入神经网络。

生成单部件数据集（输入为调用方准备的求解器数字目录）：

```powershell
python scripts/build_motor_fem_component_dataset.py input/motor_cases temp/experiments/thermal_components --component magnets --speeds 2500 --maximum-volume-samples 1024 --maximum-boundary-samples 2048 --maximum-query-samples 512
```

完整 STA 和多父工况转换见 [motor_sta_dataset.md](motor_sta_dataset.md)，全分辨率视图见 [motor_full_resolution_dataset.md](motor_full_resolution_dataset.md)。训练与评估入口见 [scripts/README.md](../../scripts/README.md)，无量纲契约见 [thermal_dimensionless_v2.md](../thermal/thermal_dimensionless_v2.md)。

结果导出的边界热流适合验证数据格式和给定边界下的场重构。验证未知工况前，需区分求解前已知的 HTC、环境温度、接触热阻，以及求解结果或外部耦合模型提供的接口量。

### 局部密度、积分权重和表面分组

v3 格式明确保存局部强度量及其积分权重：

- 体热源功率由 `volumeHeatSourceWPerM3 * volumeWeightsM3` 求和得到；
- 边界换热量由 `boundaryHeatFluxWPerM2 * boundaryWeightsM2` 求和得到；
- `boundaryPatchIds` 只用于表面分组、界面耦合和逐面守恒检查，不直接作为神经网络输入。

真实 FEM 转换会读取 `.samesh/patches/<parentPatchUuid>/<patchUuid>`，将边界样本按 surface patch 展开。共享棱边上的几何节点可以在不同 patch 中各出现一次，各自携带该 patch 分摊的面积和法向。使用 `--verbose` 可以逐面打印期望换热量、重构换热量、下采样后换热量及误差。

## 当前框架边界

- 当前格式面向稳态标签；瞬态需要增加时间、初始温度和时间相关边界场。
- `boundaryTypeIds` 的编号含义必须由整个数据集统一定义。
- 所有 Case 的附加特征维度必须一致。
- 面积和体积权重需要由已有求解器或可信的网格前处理程序提供。
- 元件单独训练时必须具有完整、物理可解的边界条件。

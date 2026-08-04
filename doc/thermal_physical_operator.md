# 热物理算子主链路

当前实现的目标是把已知电机网格、材料、损耗和 HTC 直接组装为有限元热方程，而不是让网络学习面积、体积或连接关系：

\[
M(T)\dot T + K(T)T = Q.
\]

主链路为：

1. `meshIr` 保存节点、四面体单元、材料区域、命名边界以及原始物理数据。
2. `thermalProblem` 统一长度单位，展开逐单元材料和体积热源。
3. 已知 HTC、环境温度和热流按命名边界逐节点输入，并在三角面上显式积分。
4. 外部对流、热流、温度约束与内部接触界面分别组装。
5. `thermalModel` 完成稳态、单步隐式欧拉或完整多步瞬态求解。

## 基本配置

```python
from mor.nn.data.geometry import meshFileToIr, thermalProblem

mesh = meshFileToIr('motor.inp')
problem = thermalProblem.fromDict(mesh, {
    'lengthUnit': 'mm',
    'materials': {
        'copper': {
            'conductivity': 400.0,
            'density': 8900.0,
            'specificHeat': 385.0,
        },
        'lamination': {
            'conductivity': {
                'temperatures': [300.0, 400.0, 500.0],
                'values': [
                    [[25.0, 0.0, 0.0], [0.0, 25.0, 0.0], [0.0, 0.0, 2.0]],
                    [[23.0, 0.0, 0.0], [0.0, 23.0, 0.0], [0.0, 0.0, 1.9]],
                    [[21.0, 0.0, 0.0], [0.0, 21.0, 0.0], [0.0, 0.0, 1.8]],
                ],
            },
            'density': 7650.0,
            'specificHeat': {
                'temperatures': [300.0, 400.0, 500.0],
                'values': [460.0, 500.0, 540.0],
            },
        },
    },
    'regionMaterials': {
        'WINDING_SOLID': 'copper',
        'STATOR_SOLID': 'lamination',
    },
    'loadCase': {
        'heatSources': {'WINDING_SOLID': 1.2e6},
        'convection': {
            'COOLING_JACKET': {
                'coefficient': {'location': 'node', 'values': [2400.0, 2500.0, 2600.0]},
                'ambientTemperature': {'location': 'node', 'values': [332.0, 333.0, 334.0]},
            },
        },
        'dirichletTemperatures': {'SHAFT_END': 320.0},
        'operatingMetadata': {'speedRpm': 12000, 'torqueNm': 180.0},
    },
    'referenceTemperature': 300.0,
    'nonlinearOptions': {
        'tolerance': 1e-6,
        'residualTolerance': 1e-6,
        'maxIterations': 50,
        'relaxation': 0.8,
    },
    'linearSolverOptions': {
        'method': 'auto',
        'relativeTolerance': 1e-9,
        'preconditioner': 'jacobi',
    },
})

model = problem.buildModel()
steady = model.solveSteady()
times, history = model.solveTransient(steady, timeStep=0.1, stepCount=100)
```

`lengthUnit` 支持 `m/cm/mm`，也可直接给出 `lengthScaleToMeters`。组装始终使用换算到米的网格副本，不会修改原始 `mesh.nodes`。这样面积、体积、导热、热容和边界换热的量纲保持一致。

材料属性保持逐单元或逐材料区域表达。共享节点可能连接不同材料，因此不能把材料物性强制改成逐节点值。温度相关物性使用四面体角点平均温度计算单元物性，并通过 Picard 迭代重新组装；收敛必须同时满足温度相对变化和自由节点方程残差。

## 已知 HTC 映射

如果 HTC 文件使用求解器节点标签，网格需在 `pointData['originalNodeId']` 中保存原始标签：

```python
mapping = problem.mapBoundaryFieldByNodeIds(
    'COOLING_JACKET',
    sourceNodeIds=htc_node_ids,
    sourceValues=htc_values,
)
htc_field = mapping.asBoundaryField()
```

`meshIoToIr(..., originalNodeIds=labels)` 可以显式写入这些标签。由于部分格式经过 `meshio` 后已经丢失原始标签，代码不会把内部数组下标伪装成原始节点号。

如果 HTC 文件提供空间坐标，可以使用 nearest 或 IDW：

```python
mapping = problem.mapBoundaryFieldByCoordinates(
    'COOLING_JACKET',
    sourceCoordinates=xyz,
    sourceValues=htc,
    sourceLengthUnit='mm',
    method='nearest',
    maxDistance=2e-3,
)
print(mapping.diagnostics())
htc_field = mapping.asBoundaryField()
```

`maxDistance` 的单位固定为米。诊断包含覆盖率、最大/平均距离和未覆盖的全局节点下标。默认要求 100% 覆盖，未覆盖节点不会被静默填零或外推。

边界场支持：

- `uniform`：整个命名边界一个值。
- `face`：每个三角面一个值。
- `node`：按照该边界 `boundaryNodeIndices` 的升序逐节点给值。
- `globalNode`：按照全局节点顺序给值。

逐节点 HTC、环境温度和热流使用三角形线性形函数一致积分。面积由网格显式计算，不依赖相关性学习。

## 接触界面

内部部件接触与外部对流分开配置：

```python
'contacts': {
    'winding_to_stator': {
        'primaryBoundary': 'WINDING_CONTACT',
        'secondaryBoundary': 'STATOR_CONTACT',
        'coefficient': 1500.0,
        'mapping': 'matching',
        'maxDistance': 1e-5,
    },
}
```

接触项按主侧三角面面积形成对称耦合块：

\[
\begin{bmatrix}H&-H\\-H&H\end{bmatrix},
\]

因此两侧交换热量严格成对出现，常温场不会产生虚假热源。`matching` 要求节点数相同且一一配对；`nearest` 支持带距离容差的最近节点配对，并报告覆盖和距离。共享全局节点的材料界面已经表示理想接触，不应再次添加接触热阻。内部接触边界不能同时配置成外部对流或外部热流边界。

当前最近节点方案是可诊断的工程近似，不是通用非共形面到面 mortar。节点数差异很大、滑移接口或强非共形网格应在后续实现 mortar/投影积分后再使用。

## 求解与诊断

线性求解支持 `direct/cg/gmres/auto`，CG/GMRES 可使用 Jacobi 预条件，并在 `model.lastLinearSolveInfo` 中记录方法、迭代次数、残差和是否回退到直接法。`auto` 对较小系统使用直接法，对达到阈值的大型 SciPy 稀疏系统使用 CG。

非线性求解诊断保存在 `model.lastNonlinearInfo`。稳态残差为 `K(T)T-Q`，瞬态残差为 `M(T)(T[n+1]-T[n])/dt+K(T)T[n+1]-Q`；Dirichlet 节点不参与自由度残差判据。

## 当前边界

- 体单元支持线性四面体；`tetra10` 当前仅取四个角点，尚未使用二次形函数。
- 已知 HTC 会进入物理组装，但当前主链路不训练 HTC、材料算子或几何算子。
- 转速、扭矩等只保存在 `operatingMetadata`。它们需要先通过损耗模型或 HTC 模型转成热源/边界场，才会改变 `K/M/Q`。
- 可微分边界组装和可微分线性求解留在后续子分支实现。

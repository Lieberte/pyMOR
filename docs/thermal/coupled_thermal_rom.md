# 部件降阶与整机热耦合

`thermalCouplingReductor` 对每个线性 `thermalModel` 分别投影，再通过 `thermalCouplingInterface` 组装 `coupledThermalModel`。可用于齿轮、轴承、箱体、电机部件以及带热容的油液/空气热网络节点，不依赖电机命名或训练分类模型。

## 方程与输入

每个部件提供 `M/K/Q`、可选热源输入算子 `B`、固定温度节点，以及温度快照或已有基底。原方程为 `M dT/dt + K T = Q + B u(t)`。已知外部对流可预先加入 `K/Q`；变化的环境温度可以通过 `B u(t)` 表示。当前 `M/K`、接触热导和 HTC 在一次预测期间保持不变。

两侧接口在共同积分点上插值温度，由 `primaryTrace/secondaryTrace` 给出。每行插值权重非负且和为一；`conductance` 为 `h × area`，单位 W/K。主侧向次侧的热流为：

```text
q = conductance × (primaryTrace × T_primary − secondaryTrace × T_secondary)
```

将 `+trace_primary.T × q` 和 `−trace_secondary.T × q` 加入两侧导热项。接口矩阵对称、半正定，常温场不产生接口热流；两侧交换功率抵消。正热流表示主侧向次侧流出。

部件温度使用 `T = offset + V a`，采用 Galerkin 投影 `V.T M V / V.T K V`，保留自由节点常温模式。固定温度节点的基底行置零，通过 offset 精确约束。接口也投影到部件基底，在线求解不组装全阶整机矩阵。

## 最小示例

两个带热容的热网络节点：固体向油液换热，油液向 300 K 的环境散热，固体热源在 1 秒后增加。

```python
import numpy as np
from mor.assembly import thermalCouplingInterface
from mor.models import coupledThermalModel, thermalModel
from mor.operators import matrixOperator, thermalOperatorSet
from mor.reductors import thermalCouplingReductor

def makeModel(capacity, sink, ambient):
    return thermalModel(thermalOperatorSet(
        matrixOperator([[sink]], 'scipy'),
        matrixOperator([[capacity]], 'scipy'),
        matrixOperator([sink * ambient], 'scipy'),
        np.empty(0),
    ), referenceTemperature=300.)

reductor = thermalCouplingReductor()
solid = reductor.reduceComponent(
    'gear', makeModel(2., 0., 300.), basis=[[1.]],
    inputOperator=matrixOperator([[1.]], 'scipy'),
)
oil = reductor.reduceComponent('oil', makeModel(3., 1., 300.), basis=[[1.]])
contact = thermalCouplingInterface.fromNodePairs(
    'gear_oil', 'gear', 'oil', 1, 1, [0], [0], coefficient=200., areas=.02,
)
rom = reductor.assemble([solid, oil], [contact])
result = rom.solveTransient(
    {'gear': 300., 'oil': 300.}, times=np.arange(101) * .1,
    inputs=lambda time: {'gear': [10. if time < 1. else 20.]},
)
gearTemperature = result.temperatures('gear')
oilTemperature = result.temperatures('oil')
interfacePower = result.interfacePowers()['gear_oil']
balanceError = result.diagnostics['residualW']
```

温度数组为 `[time, fullNode]`。结果默认保存降阶坐标，`temperatures(name)` 按需重构部件全场；大型数据可逐段调用 `component.reconstruct(coordinates)`，避免同时恢复全部部件。

## FEM 部件与基底

### 从原始 STA 恢复物理方程

`mor.assembly.loadThermalArchive(staPath, componentFolders)` 将原始模型与完整网格导出目录连接起来。`componentFolders` 为部件名称到导出目录的映射，必须覆盖全部活动 FEM 部件。

目前支持直接 FEM schema 的常数各向同性材料、体热源、固定均匀齿面热负荷、用户指定接触系数、常数 HTC、静态空间 HTC、集总热容、常数热阻、固定环境温度和绝热边界。其他活动物理项显式报错。物性和载荷从原始控制数据读取，导出网格必须与原始节点和单元一致；求解后的温度与边界热流不参与组装。

```python
from mor.assembly import loadThermalArchive
from mor.models import coupledThermalModel

physical = loadThermalArchive(sourceSta, componentFolders)
rom = physical.reduce(offlineBases, rank=8)  # 每个固体和热容节点均需提供基底
result = rom.solveTransient(physical.initialTemperatures, times)
rom.save('gearbox_rom.npz')
restored = coupledThermalModel.load('gearbox_rom.npz')
```

接触采用原始搜索半径内的三角面最近点和重心插值，双侧半权重积分形成对称守恒交换。未覆盖面积记录于 `physical.audit['contacts']`，不会扩大半径连接。热网络通过 FEM 面的面积平均温度连接，均匀分配面功率；质量为零的网络面不引入额外热容。面平均环境散热采用低秩算子，避免生成稠密全阶矩阵。该离散近似需要与原求解器分别验证。

当 FEM-network 接口使用 `mode='fem'` 时，油侧对流系数由 `htc3D` 驱动。组装器读取 `controlData.h5/<tableUuid>/mappingData/<timeSlice>` 中每个三角面的输入 HTC，以三点面高斯积分形成一致边界项，并向油液节点施加等量反向热流。该场覆盖对流组件上的占位常数，包含占位常数为零的支路。当前要求单个静态时刻和集总流体端点；缺少映射或出现动态场时显式报错。`useSpatialConvection=False` 仅用于复现旧版漏项的消融诊断。

真实数据验证入口是 `scripts/generate_real_thermal_report.py --dataset <component-dataset.json> --whole-dataset <whole-assembly-dataset.json>`。训练工况生成离线 POD，留出工况只用于事后对照。报告分别记录基底投影误差、自主预测对原 STA 的差异、同一组装方程的一步全阶参考误差，以及能量平衡残差。一步全阶验证不能代表全时段降阶精度。

已有四面体组装或几何热问题生成的线性 `thermalModel` 可直接输入。快照必须在同一网格、同一节点顺序下组织为 `[node, snapshot]` 的绝对温度。

调用形式为 `reductor.reduceComponent('housing', componentModel, snapshots=snapshots, rank=12, inputOperator=heatSourceOperator)`。rank 包含常温模式，实际秩可能因快照相关而更低。

也可传入 `basis=podResult` 或 `[node, mode]` 基底。POD 中心化均值自动作为 offset；普通基底默认使用 `referenceTemperature`，可显式设置 `offset`。基底自动补充常温模式并正交化；显式 rank 限制最终维数。

快照决定温度空间，输入算子决定新热源工况如何进入方程；接口热流由两侧温度在线计算。部件的 `K/Q` 应包含内部导热和外部边界，不应保留准备替换为耦合接口的结果热流或重复接触项。转速、扭矩需要经损耗或换热模型转换，不能直接当作热功率。

## 接口映射

- `fromNodePairs(...)`：提供两侧对应节点和积分面积。
- `fromNearestNodes(...)`：按空间位置最近节点配对，以 `maxDistance` 拒绝远距离连接。面积属于主侧积分点；允许多个主侧节点映射同一次侧节点。
- 直接构造接口：提供两侧在共同积分点上的稀疏插值矩阵，支持调用方生成的非共形插值或面形函数，两侧节点数可以不同。

最近节点是工程近似，当前不自动生成高精度面到面 mortar、不推断面积、不处理滑动啮合的时变接触。每个物理接触只添加一次；共享全局节点或已经组装过的接触项不能重复添加。

油液节点可具有热容、散热项和热源，通过固体—油液热导与多个部件连接，预测油温反馈。这是热网络模型，不是完整流体计算。

## 求解、守恒与保存

`solveSteady(inputs)` 返回整机降阶坐标，用 `reconstruct` 恢复全场。封闭绝热系统通常没有唯一稳态解，需要散热边界。瞬态采用隐式 Euler 联立求解，固定步长复用 Cholesky 因子。允许非均匀递增时间点，输入回调在每步末端求值。

`energyBalance` 和结果 diagnostics 记录热能、储能功率、外部净功率、固定温度边界补偿功率及守恒残差。常温模式使该全局守恒诊断对降阶模型成立；守恒不等于温度场准确，仍需独立验证预测误差。

```python
rom.save('coupled_thermal_rom.npz')
restored = coupledThermalModel.load('coupled_thermal_rom.npz')
```

保存部件基底、温度偏移、投影、接口迹、降阶算子及诊断权重。NPZ 不使用 pickle，重载不依赖原始快照或全阶模型。

全阶部件可来自 SciPy 稀疏/组合算子或 Torch 算子；投影通过公开 apply 接口避免把全阶稀疏 `M/K` 稠密化。基底和降阶整机求解使用 NumPy/SciPy CPU。温度相关材料、时变 `M/K/HTC` 和自动 STA 整机接触恢复尚未接入此方法，调用时拒绝非线性模型。

## 验证

```powershell
.\.venv\Scripts\python.exe -m pytest tests/models/test_coupled_thermal.py tests/benchmarks/test_coupled_thermal_smoke.py
.\.venv\Scripts\python.exe scripts/generate_coupled_thermal_report.py --output-directory reports/coupled_thermal
```

报告用合成的齿轮—轴承—箱体—油液网络，三个热源脉冲建立基底，用另一组热源组合验证温度误差、接口功率和能量残差。它验证实现，不能替代真实齿轮箱的独立工况验证。

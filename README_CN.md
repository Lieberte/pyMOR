# mor 代码说明与交接指南

本项目提供经典模型降阶（MOR）、热有限元组装与热模型、神经网络降阶、几何识别，以及面向 shonTA 的 Python 任务服务。可以从快照或物理系统构建降阶模型，再进行预测、重构、保存和重载。

分发名和导入名均为 `mor`，当前版本 `0.1.0`，要求 Python **3.10 及以上**。本仓库接口与官方 pyMOR 不保证兼容。许可证元数据为 Proprietary。

## 1. 代码结构

| 路径 | 内容与入口 |
| --- | --- |
| `mor/backends/` | SciPy/Torch 后端、数组与稀疏格式、设备和数值库操作 |
| `mor/operators/` | `matrixOperator`、低秩、缩放、求和与组合算子 |
| `mor/algorithm/` | SVD、DMD、线性求解、连续/离散 Lyapunov 和 auto 选择 |
| `mor/solvers/`、`mor/reductors/` | POD、平衡截断、频响、DMD、POD-DMD 和热投影 |
| `mor/assembly/` | 热 FEM 单元、全局稀疏组装、接触和热网络接口 |
| `mor/models/` | LTI、热物理、耦合热 ROM、仿射热 ROM |
| `mor/applications/motor_thermal/` | 电机线性热问题、描述系统装配、POD 和误差报告 |
| `mor/nn/` | AE/VAE/卷积 AE、动力学网络、热算子、训练、数据和几何识别 |
| `mor/workflow/`、`mor/shonta.py` | JSON 任务、事件、校验、取消、保存、重载及 ONNX |
| `tests/` | 单元、数值、集成、算子契约与轻量 benchmark 回归 |
| `scripts/` | 数据构建、训练、推理、接口检查与报告生成 |
| `docs/` | 接口、数据格式和运行说明 |

顶层 `import mor` 不初始化 Torch。两个后端均注册，实际请求 Torch 后端或 NN 功能时才加载。依赖由安装时选择的 extras 提供，缺库时执行会直接报错。

## 2. 安装

以下命令在源码根目录运行，始终使用同一解释器。Windows 示例：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Linux/macOS 使用 `python3 -m venv .venv` 和 `source .venv/bin/activate`，后续安装命令相同。

| 用途 | 安装命令 |
| --- | --- |
| 核心 MOR、SciPy 热 ROM | `python -m pip install -e .` |
| 测试、Torch、网格、电机数据、分类器 | `python -m pip install -e ".[dev]"` |
| 全量测试、任务服务与 ONNX | `python -m pip install -e ".[dev,deployment]"` |
| 数据、训练、部署、性能报告 | `python -m pip install -e ".[all,deployment,reports]"` |
| Mamba 专项 | `python -m pip install -e ".[mamba]"`，并阅读专项文档 |

还可单独选 `torch`、`mesh`、`motor`、`sklearn`。`all` 不包含 `deployment`、`reports`、`mamba`；完整依赖见 [pyproject.toml](pyproject.toml)。官方 Mamba 编译内核另需匹配的 Linux/CUDA 环境。

使用交接包 wheel 时，在其所在目录执行：

```powershell
python -m pip install "mor-0.1.0-py3-none-any.whl[dev,deployment,reports]"
```

wheel 不内置第三方依赖，安装需要联网或本地依赖源；开发和复现应使用交接包 `source/pyMOR/` 中的测试、脚本与文档。

## 3. POD 与 DMD 最小示例

快照排列为 `(状态数, 快照数)`。以下例子只需要核心依赖：

```python
import numpy as np
import mor
from mor.reductors import podReductor

snapshots = mor.matrixOperator(
    np.array([[1.0, 0.9, 0.81, 0.729],
              [1.0, 0.8, 0.64, 0.512]]),
    backendName="scipy",
)
pod = podReductor("scipy")
reduced = pod.reduce(snapshots, rank=2)
basis = reduced.basis.toNumpy()
coordinates = basis.T @ snapshots.toNumpy()
reconstruction = pod.reconstruct(reduced, coordinates).toNumpy()
np.testing.assert_allclose(reconstruction, snapshots.toNumpy(), atol=1e-12)

dynamics = mor.dmdReductor("scipy").reduce(snapshots, rank=2)
prediction = dynamics.predict(steps=5)
assert prediction.shape == (2, 5)
print("POD order:", reduced.order, "DMD shape:", prediction.shape)
```

`rank` 指定保留阶数；`tol` 通常表示丢弃的奇异值平方能量比例，具体范围和组合规则以对应接口为准。加权 POD、中心化、受控 DMD、Hankel DMD 和各后端支持范围有区别。

## 4. 仿射热 ROM 保存与重载

固定容量、导热及固定空间 Robin 边界模式离线投影一次，在线更新小矩阵系数与输入历史。两节点示例：

```python
from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np
from mor.models import affineThermalModel
from mor.reductors import affineThermalReductor

mass = np.diag([2.0, 3.0])
bulk = np.array([[4.0, -4.0], [-4.0, 4.0]])
boundary = [np.diag([1.0, 0.0])]
loads = np.array([[1.0], [0.0]])
source = np.array([0.0, 2.0])
basis = np.eye(2)

model = affineThermalReductor().reduce(
    mass, bulk, boundary, source, loads, basis,
    referenceTemperature=300.0, interfaceNames=["coolant"],
)
with TemporaryDirectory() as directory:
    path = Path(directory) / "solid_rom.npz"
    model.save(path)
    loaded = affineThermalModel.load(path)
    result = loaded.solveTransient(
        dt=0.1, sourceScales=np.ones(20),
        boundaryScales=np.ones((20, 1)),
        referenceTemperatures=np.full((20, 1), 300.0), initialRise=0.0,
    )
    assert result.coordinates.shape == (21, 2)
    assert np.max(np.abs(result.energyResidualW)) < 1e-9
    print("Final mean temperature / K:", result.meanTemperature[-1])
```

温度使用 K，热功率使用 W，其余矩阵按一致物理单位输入。基底必须正交并包含常温模态；每个边界满足 `Kboundary[j] @ ones == Bboundary[:, j]`。

在线 NPZ 保存降阶矩阵和接口输出映射，不保存网格或全场基底。`meanTemperature` 为容量加权均温，`interfaceTemperatures` 为 HTC 加权界面温度，接口功率取正表示固体流出。热点和全场输出需要另存基底并重构。

不支持任意变化的空间 HTC、温变材料、固定 Dirichlet 或非对称输运。时间离散对当前 `K_n` 使用 `M/dt ± K_n/2`，不同于使用新旧两个 K 的梯形格式。详见 [仿射热 ROM 契约](docs/thermal/affine_thermal_rom.md)。

## 5. 接入 shonTA

`mor.shonta` 提供 `getCapabilities`、`validateWorkflow`、`executeWorkflow` 及加载器。输入为 JSON 和 NPY 路径，输出为状态、事件、文件引用和模型清单。

当前方法包含 POD、AE、DMD、POD-DMD、BT 和 mock。同进程一次一个任务，每次执行一个方法节点；图快照用于审计，不执行完整 DAG。任务服务 CPU 设置与通用 NN 流水线的 CUDA 能力分别定义。

调用方先读取能力和参数默认值，再验证请求。重载须同时核对成功终态及最后提交的 `manifest.json`。详见 [任务服务契约](docs/integration/TASK_SERVICE.md)。

## 6. 验证

```powershell
python -m pip install -e ".[dev,deployment]"
python -m pytest -ra -p no:cacheprovider
python -m pytest tests/applications/motor_thermal tests/models/test_affine_thermal.py
python scripts/check_reduction_methods.py --rounds 1
```

接口检查默认用合成快照验证 POD/AE 的执行、保存和重载，`--motor-data` 另需本地电机输入。详细流程见 [验证与复现](docs/VALIDATION.md)。

2026-10-27 的验证环境为 Windows、Python 3.11.9、NumPy 2.4.1、SciPy 1.17.0、Torch 2.10.0+cu126；全量结果 **1075 passed，2 warnings**。这是环境相关记录，不表示所有平台或真实工况均已验收。交接包附原始日志和环境清单。

## 7. 已知问题

截至 2026-10-27，下列审查问题尚未修复：

| 问题 | 当前使用建议 |
| --- | --- |
| 稀疏 POD 中心化绕过 64 MiB 显式密集化检查 | 大型稀疏输入暂不设 `centerSnapshots=True`；小 rank 也需核对 auto 实际分解和预算 |
| 电机 POD 中心化后丢失均值和仿射项 | 此封装暂使用未中心化输入；温升/偏置须建立明确的物理模型契约 |
| 整数温度误差计算可能回绕 | 评价前将两份数据转为 `float64` |
| 热算子训练器按批次数平均损失，最后小批次权重过大 | 独立按明确的样本/query 权重计算全局验收指标 |
| `pdeResidualLoss` 是普通 MSE 占位 | 不把该名称当作物理约束；专用 `thermalSteadyPinnLoss` 为另一接口 |

一般稀疏分解的显式密集化检查为 64 MiB，中心化路径是已知例外；输入预算也不包含全部工作区或进程内存。

**整机电机 ROM 尚未通过启动阶段验收**：已有首轮外迭代的转轴最大节点误差约 44.98 K、转子铁芯约 27.23 K，缺少完整可比较的全阶/降阶收敛轨迹及有效整机加速比。部件级同网格验证、能量守恒和投影一致不能替代整机验收。

通用 PINN 部分目录仍为空占位，适配器协议不等于已实现所有文件读取。官方 Mamba CUDA 内核须按 Linux/GPU 流程验证，小规模测试不代表跨工况或跨几何泛化。

## 8. 数据与交接包范围

Git 包含源码、正式测试、文档与维护脚本。原始 STA、完整网格/时间序列、训练权重、报告产物以及 `applicationTests/` 本机样例默认不纳入 Git。文档中的示例路径不表示已附数据。

交接 ZIP 外层 `README.md` 说明实际文件，`MANIFEST.json` 和 `SHA256SUMS.txt` 记录来源与摘要。`source/pyMOR/` 为可安装源码快照，额外复制的必要复现脚本、小型样例和审查资料会标记为补充资产；wheel/sdist 为标准安装产物。

大型 STA、数据集、训练 checkpoint、GUI/solver 完整仓库与第三方依赖不在本次包内。附带的 C++ 补丁仅是实验材料，不表示已并入 solver 或通过整机验收。

运行结果建议放 `reports/<topic>/<run>/`，临时数据放 `temp/experiments/<run>/`。部分脚本有本机默认路径，接手人需检查输入参数；部分旧报告脚本不解析参数，传 `--help` 也会启动生成，参见 [脚本导航](scripts/README.md)。

## 9. 阅读顺序

1. 本 README 与 [全部文档导航](docs/README.md)。
2. [后端/组装](docs/BACKEND_ASSEMBLY.md)、[线性求解](docs/LINEAR_SOLVERS.md)、[DMD](docs/contracts/DMD_CONTRACT.md)。
3. [电机应用层](mor/applications/motor_thermal/README.md)、[热物理](docs/thermal/thermal_physical_operator.md)、[耦合 ROM](docs/thermal/coupled_thermal_rom.md)。
4. [NN 运行时](docs/NN_RUNTIME.md)、[Mamba](docs/MAMBA_DYNAMICS.md)、[STA 数据](docs/data/motor_sta_dataset.md)、[几何识别](docs/geometry/COMPONENT_RECOGNITION.md)。
5. [目录和资产](docs/PROJECT_LAYOUT.md)、[性能测量](docs/PERFORMANCE.md)、[经典 MOR 优化流程](OPTIMIZATION_PLAN.md)。

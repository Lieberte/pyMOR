# 后端、组装与算子分层

后端封装 NumPy/SciPy/Torch 的数组格式、数值库操作和可复用分解；组装层计算单元矩阵并累加全局矩阵；算子层提供表示、结构信息和 apply；算法层处理求解策略、移位、收敛与缓存预算。

| 位置 | 职责 |
| --- | --- |
| `mor/backends/backendsBase.py` | 数组、分解、dtype、稀疏构造、Dirichlet 消元的共同接口 |
| `mor/backends/scipyBackend.py` | SciPy CSR/COO 转换、直接与可复用 LU、受限密集化 QR |
| `mor/backends/torchBackend.py` | 张量设备、稀疏格式、原生 COO 构造、稠密 LU 和稀疏消元 |
| `mor/assembly/tetra.py` | NumPy 线性四面体单元核，不分配全局矩阵 |
| `mor/assembly/sparse.py` | 按后端执行成对稀疏累加，保持原有批次策略 |
| `mor/assembly/thermal.py` | 材料/边界输入校验、单元调用、全局热算子组装与约束入口 |
| `mor/operators/thermalSet.py` | 热算子集合、K/M/Q 与残差接口 |
| `mor/operators/thermal.py` | 原公开导入路径的兼容导出 |
| `mor/algorithm/linear/shifted.py` | 移位/普通转置规则，委托后端 factorize/solve |

`mor.operators` 原公开入口保持可用，也可以从 `mor.assembly` 导入组装函数。几何数据适配器仍负责网格、材料和命名边界映射，它继续调用上述公共组装入口。

## 格式与设备契约

`array.resultType/astype/toArrayLike/isDiagonal/diagonal/sparseFromTriplets/eyeLike` 统一 dtype、结构检测、稀疏构造和匹配单位矩阵。matrixOperator 和 lowRankOperator 使用这些接口，无需自己导入 Torch/SciPy 并分支。第三方后端使用这些算子时应实现相应接口；缺少能力明确抛 NotImplementedError。

Torch 的 COO、CSR、CSC、BSR、BSC 均识别为稀疏；对角检测/提取先规范化坐标和重复项，忽略数值为零的非对角项，不调用 to_dense。apply 和普通转置支持这些格式，压缩格式通过稀疏 COO 运算；输出矩阵的具体稀疏格式不保证与输入相同。转 NumPy 是显式稠密化接口，并处理 conjugate/negative view。

SciPy 稀疏输入转 Torch 会构造稀疏 COO 并保留 dtype，非 Tensor 输入放到所选设备；已有 Tensor 保留原 device。算子 apply 的 NumPy 输入通过 toArrayLike 放到算子实际 device，避免与注册器默认设备不一致。Torch 转 SciPy 的稀疏数据通过坐标和值生成 CSR，设备迁移、detach 属于显式转换，不保持 autograd。稀疏单位矩阵只构造 O(n) 的索引和值。结构标志沿用算子构造/首次检测的缓存规则，直接修改底层矩阵后应重建算子。

## 热组装

```python
from mor.assembly import assembleLinearTetraThermalOperators, applyDirichletConditions

operators = assembleLinearTetraThermalOperators(
    nodes, elements, conductivity=2., density=3., specificHeat=4.,
    backendName='torch', deviceName='cuda', assemblyBatchSize=256,
)
constrained, load = applyDirichletConditions(operators.K, operators.Q, fixedNodes, 300.)
```

单元几何/材料核目前使用 NumPy CPU；Tensor 输入显式转主机，局部矩阵随后通过后端构造稀疏批次，在选择的 CPU/CUDA 设备执行全局累加。元数据 `elementKernelBackend='numpy'`、`assemblyBackend` 说明实际路径。Torch 返回原生稀疏 COO 的 K/M 与同设备稠密 Q；这是后端可用性补全，不宣称全部单元计算已在 GPU 或支持可微 FEM。

Dirichlet 入口从算子推断后端，也接受显式 backendName/deviceName。校验尺寸、整数节点、有限边界值、重复节点冲突；保留复数 dtype，生成副本，不修改原矩阵/载荷。SciPy 使用稀疏行列消元；Torch 使用固定节点向量修正 RHS、过滤稀疏坐标并补单位对角，不先稠密化。空节点集合保持方程数值不变。

## 分解与求解能力

`backend.linalg.factorize(A)` 返回可复用 solve callable，支持向量/多 RHS 和实分解的复 RHS，`factorBytes` 为缓存预算使用的估计值。SciPy 支持稠密 LU 和稀疏 splu；Torch 支持稠密 CPU/CUDA LU。旧 `factorizedSolve` 是兼容委托，具体 LU 实现只放在后端，LRADI 缓存生命周期仍由算法控制。

SciPy 没有 `scipy.sparse.linalg.qr`。稀疏 QR/正交化采用与分解一致的 **64 MiB 输入密集化上限**，超过预算明确拒绝；预算不覆盖 QR 工作区。单列非零输入保留原归一化语义，零列走 QR，避免除零。

Torch 没有原生 Schur；接口明确抛 NotImplementedError 并提示 SciPy。Torch 稀疏直接求解、稀疏 LU、稀疏 QR、截断稀疏 SVD 同样明确拒绝，不隐式迁移 CPU 或分配大稠密矩阵。因此 Torch 组装/稀疏 apply 可用，并不代表其所有稀疏直接求解器均可用。

## 验证

`tests/backends/test_format_contract.py` 保护格式、转换、对角、O(n) 单位矩阵、设备、LU 和能力拒绝；`tests/operators/test_backend_thermal_assembly.py` 对比 SciPy/Torch CPU/CUDA 的体积、边界、接触、Dirichlet 和残差，并在关键路径禁止调用 to_dense。原有独立单元积分及因子复用回归继续执行。

`scripts/generate_validation_report.py` 加入 assembly_scipy/assembly_torch_cpu/assembly_torch_cuda 独立进程场景，使用 729 节点、3072 单元、256 单元批次，校验总容量、总热源、常数场零刚度和稀疏约束。报告分别给出同步后的 assemblyMs/dirichletMs、进程 RSS 和 CUDA allocated/reserved。整体场景耗时/内存还包含解析校验中的显式稠密参考，不等于纯组装峰值；存储 nnz 包含显式零，不作为不同后端数值结果是否一致的判据。

运行命令、依赖与内存指标范围见 [VALIDATION.md](VALIDATION.md)。测试和报告以实际运行环境为准。

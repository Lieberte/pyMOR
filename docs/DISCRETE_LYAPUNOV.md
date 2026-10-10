# 离散 Lyapunov 求解

SciPy 稠密 auto 现在使用 `discreteBartelsStewart`，Torch 稠密和 SciPy 稀疏 auto 使用 `lrsmith`。连续路径规则保持原有行为。高级算子可通过公开 toBackendData 接口调用；新增直接法只支持实数 SciPy 算子。

求解方程为 `A X Aᵀ − E X Eᵀ + B Bᵀ = 0`；E 缺失时为单位矩阵。trans=True 使用 Aᵀ、Eᵀ 与 Bᵀ 的统一契约，此时传入 B 的形状为 `(inputs, states)`。这是普通转置，新增直接法明确拒绝复数输入。

直接法在本次求解内分解 E，将问题转换为标准离散方程，调用 SciPy 的 solve_discrete_lyapunov；检查原坐标相对残差，最多两次修正。acceptanceResidual 默认 1e-8，超限明确失败；谱半径必须小于 1，奇异 E 明确拒绝。显式稀疏输入转为稠密时遵守已有 64 MiB 输入转换限制，auto 对稀疏仍选择 LRSmith。

```python
import numpy as np
from mor.operators import matrixOperator
from mor.solvers.lyapunov import lyapunovSolver

A = matrixOperator(np.diag([0.2, 0.6]), 'scipy')
B = matrixOperator(np.ones((2, 1)), 'scipy')
solver = lyapunovSolver('scipy', isContinuous=False)
X = solver.solve(A, None, B)
print(solver.lastInfo)
```

新增回归验证普通/描述、转置、近单位圆、稳定性拒绝、奇异 E、sum/scaled 算子、Torch auto 和离散 balanced truncation 集成。近单位圆会放大解或增加迭代成本；极端描述条件数和真实超大模型仍须独立残差/内存检查。

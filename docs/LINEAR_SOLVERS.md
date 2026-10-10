# 线性求解入口

`shiftedLinear` 求解 `(A + shift E) x = rhs`；E 缺省为单位矩阵。直接 solve 每次求解，显式 `factorize` 可在调用方复用 LU，LRADI 只在单次 solve 内按预算缓存。无需调用方复制分解逻辑。

`gmresLinear` 提供 SciPy 无矩阵显式组装的迭代路径，通过算子公开 `apply` 调用 A/E，支持稠密、稀疏、lowRank、sum、scaled、描述、普通转置、复数和多 RHS。输出保持一维向量或二维 RHS 的输入形式，逐列求解。当前不支持 Torch。

```python
from mor.algorithm.linear.gmres import gmresLinear

solver = gmresLinear('scipy', tol=1e-10, atol=0., restart=30, maxIter=100)
x = solver.solve(A, rhs, E=E, shift=-1., trans=False)
```

tol/atol 为有限非负残差阈值，restart 是 Krylov 空间大小，maxIter 是 SciPy GMRES 的重启周期预算。成功返回前检查实际绝对残差；未收敛抛 ValueError。`lastIterations` 给出各 RHS 的内迭代次数，`lastResiduals` 为实际绝对残差，`lastConverged` 在全部 RHS 成功后为 True。接口采用普通转置，复数时不把它自动解释为共轭转置。

linear auto 默认仍选直接 shifted；显式 `matrixFree=True` 时选 GMRES，Torch 请求该提示会明确拒绝。当前没有依据矩阵阶数自动切换的经验阈值，也不宣称所有大矩阵 GMRES 更快；缺少良好谱结构或预条件时应评估收敛预算。组合算子和失败契约见 `tests/algorithm/test_gmres.py`。

具体 LU 实现统一位于后端的 `linalg.factorize`；移位 solve/factorize 共用 `eyeLike`，匹配输入 device、dtype 和稀疏性。Torch 支持稠密直接求解和 LU，稀疏直接求解明确拒绝，完整分层见 [BACKEND_ASSEMBLY.md](BACKEND_ASSEMBLY.md)。

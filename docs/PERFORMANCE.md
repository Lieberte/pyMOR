# 性能策略与测量

性能改动同时保护数值结果、算子契约和边界行为。优化策略按输入结构选择；微基准不设固定加速倍数门槛，运行结果与设备信息保存到报告目录。

## 当前策略

| 节点 | 实现与适用条件 | 验证与取舍 |
| --- | --- | --- |
| 节点体积、边界面积与法向 | 使用 `bincount` 聚合节点贡献 | 保持积分权重与边界节点顺序 |
| 法向抵消后的回退 | 一次计算各节点相邻面的最大面积及首个面编号 | 保持最大面、同面积首个面的原规则，消除逐节点扫描全部面的开销 |
| 分组守恒采样 | 稳定排序一次，再按连续区间取候选点 | 保持随机种子和候选顺序；更多临时索引换取多分组场景的速度 |
| 无权重宽矩阵 dual SVD | 使用行 Gram 矩阵，避免构造更大的列 Gram 矩阵 | SciPy/Torch、实数/复数回归；无权重路径增加缩放和精度阈值，保留的小谱接近 Gram 精度时回退直接 SVD；加权路径通过正对角平方根或 Cholesky 白化后调用同一路径，再反变换左基，避免直接形成加权 Gram。要求 Hermitian 正定积；稀疏快照及一般稀疏积的密集化受 64 MiB 输入预算限制 |
| 能量截断 | 按最大奇异值归一化后平方 | 修复极大/极小尺度的选阶；SciPy/Torch 和 POD 集成验证 |
| 增量 SVD | 对新批次做投影和残差再正交化，只分解更新核心，直接更新右因子 | 保持“截断后的历史 + 新批次”语义；Fast/Brand/Standard 是兼容名称，共用同一实现。保留右因子仍随历史列数增长，不宣称固定内存流式处理 |
| SciPy POD 重构诊断 | 大于 4 MiB 的稠密快照及稀疏快照按块计算显式残差；小系数矩阵采用连续行块 | 主要降低临时分配，保护加权、中心化、复数及极端尺度；不使用相减能量公式估计残差 |
| SciPy 默认稠密范数 | Frobenius 范数通过展平后的稳定向量范数计算 | 保持显式 `ord` 及稀疏路径；防止有限的极大/极小值产生错误的无穷大或零 |
| LRADI 移位更新 | 只拼接需要的末尾因子列；收敛或达到迭代上限后不再更新移位 | 稀疏算子、广义系统、转置、SciPy/Torch 投影及实际方程残差回归；不改变移位选择公式 |
| SciPy 描述系统 LRSmith | 每次 solve 内仅分解 E 一次，重复 RHS 使用 LU / sparse LU | 只在本次调用内复用；转置、稠密/稀疏及同实例换 E 回归通过。缓存分解增加临时内存，不能据小场景调整 auto |
| NN 张量传输与损失累计 | 合并设备和 dtype 转换；CUDA 在设备累计，只在 epoch 末读取标量；CPU 保留标量累计 | CUDA 和 CPU 不等长批次回归；`zero_grad(set_to_none=True)` 减少梯度清零工作 |
| 重构指标 | CUDA 批次指标在设备累计；验证损失按样本数加权；平方及评估损失在 float64 计算 | 修复最后小批次被过度计权及低精度平方溢出；空验证数据明确报错；模型仍使用配置 dtype，诊断提升精度会增加运算成本 |
| 数组落盘 | 写入时累计 SHA256；常规 dtype 按约 1 MiB 缓冲写入 NPY | 字节与 `np.save` 一致，保留 fsync、原子替换及失败清理；结构化 dtype 使用 NumPy 原生保存 |
| 工作流训练历史 | 每 epoch 保留指标事件；完整历史快照按 `ceil(epochs/100)` 间隔落盘 | 首 epoch、循环退出与最终成功时写入；取消或异常时尝试保存已完成历史，保存失败保留原取消或异常原因。进程被强制终止时，只能恢复最近快照 |

LRADI 在单次 solve 内按移位复用 LU，按最近使用顺序淘汰，最多 `maxShifts` 个因子，估计字节预算默认 64 MiB；`factorCacheBytes=0` 禁用缓存。稀疏分解预算包含值、索引和排列的估计，不是精确 RSS。独立 shifted solve 不产生跨调用缓存。

Torch 描述系统 LRSmith 每次 solve 仅分解 E 一次。FEM 批次采用成对稀疏合并，避免不断与增长的全局矩阵相加；单批几何数组仍随网格与批次规模增长。单元核在 NumPy CPU、全局累加可使用 Torch CPU/CUDA，详见 [BACKEND_ASSEMBLY.md](BACKEND_ASSEMBLY.md)。

缓存、采样索引和因子复用可能增加临时分配；小规模耗时不用于单独决定 auto。SVD 的 Gram 路径存在小谱精度风险，由底层缩放、阈值和直接法回退处理。

## 复现命令

性能报告环境安装 `.[all,reports]`，提供 threadpoolctl、matplotlib 和 Torch；缺库时失败，不静默减少报告场景或省略绘图。先看 `--help` 确认参数与输入要求。

```powershell
python scripts/generate_pipeline_performance_report.py --output reports/pipeline_performance --repeats 5
python scripts/generate_remaining_performance_report.py --baseline-root <优化前源码目录> --output reports/remaining_performance --repeats 5
python scripts/generate_validation_report.py --output reports/validation
```

第一个脚本默认测当前实现，增加 `--baseline-root` 可比较旧版本；第二个必须提供基线目录，比较 LRADI、Torch 描述求解与 FEM。基线按仓库相对路径保留脚本加载的旧模块，具体列表以生成脚本的 `loadBaseline` 调用为准，不把新兼容转发模块当作优化前实现。

比较时在相同依赖环境加载前后实现，预热、交替测量并核对输出等价，耗时采用中位数。通过 threadpoolctl 限制 BLAS 线程，Torch 限制线程；CUDA 在计时边界同步。报告保存输入规模、backend/device/dtype、重复次数、数值差异和内存口径。

## 内存与精度口径

| 指标 | 覆盖范围与限制 |
| --- | --- |
| tracemalloc 峰值 | 可追踪的 Python/NumPy 分配，不包含全部 Torch 原生分配或 CUDA 显存 |
| 进程峰值 RSS / Working Set | 包含依赖与原生工作区；前后指标差不等于精确新增分配 |
| CUDA allocated / reserved | Torch 设备分配与保留池，排除驱动内存；测量前初始化并重置峰值 |
| 分解输入预算 | 只限制密集化输入，不能代表分解工作区或进程峰值 |

POD 使用显式分块残差，训练验证按样本数计权，诊断在 float64 累计；前后比较应允许模型精度带来的舍入误差。sign 的候选残差与回退后实际残差分开统计；频响需在独立频点评价，DMD 噪声需覆盖多种子与不足激励。

性能表征不能替代正式回归，也不能把某个系统的低误差推广成通用上界。测试和数值报告入口见 [VALIDATION.md](VALIDATION.md)，更多生成器见 [scripts/README.md](../scripts/README.md)。

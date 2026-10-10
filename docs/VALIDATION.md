# 验证与复现

验证以当前源码、安装依赖和实际设备为准。回归测试保护接口与数值行为；报告脚本在更宽参数范围测量精度、成本和资源，生成结果放在忽略目录或仓库外。测试数量和耗时不作为固定文档契约。

## 环境与常规回归

核心依赖为 NumPy/SciPy，训练、网格、电机和部署组件按 [pyproject.toml](../pyproject.toml) 的 extras 安装。`dev` 包含测试和训练/数据组件，工作流测试还需 `deployment`。分包/构建环境负责装齐所选功能的依赖；测试直接导入，缺库时失败，不通过跳过掩盖环境问题。

```powershell
python -m pip install -e '.[dev,deployment]'
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -p no:cacheprovider -o addopts= -q
```

可将 `python` 替换为 `.venv/Scripts/python.exe`，确保安装与测试使用同一解释器。经典测试也包含 Torch、HDF5 和 sklearn 场景，使用 `.[dev]`；完整测试使用 `.[dev,deployment]`。延迟导入单独验证启动成本，不以缺库环境作为功能支持契约。外部 SLICOT 数据恢复见 [PROJECT_LAYOUT.md](PROJECT_LAYOUT.md)。

## 专项验证

| 范围 | 测试入口 |
| --- | --- |
| 核心算法、算子、模型、降阶 | `tests/algorithm/`、`tests/operators/`、`tests/models/`、`tests/test_reductors.py`、`tests/test_lrsmith_lowrank.py` |
| 后端格式、转换、设备和能力拒绝 | `tests/backends/` |
| FEM 跨后端组装与 Dirichlet | `tests/operators/test_backend_thermal_assembly.py`、`tests/operators/test_thermal_operator.py` |
| NN 数据、训练、重载与指标 | `tests/nn/` |
| 任务、输入类型与方法重载 | `tests/test_workflow_service.py`、`tests/test_workflow_input_kinds.py`、`tests/test_workflow_methods.py` |
| 文件存储与部署导出 | `tests/test_workflow_storage.py`、`tests/test_workflow_onnx.py` |
| 轻量性能与数值表征 | `tests/benchmarks/` |

CPU/CUDA 支持须分别验证实际运行路径。benchmark 测试检查数值和指标结构，不断言固定加速倍数。后端能力与预期拒绝见 [BACKEND_ASSEMBLY.md](BACKEND_ASSEMBLY.md)。

## 数值与原生内存报告

```powershell
python -m pip install -e '.[all,reports]'
python scripts/generate_validation_report.py --output reports/validation
```

报告生成 JSON/CSV/Markdown/TeX/SVG，在独立进程覆盖 sign 刚性范围、LRADI 输入数与秩增长、连续/离散频响、DMD 噪声/欠激励/病态映射、Hankel delay，以及 SciPy/Torch CPU/CUDA 的稀疏组装与约束。

数值正确性使用独立方程残差或解析参考；降阶近似和含噪辨识保留实际误差。sign 收敛标志不替代残差，直接法回退单独记录。LRADI 因子可能增长到满秩，Hankel 嵌入预算不覆盖 SVD 和结果存储。

Windows 进程峰值使用 PeakWorkingSetSize，其他平台使用 resource.ru_maxrss；包括依赖导入与原生库工作区。CUDA 报告 allocated/reserved，排除驱动内存。组装场景的总峰值还包含校验参考，不能当作纯组装峰值。tracemalloc 与 RSS/显存不可混为同一指标，计时方法见 [PERFORMANCE.md](PERFORMANCE.md)。

真实数据的转换和输入限制见 [STA 数据构建](data/motor_sta_dataset.md)。运行真实工况评估时，必须区分求解前已知输入与结果派生热流，按工况/几何隔离训练及评估，不把条件场重构视为未知接口整机预测。

## CI 与官方 Mamba

[tests.yml](../.github/workflows/tests.yml) 定义经典测试和包含训练、部署的完整测试环境，两类环境均覆盖 Linux/Windows，各环境安装对应测试的完整依赖；配置存在不等于远端已执行，执行状态以实际 CI 记录为准。

Mamba 本地 PyTorch 参考路径与 Linux CUDA 官方编译内核分别验证。[mamba-kernels.yml](../.github/workflows/mamba-kernels.yml) 是需要 GPU 自托管 runner 的手动验证入口。Windows 的参考路径测试不能证明官方内核通过，命令和依赖见 [MAMBA_DYNAMICS.md](MAMBA_DYNAMICS.md)。

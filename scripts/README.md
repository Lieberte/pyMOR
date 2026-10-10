# 脚本导航

这里保留 48 个命令入口和 4 个共享辅助模块，覆盖数据转换、训练推理、接口检查与可复现报告。入口路径保持稳定；脚本间共享函数和测试引用的模块保留在原位置。

从仓库根目录使用安装了项目的解释器运行，先查看参数：

```powershell
python scripts/build_motor_sta_dataset.py --help
python scripts/check_reduction_methods.py --help
python scripts/generate_validation_report.py --help
```

`python` 可替换为 `.venv/Scripts/python.exe`。依赖按 [pyproject.toml](../pyproject.toml) 安装；核心 MOR 使用 NumPy/SciPy，训练需 Torch，网格转换需 meshio，STA 数据需 h5py，分类基准需 `.[sklearn]`（sklearn/XGBoost）。性能生成器使用 `.[all,reports]`，包含 threadpoolctl/matplotlib；任务服务还需 deployment。分包保证依赖，脚本缺库时直接失败，不自动省略候选或图表。官方 Mamba 编译内核只在对应 Linux CUDA 环境检查。

输入数据和 checkpoint 由调用方准备。结果写到 `reports/<topic>/<run>/`，中间数据写到 `temp/experiments/<run>/`，或通过参数指定仓库外目录。`--help` 只展示参数；数据维护命令的 `--apply/--dry-run/--validate` 语义各不相同，应以对应帮助为准。

文档入口见 [docs/README.md](../docs/README.md)，目录约定见 [PROJECT_LAYOUT.md](../docs/PROJECT_LAYOUT.md)，测试与表征见 [VALIDATION.md](../docs/VALIDATION.md)，性能口径见 [PERFORMANCE.md](../docs/PERFORMANCE.md)。

## 经典 MOR 与性能报告

| 脚本 | 用途 |
| --- | --- |
| [fetch_mor_benchmark_data.py](fetch_mor_benchmark_data.py) | 下载并校验外部 SLICOT 数据 |
| [generate_classical_mor_latex_report.py](generate_classical_mor_latex_report.py) | 汇总经典算法证据为 LaTeX |
| [generate_dense_lyapunov_report.py](generate_dense_lyapunov_report.py) | 稠密连续 Lyapunov 表征 |
| [generate_dmd_report.py](generate_dmd_report.py) | DMD 变体表征 |
| [generate_lradi_report.py](generate_lradi_report.py) | 连续低秩 LRADI 表征 |
| [generate_lrsmith_report.py](generate_lrsmith_report.py) | 离散低秩 LRSmith 表征 |
| [generate_lyapunov_dispatch_report.py](generate_lyapunov_dispatch_report.py) | Lyapunov auto 路由表征 |
| [generate_pod_dmd_report.py](generate_pod_dmd_report.py) | POD-DMD 集成表征 |
| [generate_reduction_application_report.py](generate_reduction_application_report.py) | 生成热系统和可选外部标准案例的降阶表征 |
| [generate_svd_report.py](generate_svd_report.py) | SVD、加权分解及数值边界表征 |
| [generate_pipeline_performance_report.py](generate_pipeline_performance_report.py) | 几何、分解、训练、诊断与存储性能 |
| [generate_remaining_performance_report.py](generate_remaining_performance_report.py) | 与显式基线比较 LRADI、Torch 描述求解与 FEM 性能 |
| [generate_validation_report.py](generate_validation_report.py) | 独立进程的残差、频响、噪声、组装、RSS 与显存表征 |
| [generate_coupled_thermal_report.py](generate_coupled_thermal_report.py) | 部件热降阶与整机接口耦合的独立热源验证 |
| [generate_real_thermal_report.py](generate_real_thermal_report.py) | 从真实 STA 恢复物理组装，生成自主 POD 预测、原仿真对照及一步全阶验证 |

## 接口与引擎检查

| 脚本 | 用途 |
| --- | --- |
| [check_reduction_methods.py](check_reduction_methods.py) | 独立于上游数据流水线检查降阶接口 |
| [check_mamba_reference.py](check_mamba_reference.py) | 对比可信本地官方 Python selective_scan_ref 的输出与梯度 |
| [check_official_mamba.py](check_official_mamba.py) | Linux CUDA 官方编译内核的输出、梯度、优化与重载检查 |

## 热模型训练与评估

| 脚本 | 用途 |
| --- | --- |
| [train_thermal_discrete_pinn.py](train_thermal_discrete_pinn.py) | 分类热算子与可选物理损失训练；提供多个入口共享的训练辅助函数 |
| [train_thermal_heterogeneous_experts.py](train_thermal_heterogeneous_experts.py) | Global 模型与异构专家训练 |
| [calibrate_thermal_expert_gates.py](calibrate_thermal_expert_gates.py) | 标定 Global/Expert 融合权重 |
| [evaluate_thermal_heterogeneous_checkpoint.py](evaluate_thermal_heterogeneous_checkpoint.py) | 从 checkpoint 恢复契约后重新评估 |
| [benchmark_thermal_component.py](benchmark_thermal_component.py) | 通用元件数据的模型与物理残差比较 |
| [evaluate_motor_holdout.py](evaluate_motor_holdout.py) | 固定预算的独立 STA 工况和磁盘重载评估 |
| [run_motor_fem_dimensionless_holdout.py](run_motor_fem_dimensionless_holdout.py) | V2 无量纲、相似性增强与多数据集训练 |
| [run_motor_fem_speed_holdout.py](run_motor_fem_speed_holdout.py) | 跨转速训练与基线辅助函数，供 V2 和绝对均温入口复用 |
| [run_motor_fem_absolute_mean_holdout.py](run_motor_fem_absolute_mean_holdout.py) | 绝对均温和 POD 插值基线；相关行为由测试保护 |
| [run_thermal_operator_smoke.py](run_thermal_operator_smoke.py) | 合成跨几何训练闭环；benchmark 测试复用此实现 |

## 数据构建与维护

| 脚本 | 用途 |
| --- | --- |
| [build_motor_sta_dataset.py](build_motor_sta_dataset.py) | 单个 STA 的活动 FEM 元件与整机视图 |
| [build_motor_directory_dataset.py](build_motor_directory_dataset.py) | 数字工况目录的批量转换 |
| [build_motor_fem_component_dataset.py](build_motor_fem_component_dataset.py) | 选定 FEM 元件的数据转换 |
| [build_motor_full_resolution_dataset.py](build_motor_full_resolution_dataset.py) | 完整分辨率 point/graph 双视图 |
| [append_motor_sta_full_resolution.py](append_motor_sta_full_resolution.py) | 向完整分辨率数据集追加 STA |
| [build_motor_transient_series.py](build_motor_transient_series.py) | 构建瞬态 HDF5 时间序列 |
| [audit_motor_sta_energy_balance.py](audit_motor_sta_energy_balance.py) | STA 能量与接口守恒审查 |
| [inspect_motor_sta_archives.py](inspect_motor_sta_archives.py) | 检查 STA 归档结构 |
| [inspect_motor_sta_h5.py](inspect_motor_sta_h5.py) | 检查归档内 HDF5 |
| [validate_multi_motor_full_dataset.py](validate_multi_motor_full_dataset.py) | 检查完整分辨率多电机数据的一致性 |
| [normalize_thermal_component_types.py](normalize_thermal_component_types.py) | 按实例统一 JSON/HDF5 语义类别，并刷新任务索引 |
| [merge_motor_disk_component_type.py](merge_motor_disk_component_type.py) | 兼容旧数据缺少实例字段的 disk 类型迁移 |
| [rename_motor_dataset_cases.py](rename_motor_dataset_cases.py) | 重命名生成数据的 Case 标识 |
| [set_motor_assembly_hierarchy.py](set_motor_assembly_hierarchy.py) | 补充装配层级索引 |
| [set_motor_geometry_classification_target.py](set_motor_geometry_classification_target.py) | 声明几何分类任务目标 |
| [set_motor_thermal_analysis_labels.py](set_motor_thermal_analysis_labels.py) | 补充稳态/瞬态任务标签 |

## 几何识别

| 脚本 | 用途 |
| --- | --- |
| [build_geometry_component_recognizer.py](build_geometry_component_recognizer.py) | 建立统计描述符与原型库 |
| [predict_geometry_component.py](predict_geometry_component.py) | 统计原型 Top-K 与拒识推理 |
| [train_geometry_dgcnn.py](train_geometry_dgcnn.py) | DGCNN 训练、独立几何划分与可选未知样本暴露 |
| [predict_geometry_component_dgcnn.py](predict_geometry_component_dgcnn.py) | DGCNN checkpoint 推理 |
| [benchmark_geometry_classical.py](benchmark_geometry_classical.py) | 统计描述符上的经典分类器比较 |

## 共享辅助模块

| 脚本 | 用途 |
| --- | --- |
| [mor_application_cases.py](mor_application_cases.py) | 生成热系统、外部标准案例与降阶指标 |
| [standard_benchmark_cases.py](standard_benchmark_cases.py) | 标准矩阵与数值场景，供报告和测试复用 |
| [report_run_utils.py](report_run_utils.py) | 报告目录与场景清单工具 |
| [report_svg_utils.py](report_svg_utils.py) | 报告 SVG 绘图工具 |

真实电机接口检查 `check_reduction_methods.py --motor-data` 需要先生成 `applicationTests/shonTAmotor/aeResult.npz`；不带该选项的合成检查可直接运行。

41 个参数化入口支持 `--help`。经典报告中的 `generate_classical_mor_latex_report.py`、`generate_dense_lyapunov_report.py`、`generate_dmd_report.py`、`generate_lradi_report.py`、`generate_lrsmith_report.py`、`generate_lyapunov_dispatch_report.py`、`generate_pod_dmd_report.py` 和 `generate_svd_report.py` 使用源码内的场景及输出配置，没有参数解析；运行这些脚本（包括传入 `--help`）会直接开始生成报告。LaTeX 汇总需先生成其依赖的算法报告。生成结果不纳入 Git。

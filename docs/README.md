# 项目文档

这里维护当前代码的接口、数据格式和使用方法。所有运行命令以仓库根目录为工作目录；示例数据和 checkpoint 由调用方准备，生成结果不随源码提交。安装依赖以 [pyproject.toml](../pyproject.toml) 为准。

接手项目先阅读 [代码说明与交接指南](../README_CN.md)，其中包含安装、可运行示例、模块说明、已知问题和交接包范围。

## 核心与组装

| 文档 | 内容 |
| --- | --- |
| [后端与组装](BACKEND_ASSEMBLY.md) | 分层职责、dtype/设备、稀疏格式、LU 与能力范围 |
| [线性求解](LINEAR_SOLVERS.md) | 直接求解、可复用分解和无矩阵组装 GMRES |
| [离散 Lyapunov](DISCRETE_LYAPUNOV.md) | 方程、描述系统、残差与 auto 路由 |
| [DMD 契约](contracts/DMD_CONTRACT.md) | 快照、秩、控制辨识和 Hankel 内存预算 |
| [热物理算子](thermal/thermal_physical_operator.md) | 网格、材料、边界、接触与稳态/瞬态求解 |
| [部件降阶与热耦合](thermal/coupled_thermal_rom.md) | 基底投影、接触与油液节点、整机预测和守恒 |
| [仿射热 ROM](thermal/affine_thermal_rom.md) | 固定 Robin 空间模式、在线系数变化、接口输出和独立重载 |
| [电机应用层](../mor/applications/motor_thermal/README.md) | 线性热问题、描述系统装配、POD 封装和误差报告 |

## NN 与数据

| 文档 | 内容 |
| --- | --- |
| [NN 运行时](NN_RUNTIME.md) | 确定性、设备、训练配置、卷积 AE 与 VAE |
| [Mamba 动力学](MAMBA_DYNAMICS.md) | 参考/官方引擎、可选依赖、训练与 HPO |
| [热模型推理](thermal/THERMAL_CATEGORIZED_OPERATOR_CN.md) | checkpoint 数据契约、装配上下文与专家路由 |
| [无量纲热模型](thermal/thermal_dimensionless_v2.md) | 温度锚点、V2 特征、损失与相似性增强 |
| [热元件数据格式](data/thermal_component_case_format.md) | 物理数组、积分权重、保存与训练划分 |
| [STA 数据构建](data/motor_sta_dataset.md) | 活动部件、元件/整机视图与批量转换 |
| [全分辨率数据](data/motor_full_resolution_dataset.md) | point/graph 视图与实例/类型标识 |
| [几何识别](geometry/COMPONENT_RECOGNITION.md) | 当前描述符、原型、DGCNN 和拒识接口 |

## 集成与维护

| 文档 | 内容 |
| --- | --- |
| [Python 任务服务](integration/TASK_SERVICE.md) | 方法、输入、事件、取消、提交、重载与 ONNX |
| [目录与产物约定](PROJECT_LAYOUT.md) | 源码职责、外部输入和输出位置 |
| [验证与复现](VALIDATION.md) | 依赖环境、专项回归、CI 和报告生成 |
| [性能约定](PERFORMANCE.md) | 已实现策略、基准方法和内存指标范围 |
| [经典 MOR 优化流程](../OPTIMIZATION_PLAN.md) | 根目录保留的算法优化流程入口 |
| [脚本导航](../scripts/README.md) | 数据、训练、评估与报告脚本 |

实验精度、耗时、设备内存和模型权重随输入及运行环境变化，应保存到 `reports/<topic>/<run>/` 或仓库外目录。维护文档只说明方法与限制；运行结果由相应脚本重新生成。

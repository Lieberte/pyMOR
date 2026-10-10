# 项目目录与产物约定

仓库保留源码、回归测试、脚本、维护文档与应用样例必需的微型输入。

| 路径 | 职责 |
| --- | --- |
| `mor/backends/` | 数组格式、设备、dtype、数值库操作、分解与后端消元 |
| `mor/assembly/` | 单元核、全局稀疏累加与热组装 |
| `mor/operators/` | 算子表示、结构信息、apply 与残差 |
| `mor/algorithm/` | 求解策略、收敛、缓存预算与 auto 路由 |
| `mor/models/`、`mor/reductors/`、`mor/solvers/` | 系统模型、降阶与求解接口 |
| `mor/nn/` | NN 数据、模型、训练、验证与几何适配 |
| `mor/workflow/` | 任务协议、执行、存储、重载与导出 |
| `tests/` | 单元、集成、算子契约与 benchmark smoke 测试 |
| `scripts/` | 数据构建、训练、评估、检查与报告生成 |
| `docs/` | 当前接口、数据格式与使用文档 |
| `applicationTests/` | 应用样例代码与微型输入 |
| `OPTIMIZATION_PLAN.md` | 经典 MOR 优化流程，供仓库技能引用 |

后端和组装职责见 [BACKEND_ASSEMBLY.md](BACKEND_ASSEMBLY.md)，任务入口见 [TASK_SERVICE.md](integration/TASK_SERVICE.md)。

## 电机 FEM 数据模块

`mor/nn/data/motorFemComponentCase.py` 保留案例构建、批量构建和报告入口，共享逻辑按职责拆分：

| 模块 | 职责 |
| --- | --- |
| `motorFemMetadata.py` | 部件命名、FEM 系统定位、稳态与瞬态元数据 |
| `motorFemGeometry.py` | 连通性、体积、边界法向与守恒采样 |
| `motorFemBoundary.py` | 边界类型、边界输入与 surface patch 样本 |
| `motorFemIO.py` | HDF5、材料属性、结果文件与热流读取 |

旧案例模块兼容既有辅助函数导入；新代码引用职责对应的模块。

## 运行产物与外部输入

结果建议写入 `reports/<topic>/<run>/`，中间数据写入 `temp/experiments/<run>/`，也可指定仓库外输出目录。这些生成产物不纳入 Git，避免将模型、完整数据集、压缩包或本机试跑结果放在仓库根目录。

文档示例路径不表示仓库附带数据或模型。推理需提供兼容 checkpoint；真实网格、快照、电机结果与 GUI 案例需由调用方准备。

正式测试一般通过 `tmp_path` 生成输入。外部 SLICOT MAT 数据存放在 `tests/data/slicot/`，不纳入 Git；缺失时对应应用测试跳过，下载与摘要校验见 [tests/data/slicot/README.md](../tests/data/slicot/README.md)。

`applicationTests/oscillatingWake/generateSnapshots.py` 可生成合成快照。电机样例需要求解器原始数据；`scripts/check_reduction_methods.py --motor-data` 需先生成 `applicationTests/shonTAmotor/aeResult.npz`，不带该参数的合成检查可直接运行。脚本分类见 [scripts/README.md](../scripts/README.md)。

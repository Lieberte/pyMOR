# 电机完整分辨率双版本数据集

使用以下命令生成不下采样的数据集，输入目录由调用方提供：

```powershell
.\.venv\Scripts\python.exe scripts\build_motor_full_resolution_dataset.py `
  "input\motor_cases" `
  "temp\experiments\geometrymotor" `
  --motor-id motor-htc-prototype
```

输出结构：

```text
geometrymotor/
├─ views.json
├─ point/
│  ├─ dataset.json
│  ├─ motors/<motorId>/cases/<caseId>/data.npz
│  └─ whole_motor/
└─ graph/
   ├─ dataset.json
   ├─ motors/<motorId>/cases/<caseId>/data.npz
   └─ whole_motor/
```

`point/` 保存完整体单元中心、边界点、全部温度查询节点和积分权重，不保存连接关系，适合点集网络、DeepONet 等方法。

`graph/` 保存完整节点、四面体连接、表面三角形连接和边界样本到节点的映射，适合 Mesh GNN、有限元图网络等方法。训练时可由四面体连接派生图边，无需在磁盘中重复保存边数组。

两个版本的每个 Case 都是自包含文件，不引用全局 `assets/`。这样不同电机可以独立导入、复制和删除，不会产生跨电机资产引用。代价是同一网格在不同工况中会重复保存。

完整构建器不执行随机下采样，不重新划分网格，也不改变原始数字目录。
实例与类型分开保存：`componentInstance` 标识用于组装和接口关联的具体 FEM region，`componentType` 是识别与专家模型使用的语义类别。例如 disk1/disk2 保留独立实例和 Case ID，两者的类型均为 `magnetic_isolation_plate`。

几何识别读取 graph 视图中的几何数组，接口见 [COMPONENT_RECOGNITION.md](../geometry/COMPONENT_RECOGNITION.md)。

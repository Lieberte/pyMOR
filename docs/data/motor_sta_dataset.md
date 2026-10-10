# 单个电机 STA 数据集构建

`scripts/build_motor_sta_dataset.py` 将一个完整 `.sta` 求解归档转换成一个单电机数据集。数据集中的每一项是一个独立 FEM 元件，元件之间的接口关系继续保存在 Case 元数据和 surface patch 元数据中。

默认流程：

1. 检查 STA 中的 `controlData.json/h5`、FEM 网格和结果文件。
2. 多个数值结果存在时选择数值最大的结果文件作为最终结果。
3. 如果 STA 旁边存在同名且内容完整的解压目录，则直接复用；否则只解压转换所需文件。
4. 自动发现所有活动 FEM region（`isActive=false` 的停用部件不参与数据构建），并逐个生成 `thermalComponentCase`。
5. 写入元件训练视图和整机训练视图，两种视图引用同一批元件数组。

示例（输入归档由调用方提供，输出目录在运行时生成）：

```powershell
.\.venv\Scripts\python.exe scripts\build_motor_sta_dataset.py `
  "input\motor.sta" `
  "temp\experiments\motor_dataset" `
  --motor-id motor-prototype-a `
  --case-id baseline `
  --maximum-volume-samples 4096 `
  --maximum-boundary-samples 4096 `
  --maximum-query-samples 2048 `
  --verbose
```

默认导入全部活动 FEM 元件。旧控制文件未提供 `isActive` 时按活动部件处理。`availableComponents`、`completeComponentSet` 按活动部件集合定义，停用实例记录在 `inactiveComponents`；停用部件无需网格或结果，显式选择它时会报错。活动部件缺少温度/热流结果时报告部件名称、UUID 与结果文件。只导入指定元件：

```powershell
.\.venv\Scripts\python.exe scripts\build_motor_sta_dataset.py `
  "input.sta" "motor_dataset" `
  --components stator winding magnets
```

结果选择可以显式指定：

```powershell
--result-time 274
```

或：

```powershell
--result-member "results/system/274.000000.h5"
```

输出结构：

```text
motor_dataset/
├─ dataset.json
├─ views.json
├─ whole_motor/
│  ├─ dataset.json
│  └─ cases/
│     └─ baseline/
│        └─ manifest.json
└─ cases/
   ├─ baseline-stator/
   │  ├─ manifest.json
   │  └─ fields.npz
   ├─ baseline-winding/
   └─ baseline-magnets/
```

根目录的 `dataset.json` 是元件训练视图。每个 FEM region 是一个独立 Case，可以使用结果提供的等效接口边界训练单元件代理模型。

`whole_motor/dataset.json` 是整机训练视图。它通过相对路径引用根目录 `cases/` 中的同一批元件数组，并在整机 manifest 中显式保存：

- 元件列表和 FEM region 对应关系；
- 成对的内部接口及其两侧 Patch；
- 外部边界；
- 接口热流守恒误差；
- 整机训练时需要屏蔽的结果派生边界 Patch。

内部接口热流和温度在整机视图中属于监督目标，不作为端到端整机模型输入。`views.json` 给出两个训练视图的入口。

该阶段使用结果提供的等效接口热流或参考温度，验证的是“已知闭合边界条件时，单 FEM 元件能否可靠重建温度场”。它不等价于未知接口条件下的整机端到端预测。

## 批量转换数字目录

已经整理成 `2500/`、`5000/`、`7500/` 等数字目录时，可以一次生成多父 Case 数据集：

```powershell
.\.venv\Scripts\python.exe scripts\build_motor_directory_dataset.py `
  "input\motor_cases" `
  "temp\experiments\motor_numeric_dataset" `
  --motor-id motor-htc-prototype `
  --maximum-volume-samples 4096 `
  --maximum-boundary-samples 4096 `
  --maximum-query-samples 2048
```

每个数字目录形成一个父 Case，每个父 Case 自动拆分全部 FEM 元件。根目录组件视图包含所有元件 Case，`whole_motor/` 视图包含对应数量的整机 Case。

# Python 任务服务

`mor.shonta` 提供面向调用方的 JSON 请求、事件、结果和模型重载接口。每个请求执行一个方法节点，同进程最多接收一个并发任务；可选图快照用于审计，不执行完整 DAG。

## 公开入口

```python
from mor.shonta import (
    getCapabilities, validateWorkflow, executeWorkflow,
    loadPodModel, loadAutoencoderModel, loadDynamicsModel,
)
```

`getCapabilities()` 返回 `apiVersion=1`、方法、参数、输入类型与资源限制。方法按随包能力固定声明，不探测依赖，也不初始化 Torch；保留的 `available/reason` 字段为 true/None，`availabilityCheck` 标记为 `packaged-dependencies`。构建/分包负责装齐依赖，缺库时执行失败。依赖集合以 [pyproject.toml](../../pyproject.toml) 为准。

| method | inputKind | 数值执行 | 模型重载 |
| --- | --- | --- | --- |
| `mock` | 无需数值输入 | 生命周期检查 | 无数值模型 |
| `pod` | `snapshots` / `timeSeries` | SciPy CPU，float64，未加权 POD | `loadPodModel` |
| `autoencoder` | `snapshots` / `timeSeries` | Torch CPU，float32，全连接 ReLU AE，Adam | `loadAutoencoderModel` |
| `dmd` | `timeSeries` | SciPy CPU，时间序列辨识 | `loadDynamicsModel` |
| `pod-dmd` | `timeSeries` | SciPy CPU，POD 后辨识 | `loadDynamicsModel` |
| `bt` | `ltiSystem` | SciPy CPU，平衡截断 | `loadDynamicsModel` |

通用 NN 流水线可配置的 CUDA、网络和优化器不等于任务服务已暴露这些设置，见 [NN_RUNTIME.md](../NN_RUNTIME.md)。

## 请求与输入

以下路径为调用方准备的示例绝对路径；输入文件和输出父目录须已存在，任务输出目录须尚不存在。

```json
{
  "apiVersion": 1,
  "taskId": "846c8c22-4cb6-4c61-a6e2-3c97c72fe913",
  "caseId": "example",
  "nodeId": "node-1",
  "method": "pod",
  "inputKind": "snapshots",
  "eventPolicy": "stream",
  "parameters": {"rank": 2, "centerSnapshots": true, "verbose": true},
  "inputs": {
    "snapshots": {
      "path": "D:/cases/example/snapshots.npy",
      "format": "npy",
      "axes": ["state", "snapshot"]
    }
  },
  "outputDirectory": "D:/cases/example/rom-runs/846c8c22-4cb6-4c61-a6e2-3c97c72fe913"
}
```

每次执行生成新的 `taskId`。同进程拒绝重复接收的 ID，输出目录独占创建，防止覆盖。ID 历史不跨进程持久化。`taskId/nodeId` 最长 128 个字符，首字符为 ASCII 字母或数字，后续可用字母、数字、`_ . : -`。`caseId` 最长 2048 个字符，可含中文与空格，不用于拼接目录。

| 输入 | 契约 |
| --- | --- |
| `inputs.snapshots` | 非空二维实数数值 NPY，轴为 `[state, snapshot]`，禁止 pickle 和非有限值 |
| `inputs.sequence` | DMD/POD-DMD 必填；正有限 `dt`，可选递增且唯一的相邻列索引 `pairs` |
| `inputs.system` | BT 必填；`matrices` 包含 A/B/C，可选 D/E，各引用 NPY，轴为 `[row, column]`；`isContinuous` 默认为 true，离散系统须给正有限 `dt` |

`inputKind` 缺省为 `snapshots`，DMD/POD-DMD/BT 必须显式选择兼容类型。NPY 引用可声明 `shape/dtype/sha256`，执行加载时核对。输入复制到任务目录后计算摘要，复制期间检查源文件变化；后续计算使用任务副本。

单份数值输入预算为 256 MiB，估计峰值预算为 2 GiB；系统输入还有合计预算和矩阵维度检查。这些是准入限制，不能保证实际工作区或进程 RSS 不超预算。`validateWorkflow(request)` 返回 `valid/request/errors`，只完成结构、参数和路径检查；有限性、数组内容及部分资源检查在执行加载阶段完成。

## 方法参数

参数的默认值和范围应从 `getCapabilities()` 读取，未知参数会被拒绝。

| 方法 | 参数与含义 |
| --- | --- |
| Mock | `steps/delayMs/failAtStep/verbose`；`failAtStep` 从 1 开始 |
| POD / DMD | `rank` 或 `tol` 互斥；tol 是丢弃的奇异值平方能量比例 `[0,1)`；POD 可设置 `centerSnapshots` |
| POD-DMD | `podRank/podTol`、`dmdRank/dmdTol` 分别互斥；可设置 `centerSnapshots` |
| BT | 正整数 `order` 或正有限 `maxError` 互斥 |
| AE | `latentDim/hiddenDim/epochs/batchSize/learningRate/seed/verbose`；不自动归一化输入 |

DMD 指标评价训练数据的一步预测，不代表独立轨迹泛化。算法级 DMDc、加权 POD 等入口见 [DMD 契约](../contracts/DMD_CONTRACT.md)，不属于此服务的方法集合。

## 图快照、事件与取消

可选 `workflow` 须使用 `schemaVersion=1`、`executionScope='selectedNode'`，`selectedNodeId` 与 `nodeId` 一致。选中节点为 `rom.reductionMethod`，其 `romMetadata.methodName` 和补齐默认值后的 `parameters` 须与任务一致。

图由 `nodes/connections` 数组组成，节点有唯一 `uuid`，连接为 `from/to: {nodeUuid, portId}`。只允许无上游或一个直接 `rom.dataInput` 上游，端口为 `data → trainingData`；拒绝多输入、预处理链和其他方法上游。其余节点只保留审计信息。实际输入使用请求中的文件引用。

`executeWorkflow(request, onEvent=None, isCancelled=None)` 返回 JSON 结果。事件包含 `apiVersion/taskId/nodeId/sequence/timestamp/type/payload`；sequence 在单次任务内严格递增，type 为 `log/metrics/state`。`eventPolicy='stream'` 默认传递过程事件，`finalOnly` 只生成终态事件，事件文件也不保留中间过程事件。

终态为 `succeeded/failed/cancelled`，最多发送一次。结果含 `state/modelReference/files/error/summary`；错误含 `code/message/stage/traceback`，结构校验失败还含 `details`。跨语言只传 JSON 与路径。

取消通过阶段和复制块检查点协作完成，AE 训练也检查取消；同步分解和底层数值库调用不能内部抢占。清单提交前进行最后取消检查，提交后不把成功结果改成取消。提交前观察者异常终止任务，提交后的终态观察者异常记录警告。回调收到独立 JSON 副本。

## 文件、重载与 ONNX

公共产物为 `submittedRequest.json/request.json/events.jsonl/result.json`。输入副本和模型数组按方法保存；完整训练历史按周期及退出时保存。JSON/NPY 通过同目录临时文件、刷新和原子替换写入，NPY 写入时累计 SHA-256。

`manifest.json` 最后提交，是模型可用标志。失败或取消可以保留诊断和部分数组，调用方须同时核对清单和成功终态，不能只凭文件存在认定成功。加载器校验版本、任务身份、路径、摘要、形状、dtype 和模型约束。任务目录可整体移动；读取以实际清单目录为基准。

各模型加载器共享 JSON 读取和提交状态校验，单份元数据上限为 1 MiB。版本必须为整数，拒绝布尔值；任务和节点身份必须为非空字符串并与成功终态一致，终态须引用 `manifest.json`。

| 重载对象 | 使用方法 |
| --- | --- |
| `PodModel` | `project(snapshots)` / `reconstruct(coefficients)`，中心化时自动加回均值 |
| `AutoencoderModel` | `project(snapshots)` / `reconstruct(coefficients)`，CPU 分批推理 |
| `DynamicsModel` | `project(states)` / `reconstruct(coordinates)` / `step(states)` |
| `BalancedModel` | `step(states, inputs, dt=None)`；连续系统需正 dt，使用零阶保持离散化；`output(states, inputs)` 计算输出 |

任务服务分包包含 `deployment` 依赖，服务导出并检查 ONNX，清单 `inference` 记录可用性、文件与语义。POD/AE 导出重构，DMD/POD-DMD 导出一步预测，BT 导出下一步状态和当前输出，需调用方提供 `transition` 状态推进矩阵。DMD 必须保留完整共轭模态对才可转为实数 ONNX。不满足数值导出约束时记录原因；缺少或损坏的 ONNX 依赖导致任务失败，不降级成成功。

实现见 [mor/workflow](../../mor/workflow/) 和 [mor/shonta.py](../../mor/shonta.py)，验证入口见 [VALIDATION.md](../VALIDATION.md)。

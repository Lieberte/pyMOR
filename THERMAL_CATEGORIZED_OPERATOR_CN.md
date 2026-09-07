# 离散稳态热模型说明

## 1. 推荐模型

当前推荐 checkpoint：

```text
_tmp_thermal_heterogeneous_final/thermal_global_heterogeneous_experts.pt
```

该模型面向无结构 FEM 点集，不依赖规则网格。最终结构不是单一网络，而是：

```text
完整父工况的元件点集
  -> 装配级热源、边界、温度锚点和接口邻居上下文
  -> Global Set-Attention 保底模型
  -> 按 componentType 精确路由的异构专家
  -> 验证集标定的 Global/Expert 融合
  -> 各查询点绝对温度
```

分类键只有 `componentType`。上游目录、系统目录和零件来源不参与路由；所有规范化为
`housing` 的壳体进入同一个专家。没有专家或遇到新元件类型时自动使用 Global 模型。

## 2. 为什么采用这套结构

本轮在相同父工况切分、相同输入和相同评价口径下测试了 RBF-MLP、ResNet、Dense、
点集 U 形 decoder、PointNet/global、DeepONet、cross-attention、RBF-attention hybrid、
Set-DeepONet 和 Set-Attention。Global 最优候选是 `Set-Attention h64`；不同元件的最优
归纳偏置不同，因此最终专家允许使用 Set-Attention、Set-DeepONet 或 Query-Set。

卷积 U-Net 和 FNO 没有纳入最终模型，因为当前输入是无规则、不同点数的 FEM 点集，
不存在可直接卷积的统一规则网格。GNN 需要稳定的网格连通关系，而当前修正数据使用
point view，不能用人为近邻图冒充原始 FEM 拓扑。已有的“U-Net”候选是点集 decoder
内部的平滑 U 形跳连，不是二维或三维卷积 U-Net。

ResNet 可以训练，也通过了 PINN 二阶导数测试，但本数据上没有超过 Set-Attention。
原因不是 ResNet 与热问题冲突，而是残差块只增强逐查询点 decoder；当前主要误差来自
跨元件边界条件、装配上下文和点集交互，这些由 attention/set encoder 更直接地处理。

## 3. 输入契约

推理必须提供同一父工况下可获得的全部元件，至少包含：

- 体点坐标、导热率、密度、比热、体热源和体积权重；
- 边界点坐标、法向、HTC、环境温度、热流、边界类型、面积权重和 patch id；
- `surfacePatches` 中的接口、对侧 region 和接触导热信息；
- `componentType`、`assemblyId`/`motorId`、`parentCaseId` 和 `regionUuid`；
- 转速等运行参数；
- 查询点坐标。

装配上下文只使用推理时可知的几何、材料、热源和边界数据，不读取温度目标。当前
`thermalComponentCase` 仍通过 `temperature.coordinatesMeters` 承载查询点；有绝对温度
锚点时，其中的 `temperatureK` 数值不会进入模型输入，可放占位值。

绝对温度参考按以下顺序确定：固定温度边界优先；否则使用 `HTC * area` 加权的环境温度。
参考温度从完整父工况传播到拆分元件，因此 stator 即使本地没有对流面，也可以利用
整机 housing 上的对流温度锚点。若整个装配确实没有任何温度锚点，runtime 默认拒绝输出
绝对温度，只允许显式请求温差预测。

checkpoint 保存了元件词表、特征顺序、边界类型、无量纲模式和温标。benchmark、gate
校准、PDE 复评和 runtime 都从 checkpoint 恢复同一数据契约，并在输入维度不一致时立即
报错，不能再依赖脚本内的隐式默认值。

## 4. 路由与专家

最终模型包含 10 个有验证支持的元件专家：

| 元件 | 专家 | Expert 权重 |
|---|---|---:|
| bearing | Set-Attention | 0.6 |
| end_cap | Set-Attention | 1.0 |
| end_plate | Set-Attention | 0.8 |
| housing | Set-Attention | 1.0 |
| magnetic_isolation_plate | Set-Attention | 1.0 |
| magnets | Query-Set | 0.6 |
| rotor_core | Set-DeepONet | 0.7 |
| shaft | Set-Attention | 1.0 |
| stator | Set-DeepONet | 0.9 |
| winding | Set-Attention | 0.7 |

`gear`、`oil_tank`、`sleeve` 等缺少足够训练样本或没有可靠专家的类型走 Global fallback。
41 个 held-out case 中，21 个使用 Global/Expert 融合，17 个使用纯专家，3 个使用 Global。

## 5. 独立复验指标

数据为修正热源边界后的 194 个稳态 case，按父工况切分为 153 个训练 case 和 41 个验证
case。以下数值来自 checkpoint 反序列化后的重新计算，而不是训练过程缓存：

| 模型 | 加权 RMSE/K | Macro RMSE/K | P95/K | 最大误差/K | mesh-free PDE |
|---|---:|---:|---:|---:|---:|
| Global Set-Attention | 24.6366 | 17.2424 | 56.6813 | 85.4315 | 34.6140 |
| 最终异构路由模型 | 10.8950 | 6.3232 | 27.7192 | 39.3066 | 28.4376 |

Stator 单独结果：加权 RMSE `4.5549 K`，Macro RMSE `3.0532 K`，P95 `7.1358 K`，
最大误差 `16.6689 K`。

最大误差仍主要来自少样本或分布差异较大的 shaft、gear 和 winding case，而不是热流
重新标定错误。当前验证的架构选择、早停和融合权重使用了同一批 4 个 held-out 父工况，
所以指标适合模型筛选，但可能偏乐观。正式泛化结论应增加独立父工况测试，或执行按
父工况的交叉验证。

## 6. 推理示例

```python
from mor.nn.data import loadSubsampledThermalDataset
from mor.nn.runtime import thermalCategorizedPredictor

dataset = loadSubsampledThermalDataset(
    '_tmp_stator_connection_dataset_full/dataset.json',
    analysisType='steady',
    maximumVolumeSamples=128,
    maximumBoundarySamples=256,
    maximumQuerySamples=128,
    randomSeed=23,
)
predictor = thermalCategorizedPredictor.fromCheckpoint(
    '_tmp_thermal_heterogeneous_final/thermal_global_heterogeneous_experts.pt',
    device='cuda',
)
predictions = predictor.predictCases(dataset.cases)
```

每个返回项包含查询坐标、绝对温度、温差、参考温度、温标和实际路由信息。生产调用时
应按完整父工况批量传入元件，不能只传一个被拆离装配上下文的 stator。

## 7. 复验命令

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_thermal_heterogeneous_checkpoint.py `
  _tmp_stator_connection_dataset_full\dataset.json `
  --checkpoint _tmp_thermal_heterogeneous_final\thermal_global_heterogeneous_experts.pt `
  --output _tmp_thermal_heterogeneous_final\evaluation_refactored.json `
  --device cuda --seed 23
```

复验输出位于 `_tmp_thermal_heterogeneous_final/evaluation_refactored.json`。

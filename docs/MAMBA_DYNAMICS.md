# Mamba 动力学与参数搜索

`mambaDynamicsTorch` 和兼容名称 `ssmDynamicsTorch` 已注册到 models。输入为 `(batch, time, inputDim)`，时间轴不能为空；输出为 `(batch, time, outputDim)`。当前每个完整序列从零状态开始，不提供跨批次状态缓存或官方推理 cache。

模型由输入映射、若干预归一化残差 Mamba 层和输出映射组成。Mamba 层包含因果深度卷积、SiLU、输入相关的 delta/B/C、负指数状态矩阵、选择性扫描、D 跳连和门控。默认参数为 dModel=64、stateDim=16、convWidth=4、expand=2、numLayers=2。

## 两种引擎

`engine='reference'` 是默认的本地 PyTorch 实现，支持 Windows/Linux、CPU/CUDA 和 float32/float64。它按时间循环以减少一次构造完整扫描中间量；训练反向图仍随序列长度增长，不提供官方融合内核性能。没有隐式切换引擎。

`engine='mamba-ssm'` 显式使用官方 Mamba-1 模块。官方包要求 Linux、NVIDIA GPU 和 CUDA；模型要求 CUDA float32/float16/bfloat16 输入。Windows 明确报平台错误，缺依赖直接暴露导入错误，分包需包含对应依赖。官方编译内核须在对应 Linux CUDA 环境单独验证。

可选依赖固定 mamba-ssm==2.2.6.post3，避免自动升级到不同内核接口。在 Linux 安装 CUDA-enabled PyTorch 后执行：

```sh
python -m pip install '.[mamba]' --no-build-isolation
```

Windows 的 mamba extra 只安装 Torch/einops，不尝试编译官方 Linux 扩展。`einops` 用于独立官方参考检查；本地模型前向不依赖它。参考：[官方安装说明](https://pypi.org/project/mamba-ssm/)、[固定版本](https://pypi.org/project/mamba-ssm/2.2.6.post3/)。

## 训练与验证

```python
from mor.nn.configs.dynamicsLearningConfig import dynamicsLearningConfig
from mor.nn.reductors.dynamicsLearningPipelineTorch import runDynamicsLearningTorch

config = dynamicsLearningConfig()
config.modelName = 'mambaDynamicsTorch'
config.modelParams = dict(inputDim=2, outputDim=2, dModel=16, stateDim=8,
                          numLayers=2, engine='reference')
config.dataModuleName = 'trajectoryDataModule'
config.lossFunction = 'oneStepPredictionLoss'
config.validationName = 'dynamicsPredictionMetrics'
config.runtime.deviceName = 'cpu'
# inputs / targets 是按时间排列的 (sample, time, 2) 数据。
result = runDynamicsLearningTorch(config, inputs, targets, returnModelState=True)
```

回归覆盖独立标量递推、gradcheck、因果性、训练和状态重载、float32/float64，以及确定性与 RNG 恢复。CPU/CUDA 以实际可用设备分别验证。官方 Python selective_scan_ref 的对比检查输出和参数梯度，不能替代编译内核验证。

使用可信本地官方源码复现：

```powershell
.venv/Scripts/python.exe scripts/check_mamba_reference.py --source <selective_scan_interface.py> --output reports/mamba_reference/reference.json
```

此脚本只加载指定文件中的 selective_scan_ref 函数，普通 pytest 不访问网络。

## 动力学 HPO

`dynamicsHpo(searchSpace, strategy='grid', maxTrials=None, randomSeed=0, direction='minimize', continueOnError=False)` 对有限候选列表搜索。随机模式使用局部种子、抽样组合索引且不重复，不预先构造整个笛卡尔积。参数结构通过深复制传给 objective，试验快照不会被目标函数修改。

objective 接收参数字典，返回有限实数评分。可在其中复制 config、调用 runDynamicsLearningTorch 并返回验证 mse。默认目标异常直接抛出；continueOnError=True 记录失败并继续，全部失败仍抛错，lastTrials 保留诊断。返回 bestParams、bestScore、trials、strategy、direction；相同评分保留先出现的试验。

```python
from mor.nn.hpo import dynamicsHpo

search = dynamicsHpo({'learningRate': [1e-3, 1e-2], 'stateDim': [8, 16]},
                     strategy='random', maxTrials=3, randomSeed=7)
result = search.optimize(objective)
```

搜索器不自动划分数据、重训最佳模型或保存 checkpoint；独立验证集与目标定义由调用者提供。贝叶斯搜索、并行调度和跨真实工况泛化不属于当前实现。

## 官方编译内核验证

`scripts/check_official_mamba.py` 针对 mamba-ssm 2.2.6.post3，要求 Linux、CUDA 和实际可导入的 `selective_scan_cuda` 编译扩展。比较长度 1/17/128 的输出、按参数名称匹配的梯度、优化步骤及保存状态重载，并输出 JSON。`.github/workflows/mamba-kernels.yml` 提供手动触发入口，需要已有 `linux/x64/gpu` 自托管 runner、CUDA Torch 和 nvcc。Windows 本地参考验证不包含这个编译内核；提供工作流不等于远端已经执行。

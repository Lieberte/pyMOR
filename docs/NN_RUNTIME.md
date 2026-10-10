# NN 训练和模型契约

## 训练上下文

`runRepresentationLearningTorch` 与 `runDynamicsLearningTorch` 在受保护的训练上下文内应用 `randomSeed`、`deterministic` 与 cuDNN 设置。函数退出或抛错后，恢复进入时的 CPU/CUDA RNG 和确定性配置；上下文锁只串行化这些流水线之间的全局配置操作。

CUDA 确定性训练在未指定 `CUBLAS_WORKSPACE_CONFIG` 时暂用 `:4096:8`，退出后恢复。确定性模式仍可能明确拒绝 PyTorch 不支持确定性执行的算子，并不保证不同硬件、库版本之间逐位一致。

`deviceName='cuda'` 是严格请求，不可用时直接失败。`auto` 可选择 CPU；若自动选中的 CUDA 在模型迁移时失败，只有 `deviceAutoFallback=True` 才尝试迁移至 CPU。运行时记录实际训练设备；显式 CUDA 的错误不会被该选项隐藏。

动力学训练与 AE 共享优化步骤、梯度裁剪、样本计权、checkpoint 和早停逻辑。轨迹数据保留原有浮点 dtype；验证按模型参数的实际 device/dtype 分批运行，使用 float64 累计平方误差，拒绝空数据和输出形状不匹配。

两条训练流水线通过 `mor/nn/reductors/pipelineConfig.py` 共享 point/operator 模式校验和默认训练器选择；operator 模式声明的 `requiredCellTypes` 必须包含在 `supportedCellTypes` 中。

## 卷积 Autoencoder

输入为 `(sample, inputChannels, height, width)`。可通过 `inputShape=(height, width)` 固定形状，也可由第一次 encode 推断。模型形状确定后必须保持一致；新形状应建立新模型。

解码根据各编码层的空间尺寸确定转置卷积输出，因此支持奇数和非正方形尺寸。空间形状作为 buffer 保存于 state dict。使用 `saveState` / `loadState` 或 `state_dict` / `load_state_dict` 恢复后，可直接 encode/decode，无需先提供样例数据。

解码器在第一次使用时初始化线性参数；首次独立 decode 需要构造时指定 `inputShape`。缺少形状 buffer 的旧 checkpoint 若编码/解码线性层均对应 4×4 编码网格，可自动按旧固定输出边长迁移。其他旧权重需显式 inputShape；编码与解码权重尺寸不一致或尚未初始化时明确拒绝，无法无损恢复原先错误形状。

## VAE

训练模式保持随机重参数化。评估模式使用潜变量均值 `mu` 重构，从而能重复比较验证指标。输出仍为 `(reconstruction, mu, logVar)`。

`reconstructionMetrics` 用第一个张量计算重构误差，给 loss 传入完整输出结构，保留 VAE 的 KL 项；诊断张量在 float64 计算。VAE 应配合 `vaeReconstructionLoss`，普通 AE 配合张量 loss。

## 能力边界

`ssmDynamicsTorch` 现在是 `mambaDynamicsTorch` 的兼容入口，两者均已注册。详细引擎、依赖、输入和 HPO 契约见 [Mamba 与动力学搜索](MAMBA_DYNAMICS.md)。

RNN、GRU、LSTM、Transformer 已有小规模 CPU/CUDA、float64、训练、验证和状态重载回归。热 mesh GNN、query set、set attention 已有小规模训练和跨设备状态回归。这些检查不代表跨真实工况的泛化验证。

## 训练配置

通用 AE/动力学训练器支持 Adam、AdamW、SGD、RMSprop、Adagrad，以及 none、step、multistep、cosine 调度。构造时验证 scheduler 名称、stepSize/tMax、gamma/etaMin、milestones、日志间隔、dtype 和监控指标/方向；未知配置明确报错。`reportTrainLoss` 与 `reportValidationLoss` 控制对应日志字段，`logInterval` 控制记录间隔。优化器与调度器的学习率历史组合由 `test_training_configuration.py` 验证。

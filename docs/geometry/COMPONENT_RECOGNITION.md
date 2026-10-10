# 几何元件识别

几何识别仅使用形状预测 `componentType`。`componentInstance` 表示具体 FEM 实例；例如 disk1/disk2 保留各自实例标识，语义类型均映射为 `magnetic_isolation_plate`。

## 数据与归一化

读取完整分辨率数据的 graph 视图，格式见 [motor_full_resolution_dataset.md](../data/motor_full_resolution_dataset.md)。几何输入为 `nodesMeters [n,3]`、`surfaceTriangles [m,3]`，可选 `elementCentersMeters [k,3]` 与 `elementVolumesM3 [k]`，两者必须同时提供；识别适配器不直接读取四面体连接。温度、热流、HTC、材料、转速及绝对装配位置不进入识别特征。

`meshUuid` 标识独立几何；重复工况按此去重。缺少该标识时使用 `geometryId`。同一几何的工况、重网格和采样副本应保持在同一训练/验证组，标签冲突明确拒绝。

表面按三角面面积采样，使用体积或表面积加权中心消除平移，RMS 尺度消除整体尺寸。采样数可配置；采样、凸包和直方图使特征具有数值近似误差，不能把它们描述成所有重网格或旋转下逐位不变。

## 统计描述符与原型

`geometryComponentRecognizer` 的默认 V2 描述符为 128 维，包含惯性与包围盒比例、径向和轴向统计、点对距离、法向夹角、表面连通分量面积，以及凸包/包围盒比值。

训练库提供特征均值和尺度，标准化后 L2 归一化。每个独立几何保留一个原型，按类型分组；不是固定的单类均值原型。配置影响特征维度，模型保存特征名称、配置、标准化量、原型和阈值，文件格式为 `geometry-component-recognizer-v2`。

```python
from mor.nn.data import loadGeometryRecognitionDatasetRecords, loadGeometryRecognitionCase
from mor.nn.runtime import geometryComponentRecognizer

records = loadGeometryRecognitionDatasetRecords('dataset/geometry/graph/dataset.json')
recognizer = geometryComponentRecognizer().fitRecords(records)
recognizer.save('reports/geometry/recognizer.json')
prediction = recognizer.predict(loadGeometryRecognitionCase('query/data.npz'), topK=3)
result = prediction.toDict()
```

拟合至少需要两个独立几何。新增类别需要带标签几何并重新建立原型库及标准化量；当前接口未提供保持旧标准化量的增量注册协议。

## 返回值与拒识

输出包含 `predictedType/confidence/unknown/candidates/geometryId/geometry/modelVersion`。候选项为 `componentType/score/distance`，几何诊断包含原始节点数、采样数、中心、尺度和表面连通分量数。

统计路径按最近原型的类别距离阈值和 Top-2 margin 判断 `unknown`。`minimumMargin` 默认 0.02，`rejectionScale` 默认 0.65。confidence 是依据距离与 margin 的启发式接受分数，不是校准后的类别概率。拟合库的回放命中率不能作为未见几何识别精度。

## DGCNN 路径

[geometryDgcnnTorch.py](../../mor/nn/models/geometry/geometryDgcnnTorch.py) 提供位置与法向的动态 kNN/EdgeConv 编码。训练、采样增强、独立几何划分和可选未知样本暴露由脚本控制。checkpoint 保存分类器、原型及可选 `unknownRejection`；没有该配置的模型不能宣称已具备未知类型拒识能力。

```powershell
python scripts/build_geometry_component_recognizer.py dataset/geometry/graph/dataset.json reports/geometry/recognizer.json
python scripts/predict_geometry_component.py reports/geometry/recognizer.json query/data.npz --top-k 3
python scripts/train_geometry_dgcnn.py dataset/geometry/graph/dataset.json reports/geometry/dgcnn.pt --device cpu
python scripts/predict_geometry_component_dgcnn.py reports/geometry/dgcnn.pt query/data.npz --device cpu
```

路径指调用方提供或生成的输入与模型；仓库不附带训练库和 checkpoint。详细参数见脚本 `--help`。

## 评价边界

训练、阈值标定和最终评估须按独立几何分开，未知样本暴露与未知评估也应分离。分别报告已知类型 Top-K/macro F1、已知接受率、未知拒绝率，以及平移、缩放、旋转、重采样和网格细化后的稳定性。

当前保留统计原型与 DGCNN 两条实现，不预设融合模型、通用网格格式导入、ONNX 或增量类别注册已经完成。历史小样本精度和阈值不作为运行默认值；实际模型能力以 checkpoint、当前脚本和独立评估为准。

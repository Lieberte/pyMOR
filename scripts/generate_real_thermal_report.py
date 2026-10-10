"""Build and validate an autonomous intrusive thermal ROM from real STA physics."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from scipy.sparse.linalg import LinearOperator, cg
from threadpoolctl import threadpool_limits

from mor.assembly import loadThermalArchive
from mor.models import coupledThermalModel


def offlineBasis(folders, rank):
    """POD from training trajectories only; retained constant mode is explicit."""
    snapshots = []
    reference = None
    for folder in folders:
        with np.load(folder / 'data.npz') as data:
            nodes, tetra = data['nodesMeters'], data['tetrahedra']
            if reference is None:
                reference = (nodes.copy(), tetra.copy())
            elif not np.array_equal(nodes, reference[0]) or not np.array_equal(tetra, reference[1]):
                raise ValueError('training mesh differs: a registered transfer is required')
        with h5py.File(folder / 'series.h5', 'r') as series:
            snapshots.append(np.asarray(series['nodeTemperatureK'][::5], float).T)
    x = np.hstack(snapshots)
    x -= x.mean(axis=0, keepdims=True)
    values, vectors = linalg.eigh(x.T @ x)
    ids = np.argsort(values)[::-1][:rank-1]
    ids = ids[values[ids] > max(values.max(), 1) * 1e-12]
    modes = x @ (vectors[:, ids] / np.sqrt(values[ids]))
    return np.column_stack((np.ones(len(x))/np.sqrt(len(x)), modes)), reference[0]


def fullStepCheck(assembly, rom, dt=5.):
    """One independent matrix-free implicit step on every real full-order DOF."""
    names = list(assembly.components)
    offsets = np.cumsum([0]+[assembly.components[n].order for n in names])
    slices = {n: slice(offsets[i], offsets[i+1]) for i,n in enumerate(names)}
    initial = np.concatenate([np.full(assembly.components[n].order, assembly.initialTemperatures[n]) for n in names])
    # Solve increments; conduction of constant absolute temperature otherwise
    # hides the source behind a large RHS and gives a misleading CG tolerance.
    def apply(x, transient=True):
        y = np.empty(len(x), dtype=float)
        for n, model in assembly.components.items():
            s = slices[n]
            y[s] = dt * model.K.apply(x[s])
            if transient:
                y[s] += model.M.apply(x[s])
        for interface in assembly.interfaces:
            p, s = slices[interface.primaryComponent], slices[interface.secondaryComponent]
            flux = dt * interface.conductance * (interface.primaryTrace @ x[p] - interface.secondaryTrace @ x[s])
            y[p] += np.asarray(interface.primaryTrace.T @ flux).ravel()
            y[s] -= np.asarray(interface.secondaryTrace.T @ flux).ravel()
        return y
    rhs = dt * np.concatenate([assembly.components[n].Q.toNumpy().ravel() for n in names]) - apply(initial, False)
    diagonal = np.empty(len(initial))
    def operatorDiagonal(op):
        from mor.operators import sumOperator, lowRankOperator
        if isinstance(op, sumOperator):
            return sum(operatorDiagonal(term) for term in op.operators)
        if isinstance(op, lowRankOperator):
            return (op.left * (op.left if op.right is None else op.right)).sum(axis=1)
        return op.toBackendData().diagonal()
    for n, model in assembly.components.items():
        diagonal[slices[n]] = operatorDiagonal(model.M) + dt * operatorDiagonal(model.K)
    for interface in assembly.interfaces:
        diagonal[slices[interface.primaryComponent]] += dt * np.asarray(interface.primaryTrace.power(2).T @ interface.conductance).ravel()
        diagonal[slices[interface.secondaryComponent]] += dt * np.asarray(interface.secondaryTrace.power(2).T @ interface.conductance).ravel()
    iterations = [0]
    def callback(_):
        iterations[0] += 1
    start = time.perf_counter()
    increment, info = cg(LinearOperator((len(initial), len(initial)), matvec=apply, dtype=float), rhs,
                         M=LinearOperator((len(initial), len(initial)), matvec=lambda x:x/diagonal, dtype=float),
                         rtol=1e-9, atol=1e-10, maxiter=8000, callback=callback)
    if info != 0:
        raise RuntimeError(f'full-order CG validation did not converge: {info}')
    reduced = rom.solveTransient(assembly.initialTemperatures, [0., dt])
    prediction = np.concatenate([reduced.temperatures(n)[-1] for n in names])
    error = prediction - (initial + increment)
    return {'timeStepS': dt, 'fullOrder': len(initial), 'iterations': iterations[0],
            'solveSeconds': time.perf_counter()-start, 'relativeEquationResidual': float(np.linalg.norm(apply(increment)-rhs)/np.linalg.norm(rhs)),
            'romVsAssembledFullStepRmseK': float(np.sqrt(np.mean(error**2))),
            'romVsAssembledFullStepMaxErrorK': float(np.abs(error).max()),
            'validationScope': 'one initial 5 s step; not a whole-horizon reduction-error bound'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--whole-dataset', type=Path, required=True)
    parser.add_argument('--holdout', default='tutorial-y')
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--output', type=Path, default=Path('reports/real_thermal'))
    args = parser.parse_args()
    if args.rank < 2:
        parser.error('rank must be at least 2')
    args.output.mkdir(parents=True, exist_ok=True)
    entries = json.loads(args.dataset.read_text(encoding='utf-8-sig'))['cases']
    whole = json.loads(args.whole_dataset.read_text(encoding='utf-8-sig'))['cases']
    selected = [e for e in entries if e['parentCaseId']==args.holdout]
    folders = {e['componentInstance']: (args.dataset.parent/e['path']).resolve() for e in selected}
    source = next(e for e in whole if e['parentCaseId']==args.holdout)['metadata']['sourceSta']
    start = time.perf_counter()
    assembly = loadThermalArchive(source, folders, progress=lambda s:print(s,flush=True))
    args.output.joinpath('assembly_audit.json').write_text(json.dumps(assembly.audit,ensure_ascii=False,indent=2),encoding='utf-8-sig')
    bases = {}
    trainingIds = sorted({e['parentCaseId'] for e in entries if e['parentCaseId']!=args.holdout})
    for name in folders:
        training = [(args.dataset.parent/e['path']).resolve() for e in entries
                    if e['componentInstance']==name and e['parentCaseId'] in trainingIds]
        print(f'offline POD {name}',flush=True)
        bases[name], trainNodes = offlineBasis(training, args.rank)
        if not np.array_equal(trainNodes, assembly.nodes[name]):
            raise ValueError('holdout geometry requires registered basis transfer')
    for name in assembly.components.keys()-folders.keys():
        bases[name] = np.ones((1,1))
    rom = assembly.reduce(bases, rank=args.rank)
    rom.save(args.output/'gearbox_rom.npz')
    buildSeconds = time.perf_counter()-start
    with h5py.File(next(iter(folders.values()))/'series.h5','r') as f:
        times = np.asarray(f['timeS'])
    if not np.allclose(times/5, np.round(times/5)):
        raise ValueError('validation times must match the original 5 s timestep')
    integrationTimes = np.arange(times[0], times[-1]+2.5, 5.)
    start = time.perf_counter()
    result = rom.solveTransient(assembly.initialTemperatures,integrationTimes)
    solveSeconds = time.perf_counter()-start
    ids = np.round((times-integrationTimes[0])/5).astype(int)
    np.savez_compressed(args.output/'prediction_coordinates.npz',timeS=integrationTimes,coordinates=result.coordinates)
    rows, statistics = [], []
    fig, axes = plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for name,folder in folders.items():
        with h5py.File(folder/'series.h5','r') as f:
            if not np.array_equal(f['timeS'][:],times):
                raise ValueError('holdout component time grids differ')
            reference = np.asarray(f['nodeTemperatureK'],float)
        component = rom.components[name]
        prediction = component.reconstruct(result.coordinates[ids,rom.componentSlices[name]])
        error = prediction-reference
        weights = assembly.components[name].M.apply(np.ones(len(reference[0])))
        weights /= weights.sum()
        meanReference, meanPrediction = reference@weights, prediction@weights
        rmse = float(np.sqrt(np.mean((error**2)@weights)))
        reconstruction = component.reconstruct((component.projection@(reference-component.offset).T).T)
        projectionRmse = float(np.sqrt(np.mean(((reconstruction-reference)**2)@weights)))
        statistics.append({'component':name,'order':component.order,'autonomousVsStaRmseK':rmse,
                           'podProjectionRmseK':projectionRmse,'finalMeanPredictionC':float(meanPrediction[-1]-273.15),
                           'finalMeanReferenceC':float(meanReference[-1]-273.15),'maxErrorK':float(np.abs(error).max())})
        for t,p,r in zip(times,meanPrediction,meanReference):
            rows.append({'component':name,'timeS':float(t),'predictedMeanC':float(p-273.15),'referenceMeanC':float(r-273.15)})
        if name in ('g0','b4','housing_1'):
            ax=axes[('g0','b4','housing_1').index(name)]
            ax.plot(times,meanReference-273.15,label='Original STA')
            ax.plot(times,meanPrediction-273.15,'--',label='Autonomous physical ROM')
            ax.set(title=name,xlabel='Time (s)',ylabel='Mean temperature (C)');ax.legend();ax.grid(alpha=.3)
        del reference,prediction,reconstruction,error
    fig.savefig(args.output/'autonomous_prediction.png',dpi=180);plt.close(fig)
    print('checking whole assembled full-order initial step',flush=True)
    fullCheck = fullStepCheck(assembly,rom)
    restored = coupledThermalModel.load(args.output/'gearbox_rom.npz')
    np.testing.assert_allclose(restored.K,rom.K,rtol=1e-12,atol=1e-9)
    summary = {'holdout':args.holdout,'trainingCases':trainingIds,'fullOrder':rom.fullOrder,'reducedOrder':rom.order,
               'buildSeconds':buildSeconds,'romSolveSeconds':solveSeconds,'timeStepS':5.,'endTimeS':float(times[-1]),
               'energyResidualMaxW':float(np.abs(result.diagnostics['residualW']).max()),'components':statistics,
               'oilNodeCount':sum(name.startswith('mass_') for name in rom.components),
               'spatialHtcInputCount':sum(v['coefficientSource']=='femSpatialHtc' for v in assembly.audit['network']),
               'fullOrderCheck':fullCheck,'sourceMismatchIncludes':['remaining surface-average network links',
               'cached spatial HTC mapping and original solver quadrature',
               'nonmatching contact quadrature and uncovered area','original solver spatial/coupling/time discretization'],
               'autonomousVsStaIsNotPureReductionError':True}
    args.output.joinpath('metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8-sig')
    for filename,data in [('temperature_history.csv',rows),('component_metrics.csv',statistics)]:
        with (args.output/filename).open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    report = ['# 真实齿轮箱侵入式热降阶模型','',
        '从原始 STA 读取物性、体热源、四个齿面功率、19 处接触、油液热容、21 个对流和 3 个热阻，组装真实四面体方程并做 POD–Galerkin 投影。',
        '预测只输入初始温度和物理载荷，不输入留出工况的温度或求解后的边界热流。', '',
        f'油液热容节点 {summary["oilNodeCount"]} 个；恢复 FEM 接口 htc3D 空间换热场 {summary["spatialHtcInputCount"]} 条。空间系数取自 controlData.h5 的输入映射，覆盖对流组件的占位常数。', '',
        f'全阶 {rom.fullOrder} 个自由度 → 降阶 {rom.order} 个自由度；积分至 {times[-1]:g} s，步长 5 s，降阶求解 {solveSeconds:.4f} s。',
        f'训练：{", ".join(trainingIds)}；留出：{args.holdout}。能量平衡最大残差 {summary["energyResidualMaxW"]:.3g} W。','',
        '## 验证口径','',
        f'同一组装方程的全阶独立 CG 验证仅覆盖初始 5 s 一步：ROM–FOM RMSE {fullCheck["romVsAssembledFullStepRmseK"]:.6g} K；方程相对残差 {fullCheck["relativeEquationResidual"]:.3g}。不能据此声称全时段降阶误差。',
        '下面的自主预测对比原始 STA 包含物理组装差异与降阶误差；投影误差仅衡量基底表示能力。', '',
        '|部件|自主预测 vs STA RMSE / K|POD 投影 RMSE / K|末时刻预测均温 / °C|STA 均温 / °C|',
        '|---|---:|---:|---:|---:|']
    report += [f'|{s["component"]}|{s["autonomousVsStaRmseK"]:.4f}|{s["podProjectionRmseK"]:.4f}|{s["finalMeanPredictionC"]:.3f}|{s["finalMeanReferenceC"]:.3f}|' for s in statistics]
    report += ['', '## 实现边界','',
        'FEM 模式油侧对流使用输入映射中的每面 HTC，三点高斯积分组装一致边界项，并将反向功率施加给油液节点。其余热网络使用面积平均温度；环境散热为低秩面平均 Robin 项。',
        '空间映射缓存的积分热导与原求解结果中的有效热导仍有差别，因此不声称换热离散完全一致。',
        '非匹配接触按原始 1 mm 半径投影到对侧三角面，双侧半权重积分保持交换守恒；范围外面积不建立人为连接。逐接触覆盖率见 assembly_audit.json。',
        '这是可独立运行的真实物理 ROM 首版，尚未证明与原商业求解器完全等价或可泛化到新几何。', '',
        '输出 gearbox_rom.npz 可独立加载；prediction_coordinates.npz 保存预测坐标，模型基底可重构任意部件的三维节点温度。']
    args.output.joinpath('report.md').write_text('\n'.join(report)+'\n',encoding='utf-8-sig')
    print(json.dumps({k:v for k,v in summary.items() if k!='components'},ensure_ascii=True),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):
        main()

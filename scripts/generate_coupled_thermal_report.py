from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.sparse import diags

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mor.assembly import thermalCouplingInterface
from mor.models import thermalModel
from mor.operators import matrixOperator, thermalOperatorSet
from mor.reductors import thermalCouplingReductor


def buildThermalChain(nodeCount, capacity, conductivity, sink=0.):
    K = diags([-conductivity * np.ones(nodeCount-1), np.full(nodeCount, 2*conductivity),
               -conductivity * np.ones(nodeCount-1)], [-1, 0, 1], format='lil')
    K[0, 0] -= conductivity
    K[-1, -1] -= conductivity
    K[-1, -1] += sink
    Q = np.zeros(nodeCount)
    Q[-1] = sink * 300.
    return thermalModel(thermalOperatorSet(matrixOperator(K.tocsr(), 'scipy'),
        matrixOperator(diags(np.full(nodeCount, capacity)), 'scipy'), matrixOperator(Q, 'scipy'), np.empty(0)),
        referenceTemperature=300.)


def runCharacterization(nodeCount=24, ranks=(2, 4, 8)):
    if nodeCount < 2:
        raise ValueError('nodeCount must be at least two')
    reductor = thermalCouplingReductor()
    originals = {name: buildThermalChain(nodeCount, capacity, conductivity, sink)
                 for name, capacity, conductivity, sink in
                 [('gear', 2., 10., 0.), ('bearing', 1., 8., 0.), ('housing', 3., 12., 3.)]}
    originals['oil'] = buildThermalChain(1, 50., 0., 1.)
    inputOperators = {}
    for name, model in originals.items():
        values = np.zeros((model.order, 1))
        values[0, 0] = 1.
        inputOperators[name] = matrixOperator(values, 'scipy')
    interfaces = [thermalCouplingInterface.fromNodePairs('gear_bearing', 'gear', 'bearing', nodeCount, nodeCount,
                    [nodeCount-1], [0], coefficient=20., areas=1.),
                  thermalCouplingInterface.fromNodePairs('bearing_housing', 'bearing', 'housing', nodeCount, nodeCount,
                    [nodeCount-1], [0], coefficient=15., areas=1.),
                  thermalCouplingInterface('housing_oil', 'housing', 'oil', np.ones((1, nodeCount))/nodeCount, [[1.]], [5.])]
    fullComponents = [reductor.reduceComponent(name, model, basis=np.eye(model.order), inputOperator=inputOperators[name])
                      for name, model in originals.items()]
    full = reductor.assemble(fullComponents, interfaces)
    times = np.arange(101, dtype=float) * .5
    initial = {name: 300. for name in originals}
    training = {name: [] for name in originals}
    for heated in ['gear', 'bearing', 'housing']:
        history = full.solveTransient(initial, times, lambda t, heated=heated: {heated: [30. if t <= 15. else 0.]})
        for name in originals:
            training[name].append(history.temperatures(name).T)
    inputs = lambda t: {'gear': [18. + 8. * np.sin(t/8.)], 'bearing': [12. if t > 10. else 0.]}
    start = perf_counter()
    reference = full.solveTransient(initial, times, inputs)
    referenceSeconds = perf_counter() - start
    expected = np.hstack([reference.temperatures(name) for name in originals])
    rows, curves = [], {'timeS': times.tolist(), 'referenceGearMaxK': reference.temperatures('gear').max(axis=1).tolist()}
    for rank in ranks:
        start = perf_counter()
        components = [reductor.reduceComponent(name, model, snapshots=np.hstack(training[name]),
                            rank=min(rank, model.order), inputOperator=inputOperators[name])
                      for name, model in originals.items()]
        reduced = reductor.assemble(components, interfaces)
        constructionSeconds = perf_counter() - start
        start = perf_counter()
        prediction = reduced.solveTransient(initial, times, inputs)
        onlineSeconds = perf_counter() - start
        actual = np.hstack([prediction.temperatures(name) for name in originals])
        errors = actual - expected
        fluxError = max(np.max(np.abs(prediction.interfacePowers()[name] - reference.interfacePowers()[name])) for name in full.interfaces)
        rows.append({'componentRankLimit': rank, 'fullOrder': full.order, 'reducedOrder': reduced.order,
                     'constructionSeconds': constructionSeconds, 'fullOnlineSeconds': referenceSeconds,
                     'reducedOnlineSeconds': onlineSeconds, 'rmseKelvin': float(np.sqrt(np.mean(errors**2))),
                     'maxAbsErrorKelvin': float(np.max(np.abs(errors))),
                     'relativeTemperatureRiseError': float(np.linalg.norm(errors)/np.linalg.norm(expected-300.)),
                     'maxInterfacePowerErrorW': float(fluxError),
                     'maxEnergyBalanceResidualW': float(np.max(np.abs(prediction.diagnostics['residualW']))),
                     'factorizationCount': reduced.lastSolveInfo['factorizationCount']})
        curves[f'rank{rank}GearMaxK'] = prediction.temperatures('gear').max(axis=1).tolist()
    return {'scenario': 'gear-bearing-housing-oil heat diffusion chain',
            'validation': 'held-out combination of sinusoidal gear heating and delayed bearing heating',
            'rows': rows, 'curves': curves}


def main():
    parser = argparse.ArgumentParser(description='Characterize conservative coupled thermal component ROMs.')
    parser.add_argument('--output-directory', type=Path, default=Path('reports/coupled_thermal'))
    parser.add_argument('--nodes-per-solid', type=int, default=24)
    args = parser.parse_args()
    result = runCharacterization(args.nodes_per_solid, ranks=tuple(r for r in (2, 4, 8) if r <= args.nodes_per_solid))
    output = args.output_directory
    output.mkdir(parents=True, exist_ok=True)
    (output/'coupled_thermal_results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    with (output/'coupled_thermal_results.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=result['rows'][0])
        writer.writeheader()
        writer.writerows(result['rows'])
    lines = ['# 耦合热降阶验证', '', '齿轮—轴承—箱体—油液的合成热扩散网络；三个独立热源脉冲用于建基底，另一组热源组合用于验证。',
             '该报告验证方法，不代表真实齿轮箱预测精度。耗时包含降阶坐标求解和诊断，不包含全场重构；小问题不保证加速。', '',
             '| 部件秩上限 | 整机维数 | RMSE/K | 最大误差/K | 能量残差/W |', '| --- | --- | --- | --- | --- |']
    for row in result['rows']:
        lines.append(f"| {row['componentRankLimit']} | {row['reducedOrder']} | {row['rmseKelvin']:.4g} | {row['maxAbsErrorKelvin']:.4g} | {row['maxEnergyBalanceResidualW']:.4g} |")
    lines += ['', '![齿轮最高温度](coupled_thermal.svg)', '']
    (output/'COUPLED_THERMAL_REPORT.md').write_text('\n'.join(lines), encoding='utf-8-sig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, ax = plt.subplots()
    curves = result['curves']
    for name, values in curves.items():
        if name != 'timeS':
            ax.plot(curves['timeS'], values, label=name)
    ax.set(xlabel='Time (s)', ylabel='Gear maximum temperature (K)')
    ax.legend()
    figure.savefig(output/'coupled_thermal.svg')
    plt.close(figure)
    texRows = [f"{row['componentRankLimit']} & {row['reducedOrder']} & {row['rmseKelvin']:.4g} & {row['maxEnergyBalanceResidualW']:.4g} \\\\" for row in result['rows']]
    tex = '\n'.join([r'\documentclass{article}', r'\begin{document}', r'\section*{Coupled thermal component ROM}',
        'Synthetic gear-bearing-housing-oil network; held-out forcing validation.', r'\begin{tabular}{rrrr}',
        r'Rank limit & ROM order & RMSE (K) & Energy residual (W) \\', *texRows, r'\end{tabular}', r'\end{document}'])
    (output/'coupled_thermal_report.tex').write_text(tex, encoding='utf-8')
    print(json.dumps(result['rows'], indent=2))


if __name__ == '__main__':
    main()

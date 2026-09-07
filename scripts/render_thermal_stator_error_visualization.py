from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('analysis', type=Path)
    parser.add_argument('output', type=Path)
    arguments = parser.parse_args()
    data = json.loads(arguments.analysis.read_text(encoding='utf-8'))
    payload = json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace(
        '<', '\\u003c'
    )
    fragment = r'''<section id="stator-error-clouds-v1">
  <style>
    #stator-error-clouds-v1 { position: relative; width: 100%; color: var(--foreground); }
    #stator-error-clouds-v1 h1 { margin: 0 0 8px; font-weight: 500; letter-spacing: 0; }
    #stator-error-clouds-v1 h2 { margin: 22px 0 4px; font-weight: 500; letter-spacing: 0; }
    #stator-error-clouds-v1 .sec-line { color: var(--muted-foreground); margin-bottom: 8px; }
    #stator-error-clouds-v1 .facet-grid { display: grid; grid-template-columns: repeat(3, minmax(220px, 1fr)); gap: 12px; }
    #stator-error-clouds-v1 figure { margin: 0; min-width: 0; }
    #stator-error-clouds-v1 figcaption { color: var(--foreground); font-weight: 500; margin: 0 0 2px; }
    #stator-error-clouds-v1 .plot { width: 100%; min-height: 230px; }
    #stator-error-clouds-v1 .plot svg { display: block; width: 100%; height: auto; }
    #stator-error-clouds-v1 .legend { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; color: var(--muted-foreground); margin: 8px 0 4px; }
    #stator-error-clouds-v1 .legend-item { display: inline-flex; align-items: center; gap: 5px; }
    #stator-error-clouds-v1 .swatch { width: 30px; height: 8px; display: inline-block; }
    #stator-error-clouds-v1 .swatch-temp { background: linear-gradient(90deg, var(--viz-series-1), var(--yellow)); }
    #stator-error-clouds-v1 .swatch-error { background: linear-gradient(90deg, var(--blue), var(--muted), var(--red)); }
    #stator-error-clouds-v1 .range-chart { width: 100%; min-height: 230px; margin-top: 4px; }
    #stator-error-clouds-v1 .tooltip { position: absolute; pointer-events: none; z-index: 4; display: none; padding: 7px 9px; background: var(--popover); color: var(--popover-foreground); border: 1px solid var(--border); border-radius: 6px; font-size: 12px; }
    #stator-error-clouds-v1 .axis text, #stator-error-clouds-v1 .axis-title, #stator-error-clouds-v1 .direct-label { fill: var(--foreground); font-size: 12px; }
    #stator-error-clouds-v1 .axis path, #stator-error-clouds-v1 .axis line { stroke: var(--border); }
    @media (max-width: 760px) {
      #stator-error-clouds-v1 .facet-grid { grid-template-columns: 1fr; }
    }
  </style>
  <h1>Stator 绝对温度与误差云图</h1>
  <div class="legend" aria-label="颜色图例">
    <span class="legend-item"><span class="swatch swatch-temp"></span>温度：低 → 高</span>
    <span class="legend-item"><span class="swatch swatch-error"></span>误差：低估 → 高估</span>
  </div>
  <div data-cases></div>
  <h2>B 与 H 的整体温度分布</h2>
  <div class="sec-line">线段表示最小值到最大值，圆点表示体积加权均温。</div>
  <div class="range-chart" data-range-chart></div>
  <div class="tooltip" role="tooltip"></div>
  <script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
  <script>
  (() => {
    const root = document.getElementById('stator-error-clouds-v1');
    const data = __DATA__;
    const css = getComputedStyle(root);
    const colors = {
      foreground: css.getPropertyValue('--foreground').trim(),
      border: css.getPropertyValue('--border').trim(),
      lowTemp: css.getPropertyValue('--viz-series-1').trim(),
      highTemp: css.getPropertyValue('--yellow').trim(),
      lowError: css.getPropertyValue('--blue').trim(),
      zeroError: css.getPropertyValue('--muted').trim(),
      highError: css.getPropertyValue('--red').trim(),
      trueSeries: css.getPropertyValue('--viz-series-1').trim(),
      predictionSeries: css.getPropertyValue('--viz-series-2').trim()
    };
    const fields = [
      { key: 'trueK', label: '真实温度', unit: 'K' },
      { key: 'predictionK', label: '预测温度', unit: 'K' },
      { key: 'errorK', label: '有符号误差', unit: 'K' }
    ];
    const allPoints = data.cases.flatMap(d => d.points);
    const tempExtent = d3.extent(allPoints.flatMap(d => [d.trueK, d.predictionK]));
    const errorLimit = d3.max(allPoints, d => Math.abs(d.errorK));
    const temperatureColor = d3.scaleSequential(
      d3.interpolateRgb(colors.lowTemp, colors.highTemp)
    ).domain(tempExtent);
    const errorColor = d3.scaleDiverging()
      .domain([-errorLimit, 0, errorLimit])
      .interpolator(d3.piecewise(d3.interpolateRgb, [colors.lowError, colors.zeroError, colors.highError]));
    const casesHost = root.querySelector('[data-cases]');
    data.cases.forEach(caseData => {
      const section = document.createElement('section');
      const m = caseData.metrics;
      section.innerHTML = `<h2>${caseData.label}</h2><div class="sec-line">RMSE ${m.rmseK.toFixed(3)} K · 均温偏差 ${m.meanBiasK.toFixed(3)} K · 场形状 RMSE ${m.shapeRmseK.toFixed(3)} K · 真实温差 ${m.trueSpanK.toFixed(3)} K</div><div class="facet-grid"></div>`;
      const grid = section.querySelector('.facet-grid');
      fields.forEach(field => {
        const figure = document.createElement('figure');
        figure.innerHTML = `<figcaption>${field.label}</figcaption><div class="plot" data-case="${caseData.label}" data-field="${field.key}"></div>`;
        grid.appendChild(figure);
      });
      casesHost.appendChild(section);
    });

    const tooltip = root.querySelector('.tooltip');
    function paddedExtent(values) {
      const [minimum, maximum] = d3.extent(values);
      const pad = Math.max((maximum - minimum) * 0.07, 0.01);
      return [minimum - pad, maximum + pad];
    }
    function drawCloud(container, caseData, field) {
      container.replaceChildren();
      const width = Math.max(280, container.clientWidth);
      const height = Math.max(230, Math.min(320, width * 0.78));
      const margin = { top: 8, right: 16, bottom: 42, left: 54 };
      const plotWidth = width - margin.left - margin.right;
      const plotHeight = height - margin.top - margin.bottom;
      const svg = d3.select(container).append('svg')
        .attr('viewBox', `0 0 ${width} ${height}`)
        .attr('role', 'img')
        .attr('aria-label', `${caseData.label} ${field.label}点云`);
      svg.append('title').text(`${caseData.label} ${field.label}`);
      const x = d3.scaleLinear().domain(paddedExtent(caseData.points.map(d => d.p1))).range([margin.left, margin.left + plotWidth]);
      const y = d3.scaleLinear().domain(paddedExtent(caseData.points.map(d => d.p2))).range([margin.top + plotHeight, margin.top]);
      svg.append('rect').attr('data-chart-frame', '').attr('x', margin.left).attr('y', margin.top).attr('width', plotWidth).attr('height', plotHeight).attr('fill', 'none').attr('stroke', colors.border);
      svg.append('g').attr('class', 'axis').attr('transform', `translate(0,${margin.top + plotHeight})`).call(d3.axisBottom(x).ticks(width < 360 ? 3 : 4));
      svg.append('g').attr('class', 'axis').attr('transform', `translate(${margin.left},0)`).call(d3.axisLeft(y).ticks(4));
      svg.append('text').attr('class', 'axis-title').attr('data-axis', 'x').attr('x', margin.left + plotWidth / 2).attr('y', height - 5).attr('text-anchor', 'middle').text('主坐标 1');
      svg.append('text').attr('class', 'axis-title').attr('data-axis', 'y').attr('transform', `translate(14,${margin.top + plotHeight / 2}) rotate(-90)`).attr('text-anchor', 'middle').text('主坐标 2');
      const color = field.key === 'errorK' ? errorColor : temperatureColor;
      svg.append('g').selectAll('circle').data(caseData.points).join('circle')
        .attr('cx', d => x(d.p1)).attr('cy', d => y(d.p2)).attr('r', 3.4)
        .attr('fill', d => color(d[field.key])).attr('opacity', 0.9);
      const delaunay = d3.Delaunay.from(caseData.points, d => x(d.p1), d => y(d.p2));
      const marker = svg.append('circle').attr('r', 6).attr('fill', 'none').attr('stroke', colors.foreground).attr('stroke-width', 1.5).style('display', 'none');
      svg.append('rect').attr('x', margin.left).attr('y', margin.top).attr('width', plotWidth).attr('height', plotHeight).attr('fill', 'transparent')
        .on('pointermove', event => {
          const [px, py] = d3.pointer(event, svg.node());
          const point = caseData.points[delaunay.find(px, py)];
          marker.attr('cx', x(point.p1)).attr('cy', y(point.p2)).style('display', null);
          tooltip.style.display = 'block';
          tooltip.innerHTML = `<strong>${caseData.label} ${field.label}</strong><br>${point[field.key].toFixed(3)} ${field.unit}<br>真实 ${point.trueK.toFixed(3)} K · 预测 ${point.predictionK.toFixed(3)} K`;
          const bounds = root.getBoundingClientRect();
          tooltip.style.left = `${event.clientX - bounds.left + 12}px`;
          tooltip.style.top = `${event.clientY - bounds.top + 12}px`;
        })
        .on('pointerleave', () => { marker.style('display', 'none'); tooltip.style.display = 'none'; });
    }

    function drawRanges() {
      const container = root.querySelector('[data-range-chart]');
      container.replaceChildren();
      const values = data.cases.filter(d => d.label === 'B' || d.label === 'H').flatMap(d => {
        const m = d.metrics;
        return [
          { label: `${d.label} 真实`, minimum: m.trueMinimumK, maximum: m.trueMaximumK, mean: m.weightedTrueMeanK, kind: 'true' },
          { label: `${d.label} 预测`, minimum: m.predictionMinimumK, maximum: m.predictionMaximumK, mean: m.weightedPredictionMeanK, kind: 'prediction' }
        ];
      });
      const width = Math.max(320, container.clientWidth);
      const height = 220;
      const margin = { top: 18, right: 82, bottom: 42, left: 72 };
      const svg = d3.select(container).append('svg').attr('viewBox', `0 0 ${width} ${height}`).attr('role', 'img').attr('aria-label', 'B 与 H 温度范围和均温比较');
      svg.append('title').text('B 与 H 温度范围和均温比较');
      const x = d3.scaleLinear().domain(paddedExtent(values.flatMap(d => [d.minimum, d.maximum, d.mean]))).range([margin.left, width - margin.right]);
      const y = d3.scaleBand().domain(values.map(d => d.label)).range([margin.top, height - margin.bottom]).padding(0.45);
      svg.append('rect').attr('data-chart-frame', '').attr('x', margin.left).attr('y', margin.top).attr('width', width - margin.left - margin.right).attr('height', height - margin.top - margin.bottom).attr('fill', 'none').attr('stroke', colors.border);
      svg.append('g').attr('class', 'axis').attr('transform', `translate(0,${height - margin.bottom})`).call(d3.axisBottom(x).ticks(width < 500 ? 4 : 6));
      svg.append('g').attr('class', 'axis').attr('transform', `translate(${margin.left},0)`).call(d3.axisLeft(y).tickSize(0));
      svg.append('text').attr('class', 'axis-title').attr('data-axis', 'x').attr('x', margin.left + (width - margin.left - margin.right) / 2).attr('y', height - 5).attr('text-anchor', 'middle').text('温度 / K');
      const rows = svg.append('g').selectAll('g').data(values).join('g');
      rows.append('line').attr('x1', d => x(d.minimum)).attr('x2', d => x(d.maximum)).attr('y1', d => y(d.label) + y.bandwidth() / 2).attr('y2', d => y(d.label) + y.bandwidth() / 2).attr('stroke', d => d.kind === 'true' ? colors.trueSeries : colors.predictionSeries).attr('stroke-width', 5);
      rows.append('circle').attr('cx', d => x(d.mean)).attr('cy', d => y(d.label) + y.bandwidth() / 2).attr('r', 5).attr('fill', colors.foreground);
      rows.append('text').attr('class', 'direct-label').attr('x', d => x(d.maximum) + 7).attr('y', d => y(d.label) + y.bandwidth() / 2 + 4).text(d => `${(d.maximum - d.minimum).toFixed(3)} K`);
    }

    function drawAll() {
      data.cases.forEach(caseData => fields.forEach(field => {
        drawCloud(root.querySelector(`[data-case="${caseData.label}"][data-field="${field.key}"]`), caseData, field);
      }));
      drawRanges();
    }
    drawAll();
    new ResizeObserver(drawAll).observe(root);
  })();
  </script>
</section>'''.replace('__DATA__', payload)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(fragment, encoding='utf-8')


if __name__ == '__main__':
    main()

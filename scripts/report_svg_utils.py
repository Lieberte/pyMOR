from __future__ import annotations

from pathlib import Path


def svg_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _format_tick(value: float, log10: bool = False) -> str:
    if log10:
        return f"1e{value:.0f}"
    if abs(value) >= 1000 or (abs(value) > 0 and abs(value) < 1e-2):
        return f"{value:.1e}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.3f}"


def _append_axis_ticks(parts: list[str], left: int, top: int, bottom_y: int, min_y: float, max_y: float, log10: bool) -> None:
    tick_count = 5
    span = max(max_y - min_y, 1e-12)
    for idx in range(tick_count):
        fraction = idx / (tick_count - 1)
        value = min_y + (1.0 - fraction) * span
        y = top + fraction * (bottom_y - top)
        parts.append(f'<line x1="{left - 6}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#9aa5b1" stroke-width="1"/>')
        parts.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Segoe UI, sans-serif" font-size="10" fill="#52616b">{svg_escape(_format_tick(value, log10=log10))}</text>'
        )


def write_line_chart_svg(
    series: list[dict],
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
    log10: bool = False,
) -> None:
    max_points = max((len(item["points"]) for item in series), default=0)
    width, height = max(1080, 90 + 40 + max_points * 34), 420
    left, right, top, bottom = 90, 40, 60, 90
    chart_width = width - left - right
    chart_height = height - top - bottom
    palette = ["#256d7b", "#7b5d73", "#4f6f52", "#6f5b3e", "#bc6c25", "#4c6ef5"]

    all_x = [point["x"] for item in series for point in item["points"]]
    all_y_raw = [point["y"] for item in series for point in item["points"]]
    all_y = [max(value, 1e-300) for value in all_y_raw]
    if log10:
        import math
        all_y = [math.log10(value) for value in all_y]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    x_span = max(max_x - min_x, 1e-12)
    y_span = max(max_y - min_y, 1e-12)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbf8"/>',
        f'<text x="{left}" y="32" font-family="Segoe UI, sans-serif" font-size="22" fill="#1f2933">{svg_escape(title)}</text>',
        f'<text x="{left}" y="{height - 20}" font-family="Segoe UI, sans-serif" font-size="12" fill="#52616b">{svg_escape(x_label)}</text>',
        f'<text x="16" y="{top}" font-family="Segoe UI, sans-serif" font-size="12" fill="#52616b">{svg_escape(y_label)}</text>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#9aa5b1" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#9aa5b1" stroke-width="1"/>',
    ]
    _append_axis_ticks(parts, left, top, height - bottom, min_y, max_y, log10)

    for idx, item in enumerate(series):
        color = palette[idx % len(palette)]
        points = []
        for point in item["points"]:
            x = left + ((point["x"] - min_x) / x_span) * chart_width
            y_value = max(point["y"], 1e-300)
            if log10:
                import math
                y_value = math.log10(y_value)
            y = top + (1.0 - ((y_value - min_y) / y_span)) * chart_height
            points.append((x, y, point))
        polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in points)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{polyline}"/>')
        label_step = max(1, len(points) // 12)
        for point_index, (x, y, point) in enumerate(points):
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}"/>')
            if point_index == 0 or point_index == len(points) - 1 or point_index % label_step == 0:
                parts.append(
                    f'<text x="{x:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-family="Segoe UI, sans-serif" font-size="9" fill="#1f2933">{svg_escape(point["label"])}</text>'
                )
        legend_x = width - right - 220
        legend_y = top + idx * 18
        parts.append(f'<rect x="{legend_x}" y="{legend_y - 10}" width="12" height="12" fill="{color}"/>')
        parts.append(
            f'<text x="{legend_x + 18}" y="{legend_y}" font-family="Segoe UI, sans-serif" font-size="11" fill="#1f2933">{svg_escape(item["name"])}</text>'
        )

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_success_matrix_svg(
    variants: list[str],
    cases: list[str],
    success_map: dict[tuple[str, str], bool],
    path: Path,
    title: str,
) -> None:
    cell_w, cell_h = 58, 26
    left, top = 180, 70
    width = left + cell_w * len(cases) + 40
    height = top + cell_h * len(variants) + 60
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbf8"/>',
        f'<text x="{left}" y="32" font-family="Segoe UI, sans-serif" font-size="22" fill="#1f2933">{svg_escape(title)}</text>',
    ]
    for col, case in enumerate(cases):
        x = left + col * cell_w
        parts.append(
            f'<text x="{x + cell_w / 2:.1f}" y="{top - 16}" text-anchor="middle" font-family="Segoe UI, sans-serif" font-size="9" fill="#52616b" transform="rotate(-55 {x + cell_w / 2:.1f} {top - 16})">{svg_escape(case)}</text>'
        )
    for row, variant in enumerate(variants):
        y = top + row * cell_h
        parts.append(
            f'<text x="{left - 10}" y="{y + 17}" text-anchor="end" font-family="Segoe UI, sans-serif" font-size="10" fill="#52616b">{svg_escape(variant)}</text>'
        )
        for col, case in enumerate(cases):
            x = left + col * cell_w
            success = success_map.get((variant, case), False)
            fill = "#4f6f52" if success else "#b03a2e"
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w - 6}" height="{cell_h - 6}" fill="{fill}" rx="3"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_bar_chart_svg(
    rows: list[dict],
    path: Path,
    title: str,
    value_key: str,
    subtitle: str,
    label_key: str,
    log10: bool = False,
    color: str = "#4f6f52",
    show_value_labels: bool | None = None,
) -> None:
    width = max(1080, 90 + 40 + max(len(rows), 1) * 36)
    height = 460
    left, right, top, bottom = 90, 40, 58, 170
    chart_width = width - left - right
    chart_height = height - top - bottom
    values = [max(float(row[value_key]), 1e-300) for row in rows]
    display_values = values[:]
    if log10:
        import math
        values = [math.log10(value) for value in values]
    min_value = min(values)
    max_value = max(values)
    span = max(max_value - min_value, 1e-12)
    bar_gap = 8
    bar_width = (chart_width - bar_gap * (len(values) - 1)) / max(len(values), 1)
    baseline = top + chart_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbf8"/>',
        f'<text x="{left}" y="32" font-family="Segoe UI, sans-serif" font-size="22" fill="#1f2933">{svg_escape(title)}</text>',
        f'<text x="{left}" y="52" font-family="Segoe UI, sans-serif" font-size="12" fill="#52616b">{svg_escape(subtitle)}</text>',
        f'<line x1="{left}" y1="{baseline}" x2="{width - right}" y2="{baseline}" stroke="#9aa5b1" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{baseline}" stroke="#9aa5b1" stroke-width="1"/>',
    ]
    _append_axis_ticks(parts, left, top, baseline, min_value, max_value, log10)
    if show_value_labels is None:
        show_value_labels = len(rows) <= 16

    for index, row in enumerate(rows):
        value = values[index]
        normalized = (value - min_value) / span
        bar_height = 18 + normalized * (chart_height - 18)
        x = left + index * (bar_width + bar_gap)
        y = baseline - bar_height
        display_value = f"{display_values[index]:.3e}" if log10 else f"{display_values[index]:.3f}"
        parts.extend(
            [
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{color}"/>',
                f'<title>{svg_escape(row.get("case", row[label_key]))}: {display_value}</title>',
                f'<text x="{x + bar_width / 2:.1f}" y="{baseline + 22}" text-anchor="end" font-family="Segoe UI, sans-serif" font-size="9" fill="#52616b" transform="rotate(-60 {x + bar_width / 2:.1f} {baseline + 22})">{svg_escape(row[label_key])}</text>',
            ]
        )
        if show_value_labels:
            parts.append(
                f'<text x="{x + bar_width / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-family="Segoe UI, sans-serif" font-size="10" fill="#1f2933">{display_value}</text>'
            )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")

"""Create the single-bid README SVG from checked-in result CSV files.

The script deliberately uses only Python's standard library so the figure can
be regenerated before NumPy is installed.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
OUTPUT = ROOT / "docs" / "key-results.svg"


def read_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def make_svg() -> str:
    fine = read_csv("illustrative_fine_grid.csv")
    quote_points = [
        (float(row["bid"]), float(row["p_win_pct"]))
        for row in fine
        if 163.80 <= float(row["bid"]) <= 165.30
    ]
    people = read_csv("participants_sensitivity.csv")
    people_points = [(int(row["N"]), float(row["p_win_pct"])) for row in people]

    width, height = 1120, 620
    left_x, right_x, top_y, panel_w, panel_h = 74, 630, 112, 455, 390
    chart_top = top_y + 34
    chart_bottom = top_y + 330

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">示例场景下的单次报价模拟结果</title>',
        '<desc id="desc">左图显示示例场景中的报价与单次中标率，165元处因整数拥挤明显下跌；右图显示参与人数增加时最佳单次报价的模型中标率下降。</desc>',
        '<style>',
        'text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans CJK SC","Microsoft YaHei",sans-serif;fill:#24292f}',
        '.title{font-size:26px;font-weight:600}.subtitle{font-size:17px;font-weight:600}.axis{font-size:13px;fill:#57606a}.value{font-size:13px;font-weight:600}.grid{stroke:#d8dee4;stroke-width:1}.frame{fill:#ffffff;stroke:#d0d7de;stroke-width:1}.line{fill:none;stroke:#1f6feb;stroke-width:3}.point{fill:#ffffff;stroke:#1f6feb;stroke-width:2}.focus{fill:#cf222e;stroke:#ffffff;stroke-width:1}',
        '</style>',
        '<rect width="1120" height="620" fill="#ffffff"/>',
        '<text class="title" x="560" y="46" text-anchor="middle">示例场景下的单次报价模拟结果</text>',
        '<text class="axis" x="560" y="72" text-anchor="middle">组件比例未经受众数据校准；数值不是现实活动的保证概率</text>',
        '<text class="subtitle" x="74" y="98">报价与单次中标率（N=25,000）</text>',
        '<text class="subtitle" x="630" y="98">参与人数与最佳单次报价中标率</text>',
        f'<rect class="frame" x="{left_x}" y="{top_y}" width="{panel_w}" height="{panel_h}"/>',
        f'<rect class="frame" x="{right_x}" y="{top_y}" width="{panel_w}" height="{panel_h}"/>',
    ]

    def draw_line_chart(
        points: list[tuple[float, float]],
        panel_x: float,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        x_ticks: list[tuple[float, str]],
        y_ticks: list[float],
        x_title: str,
    ) -> tuple:
        chart_left = panel_x + 64
        chart_right = panel_x + panel_w - 24

        def sx(value: float) -> float:
            return chart_left + (value - x_min) / (x_max - x_min) * (chart_right - chart_left)

        def sy(value: float) -> float:
            return chart_bottom - (value - y_min) / (y_max - y_min) * (chart_bottom - chart_top)

        for tick in y_ticks:
            y = sy(tick)
            parts.append(f'<line class="grid" x1="{chart_left}" y1="{y:.1f}" x2="{chart_right}" y2="{y:.1f}"/>')
            parts.append(f'<text class="axis" x="{chart_left - 10}" y="{y + 4:.1f}" text-anchor="end">{tick:.1f}%</text>')
        for value, label in x_ticks:
            x = sx(value)
            parts.append(f'<text class="axis" x="{x:.1f}" y="{chart_bottom + 25}" text-anchor="middle">{label}</text>')
        path = " ".join(
            ("M" if index == 0 else "L") + f" {sx(x):.1f} {sy(y):.1f}"
            for index, (x, y) in enumerate(points)
        )
        parts.append(f'<path class="line" d="{path}"/>')
        parts.append(f'<text class="axis" x="{(chart_left + chart_right) / 2:.1f}" y="{top_y + 383}" text-anchor="middle">{x_title}</text>')
        return sx, sy

    quote_sx, quote_sy = draw_line_chart(
        quote_points,
        left_x,
        163.8,
        165.3,
        0.0,
        0.42,
        [(164.0, "164.0"), (164.5, "164.5"), (165.0, "165.0"), (165.3, "165.3")],
        [0.0, 0.1, 0.2, 0.3, 0.4],
        "单次报价（元）",
    )
    for bid, value in quote_points:
        if abs(bid - 164.74) < 0.001:
            parts.append(f'<circle class="point" cx="{quote_sx(bid):.1f}" cy="{quote_sy(value):.1f}" r="5"/>')
            parts.append(f'<text class="value" x="{quote_sx(bid) - 7:.1f}" y="{quote_sy(value) - 12:.1f}" text-anchor="end">164.74</text>')
        elif abs(bid - 165.00) < 0.001:
            parts.append(f'<circle class="focus" cx="{quote_sx(bid):.1f}" cy="{quote_sy(value):.1f}" r="5"/>')
            parts.append(f'<text class="value" x="{quote_sx(bid) + 8:.1f}" y="{quote_sy(value) + 18:.1f}">165.00</text>')

    people_sx, people_sy = draw_line_chart(
        people_points,
        right_x,
        10_000,
        50_000,
        0.15,
        0.80,
        [(10_000, "10k"), (20_000, "20k"), (30_000, "30k"), (40_000, "40k"), (50_000, "50k")],
        [0.2, 0.4, 0.6, 0.8],
        "总参与人数 N",
    )
    for n_total, value in people_points:
        parts.append(f'<circle class="point" cx="{people_sx(n_total):.1f}" cy="{people_sy(value):.1f}" r="5"/>')
    first_n, first_value = people_points[0]
    last_n, last_value = people_points[-1]
    parts.append(f'<text class="value" x="{people_sx(first_n) + 8:.1f}" y="{people_sy(first_value) - 10:.1f}">{first_value:.2f}%</text>')
    parts.append(f'<text class="value" x="{people_sx(last_n) - 8:.1f}" y="{people_sy(last_value) - 10:.1f}" text-anchor="end">{last_value:.2f}%</text>')
    parts.append('<text class="axis" x="560" y="565" text-anchor="middle">左图显示非整数近优平台与 165.00 的拥挤断点；右图显示竞争人数增加的影响</text>')
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


if __name__ == "__main__":
    OUTPUT.write_text(make_svg(), encoding="utf-8")
    print(OUTPUT)

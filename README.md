# 最近均值报价博弈：单次报价蒙特卡洛案例研究

> A reproducible Monte Carlo case study of a single decision in a large-population nearest-mean bidding game.

本项目把一次真实活动中的**单次报价决策**整理为可复核的研究案例：参与者需要预测所有报价的均值，同时考虑整数价和共同信息源可能造成的同价拥挤。仓库保存了公开证据审计、情景模型、固定随机种子的模拟程序、结果表和敏感性分析，而不只公布一个孤立答案。

> [!IMPORTANT]
> 模拟结果是截至 2026-09-01 的模型快照，公开证据复核至 2026-09-02；它不是对真实最终报价分布的测量，也不保证中标。活动参与应遵守平台规则。

## 证据审计后的核心结论

**公开资料不足以识别一个由数据支持的唯一最优报价。** 没有公开、可复核的 LKs 粉丝年龄/兴趣交叉数据，也没有活动参与者的报价分布；尤其不能从现有资料推出“35% 是球友”或“20% 使用 AI”。

在保留原参数、仅用于复现实验的**示例场景**下，单次报价的条件性结果是：

```text
164.74 元
```

- 示例场景的群体报价均值约为 **164.74 元**。
- 局部平滑后的条件性最优点为 **164.74 元**；`164.65–164.85` 是该场景的近优平台。
- 只改变共同信号参数时，较宽的模型近优区间约为 `164.57–164.89`。
- `165.00` 虽然接近示例均值，但价格堆积使其模型中标率约为 `0.0017%`，而附近非整数报价约为 `0.36%`。
- 相邻非整数分币的差异小于保守的蒙特卡洛误差，不能把某一分钱包装成确定的唯一最优。

当潜在策略占比或报价中心在压力测试范围内变化时，最优点可移动到约 `158.29–171.18` 元。这个宽区间说明：当前最重要的结论是**参数尚未被真实报价数据识别**，而不是继续优化最后几分钱。

![核心模拟结果](docs/key-results.svg)

## 这个仓库为什么值得保留

它保存了结论背后可以被复核、质疑和更新的部分：

1. **证据分级**：LKs 主页、平台年报、媒体采访、人工内容编码与未验证假设分别标注。
2. **结果可复现**：程序固定随机种子并输出 CSV。
3. **结论附带边界**：人数、共同信号聚类、报价中心等都做了敏感性分析。
4. **近似经过校验**：聚合模型另有逐人模拟，用于核对数量级。
5. **不制造虚假精度**：推荐一个结构性中心，同时公开近优区间与模型风险。

## 仓库结构

```text
.
├── README.md
├── requirements.txt
├── src/
│   └── lks_tennis_mc.py
├── reports/
│   └── full-analysis.md
├── data/
│   ├── public-evidence/
│   │   └── lks_recent_video_sample_2026-09-02.csv
│   └── results/
│       ├── illustrative_fine_grid.csv
│       ├── illustrative_requested_quotes.csv
│       ├── direct_person_level_validation.csv
│       ├── one_way_sensitivity.csv
│       ├── participants_sensitivity.csv
│       └── replication_convergence.csv
├── docs/
│   ├── evidence-assessment.md
│   ├── key-results.svg
│   └── reproducibility.md
└── scripts/
    └── create_figure.py
```

## 快速复现

需要 Python 3.10 或更高版本。

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# 快速检查；写入临时目录，不覆盖仓库内的完整模拟结果
python src/lks_tennis_mc.py --quick --section main --output-dir work/quick-check
```

完整模拟的重复次数较高，耗时取决于机器性能：

```bash
python src/lks_tennis_mc.py --section all --output-dir data/reproduced-main
python src/lks_tennis_mc.py --section direct --output-dir data/reproduced-direct
python scripts/create_figure.py
```

更详细的命令、输出与复核顺序见 [复现说明](docs/reproducibility.md)。

## 方法概览

设总报价均值为 `M`，候选报价为 `b`。单次报价的目标同时考虑：

```text
没有更近报价的概率 × 并列时的获胜份额 × 正消费者剩余
```

模型不再把潜在组件解释成真实人口身份，而是使用“较低/较高报价中心 × 独立/共同信号”四种潜在报价机制。共享 Dirichlet 权重只描述多个参与者受相同公开锚点、热门讨论、搜索结果或相似工具输出影响时的相关性，不等于经过测量的 AI 使用率。所有组件比例都是可修改的情景输入。

公开证据的来源、强弱与不能推出的结论见 [证据审计](docs/evidence-assessment.md)；完整的博弈论解释、参数、单次报价比较和局限见 [完整分析](reports/full-analysis.md)。

## 如何解读结果

- `164.74` 是示例参数下的条件性结果，不是基于 LKs 粉丝画像估计出的现实最优价。
- 最大模型风险来自群体构成与各类型报价中心，而不是小数点后几分钱。
- 活动没有公开最终报价分布与完整并列规则，因此任何“唯一最优价”都不应被包装成确定答案。
- `data/results/` 保存的是报告使用的参考运行；自行试验建议写入新的输出目录。

## 数据来源与责任边界

仓库只包含公开活动页面的汇总快照、研究者设定的模型参数和模拟结果，不包含参与者个人数据。网页事实可能随时间变化，外部链接也可能失效；报告中的核验日期应与当前页面区分。

本项目用于博弈论、行为建模和蒙特卡洛方法的教育与复核。使用者应自行确认活动规则、账号要求及适用法律。

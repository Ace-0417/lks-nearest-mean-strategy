# 复现说明

## 1. 环境

- Python 3.10+
- NumPy 1.24–2.x
- 64 位操作系统与 Python

创建独立虚拟环境可以避免项目依赖影响电脑上的其他 Python 项目：

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS / Linux：

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 2. 运行层级

建议先运行快速检查，确认环境与文件写入正常：

```bash
python src/lks_tennis_mc.py --quick --section main --output-dir work/quick-check
```

`--quick` 会减少重复次数，因此结果只应与参考结果比较数量级，不应逐位相等。

主模型默认使用活动截止后页面显示的 LOT 18 报价人数 `N=12,368`；可用 `--participants` 更新。标准分析分为两组：

```bash
# 单次报价、人数与单因素敏感性
python src/lks_tennis_mc.py --section all --output-dir data/reproduced-main

# 计算量较大的逐人校验
python src/lks_tennis_mc.py --section direct --output-dir data/reproduced-direct
```

程序使用固定随机种子；在相同 NumPy 环境中应得到可重复结果。不同 NumPy 版本或平台的随机数实现、浮点运算可能造成细微差异。

默认值对应报告中的辅助先验路线，不是受众统计。可在命令行覆盖它们，而不修改源码：

```bash
python src/lks_tennis_mc.py --quick --section main \
  --upper-strategy-share 0.30 \
  --shared-signal-share 0.10 \
  --anchor-scale 0.80 \
  --output-dir work/alternative-scenario
```

这些参数分别表示较高报价组件占比、共同信号组件占比和相对公开价格锚的偏离尺度。默认的 35% 与 20% 分别承接原始“网球兴趣者”和“AI 辅助者”先验，但在模型中只作为报价机制的代理参数。

程序不会把逐分细网格硬编码在理论均值附近。每次主运行都先完成以下搜索审计：

1. 在参考价 1,500 元的 5%–20% 经济宽域上建立 `75.13–300.13` 元错相位网格，步长 0.25 元；
2. 取包含粗网格最高点的连续 90% 相对效用平台；
3. 把该平台中点吸附到最近的 0.50 元，并以其为中心左右各展开 1 元；
4. 在局部窗口中逐分搜索；
5. 检查两端相对效用是否都低于内部峰的 50%，否则自动扩窗。

默认完整运行得到粗峰 164.63、连续 90% 粗平台 164.38–164.88、标准化中心 164.50，因而生成 163.50–165.50 元细网格。`--quick` 的粗搜索只有 10,000 次，单个最高点可能因噪声变化；平台中点规则降低了这种变化，但快速模式仍只用于链路检查。程序输出的 `smoothed_choice` 给出 15 个相邻分币的效用平滑结果；`operational_choice` 在其 99% 近优平台内选择最接近解析中心的分币。

## 3. 输出对应关系

| 文件 | 含义 |
|---|---|
| `global_coarse_grid.csv` | 75.13–300.13 元、0.25 元步长的宽域定位结果 |
| `illustrative_fine_grid.csv` | 辅助先验路线下 163.50–165.50 元的单次报价分币网格 |
| `illustrative_requested_quotes.csv` | 辅助先验路线下的指定单次候选报价对比 |
| `participants_sensitivity.csv` | 总参与人数变化下的单次报价结果 |
| `replication_convergence.csv` | 不同重复次数的收敛检查 |
| `one_way_sensitivity.csv` | 参数单因素敏感性 |
| `direct_person_level_validation.csv` | 显式生成逐人报价的校验 |
| `simulation_summary.json` | 解析中心、方差分解、粗/细搜索审计、原始极值、平滑区间与操作报价 |

候选报价 CSV 中的 `p_win_pct_mc_se` 和 `utility_v1500_mc_se` 是单点蒙特卡洛标准误。候选使用共同随机数，比较两点时不能把这两个边际标准误当作相互独立；如需正式两两推断，应在模拟中保留每轮配对差值。

## 4. 推荐的复核顺序

1. 阅读 `reports/full-analysis.md` 中从辅助先验到 164.74 元的推导。
2. 用 `--quick --section main` 检查环境。
3. 对照 `global_coarse_grid.csv` 检查宽域峰值不在边界，并由粗峰复算半元中心。
4. 对照 `illustrative_fine_grid.csv` 检查两端边界、164.7 元附近的平台与 165.00 的拥挤效应。
5. 在 `simulation_summary.json` 中核对 `search_audit` 的边界检查和 `predictive_variance_decomposition`。
6. 对照 `participants_sensitivity.csv` 确认人数增加时单次报价中标率下降。
7. 改动一个模型参数并写入新目录，观察程序是否重新定位细网格，以及结论对假设的敏感程度。

不要用快速运行覆盖 `data/results/`；该目录保留报告引用的参考运行。

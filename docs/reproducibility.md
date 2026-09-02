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

标准分析分为两组：

```bash
# 单次报价、人数与单因素敏感性
python src/lks_tennis_mc.py --section all --output-dir data/reproduced-main

# 计算量较大的逐人校验
python src/lks_tennis_mc.py --section direct --output-dir data/reproduced-direct
```

程序使用固定随机种子；在相同 NumPy 环境中应得到可重复结果。不同 NumPy 版本或平台的随机数实现、浮点运算可能造成细微差异。

## 3. 输出对应关系

| 文件 | 含义 |
|---|---|
| `baseline_fine_grid.csv` | 163.50–165.50 元的单次报价分币网格 |
| `baseline_requested_quotes.csv` | 指定单次候选报价对比 |
| `participants_sensitivity.csv` | 总参与人数变化下的单次报价结果 |
| `replication_convergence.csv` | 不同重复次数的收敛检查 |
| `one_way_sensitivity.csv` | 参数单因素敏感性 |
| `direct_person_level_validation.csv` | 显式生成逐人报价的校验 |

## 4. 推荐的复核顺序

1. 阅读 `reports/full-analysis.md` 中的“事实、先验、假设”分层。
2. 用 `--quick --section main` 检查环境。
3. 对照 `baseline_fine_grid.csv` 检查 164.7 元附近的平台与 165.00 的拥挤效应。
4. 对照 `participants_sensitivity.csv` 确认人数增加时单次报价中标率下降。
5. 改动一个模型参数并写入新目录，观察结论对假设的敏感程度。

不要用快速运行覆盖 `data/results/`；该目录保留报告引用的参考运行。

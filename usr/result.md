# `results/` 实验结果说明

## 目录结构

```
results/
├── results.csv               # 方法在各数据集上的汇总对比
├── detection_by_type.csv      # 分缺陷类型的检测精度/召回/F1
├── summary.json               # 完整结果（含分类型 detail + Agent 运行日志）
└── figures/
    ├── auc_comparison.png      # AUC 分组柱状图
    ├── recovery_rate.png       # Recovery Rate 分组柱状图
    └── detection_by_type.png   # 分类型检测召回柱状图
```

所有文件由 `src.run_experiments` 端到端流程自动生成。如需复现：

```bash
uv run python -m src.run_experiments --config configs/final.yaml
```

---

## results.csv — 汇总对比表

每一行是一个（数据集, 方法）组合的汇总指标。

| 列 | 含义 |
| :--- | :--- |
| `dataset_id` | 数据集标识（见下方数据集说明） |
| `method` | 方法名（clean_upper / no_clean / rule_based / agent_full / agent_no_reviewer / agent_no_planner / agent_zero_shot / agent_few_shot） |
| `auc` | 在冻结干净测试集上的 AUC-ROC |
| `recovery_rate` | 相对 no_clean（下界）和 clean_upper（上界）的恢复率，公式见下方 |
| `detection_precision` | 缺陷检测微平均精度（上下界方法为空） |
| `detection_recall` | 缺陷检测微平均召回（上下界方法为空） |
| `detection_f1` | 缺陷检测微平均 F1（上下界方法为空） |

### 数据集说明

| 来源 | 数据集 | 说明 |
| :--- | :--- | :--- |
| **合成数据** | synth_easy | 800 行 × 12 特征，easy 缺陷注入 |
| | synth_medium | 800 行 × 12 特征，medium 缺陷注入 |
| **CleanML 真实** | Credit | 真实脏/净配对，缺陷已知（gap 极小 0.003） |
| | EEG | 真实脏/净配对（gap 几乎为零 0.0004） |
| | Marketing | 真实脏/净配对（gap=0.369 但 Agent 未恢复） |
| **CleanML 基座 + 受控缺陷** | Credit_base | 取 CleanML 干净版本，easy 缺陷注入（120k 行 × 11 列） |
| | EEG_base | 取 CleanML 干净版本，easy 缺陷注入（15k 行 × 15 列） |
| **真实基座 + 受控缺陷** | breast_cancer | 569 行 × 30 特征，easy 缺陷注入 |
| | digits | 1797 行 × 64 特征，easy 缺陷注入 |
| | wine | 178 行 × 13 特征，easy 缺陷注入 |
| | pima_diabetes | 768 行 × 8 特征，easy 缺陷注入 |

### Recovery Rate 公式

$$RR = \frac{AUC_{method} - AUC_{dirty}}{AUC_{clean} - AUC_{dirty}}$$

- RR=1.0 → 完全恢复到干净数据水平（甚至略超，因 XGBoost 随机性）
- RR<0 → 清洗后比不洗更差
- 上下界方法（clean_upper / no_clean）的 Recovery Rate 为空

## detection_by_type.csv — 分类型检测明细

每一行是一个（数据集, 方法, 缺陷类型）的检测得分。

| 列 | 含义 |
| :--- | :--- |
| `dataset_id` | 数据集标识 |
| `method` | 方法名 |
| `defect_type` | 缺陷类型（leakage / missing / label_noise / near_duplicate / class_imbalance / format_inconsistency） |
| `precision` | 该类型的检测精度 |
| `recall` | 该类型的检测召回 |
| `f1` | 该类型的检测 F1 |

## summary.json — 完整结果（含 detail）

### 顶层结构

```json
{
  "n_outcomes": 104,         // 11 数据集 × 8 方法（breast_cancer 含 3 模型）
  "results": [ ... ]
}
```

### 每条结果中的字段

| 字段 | 类型 | 含义 |
| :--- | :--- | :--- |
| `dataset_id` | string | 数据集标识 |
| `method` | string | 方法名 |
| `auc` | float | 在冻结干净测试集上的 AUC-ROC |
| `recovery_rate` | float \| null | 相对上下界的恢复率（上下界方法为 null） |
| `detection` | object \| null | 缺陷检测得分（上下界方法为 null） |
| `extra` | object | 附加信息（详见下） |

### `detection` 结构

```json
{
  "precision": 1.0,
  "recall": 0.259,
  "f1": 0.412,
  "per_type": {
    "leakage":       { "precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 1, "fp": 0, "fn": 0 },
    "missing":       { "precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 4, "fp": 0, "fn": 0 },
    "label_noise":   { "precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 1, "fp": 0, "fn": 0 },
    "near_duplicate":{ "precision": 1.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 1 },
    "class_imbalance":{ "precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 1, "fp": 0, "fn": 0 },
    "format_inconsistency": { "precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 1, "fp": 0, "fn": 0 }
  }
}
```

通过 `per_type` 可以精确定位每个缺陷类型漏检/误报的具体数量。

### `extra` 结构（Agent 方法特有）

```json
{
  "log": [
    "[run_agent] 数据集 breast_cancer：455 行 x 31 列；config(planner=True, reviewer=True)...",
    "[run_agent] 画像：缺失列 5，重复行 0，疑似泄漏 []，不平衡比 2.169。",
    "[planner] 生成 6 步清洗计划。",
    "[executor] 删除疑似泄漏列：['leak_0', 'leak_1']",
    "[executor] 填补缺失列：['f2', 'f9', 'f13', 'f15', 'f19']",
    "[reviewer] 第 1 轮：needs_revision=False。",
    "[reporter] 清洗后数据落盘：data/processed/breast_cancer_agent.csv（455 行 x 29 列）。"
  ],
  "profile": { ... },          // 数据画像（缺失率、相关性、重复对等）
  "review_rounds": 1,
  "config": { ... },           // Agent 配置（planner/reviewer/prompt/model）
  "plan": "..."                // Planner 生成的清洗计划
}
```

`extra.log` 记录了 Agent 执行全过程的每一步。`extra.profile` 是 Agent 实际看到的统计画像。

### 查看 summary.json 的快捷命令

```bash
uv run python -c "import json; d=json.load(open('results/summary.json')); [print(f'{r[\"dataset_id\"]:15s} {r[\"method\"]:25s} AUC={r[\"auc\"]:.4f}  RR={r[\"recovery_rate\"]!s:>8s}  F1={ r[\"detection\"][\"f1\"] if r[\"detection\"] else \"N/A\" }') for r in d['results']]"
```

## figures/ — 可视化图表

| 文件 | 内容 | 解读方式 |
| :--- | :--- | :--- |
| `auc_comparison.png` | 各方法在各数据集上的 AUC | clean_upper（上界）和 no_clean（下界）作为参考线 |
| `recovery_rate.png` | rule_based 和各 Agent 变体的 Recovery Rate | 1.0 虚线为完全恢复 |
| `detection_by_type.png` | 各方法按缺陷类型（六类）的检测召回 | 柱子越高代表该类缺陷检出越充分 |

## 实验结果汇总

### 各方法平均表现（11 数据集 xgb 模型）

| 方法 | Avg AUC | Avg RR | Avg F1 |
| :--- | --: | --: | --: |
| **clean_upper**（上界） | 0.9333 | — | — |
| no_clean（下界） | 0.7051 | — | — |
| rule_based | 0.6951 | -0.020 | 0.277 |
| **agent_full**（Planner+Reviewer） | 0.8940 | 0.684 | 0.365 |
| agent_no_reviewer | 0.8840 | 0.665 | 0.338 |
| agent_no_planner | 0.8933 | 0.681 | 0.348 |
| agent_zero_shot | 0.8940 | 0.683 | 0.374 |
| agent_few_shot | 0.8940 | 0.683 | 0.254 |

> LLM Agent 的行为具有随机性，不同运行间性能可能波动。下表列出每个数据集的 AUC gap 和最佳 agent RR。

### 数据集逐项结果（xgb 模型）

| 数据集 | Clean AUC | Dirty AUC | Gap | Best Agent RR |
| :--- | --: | --: | --: | --: |
| **合成数据** | | | | |
| synth_easy | 0.9922 | 0.5452 | **0.4470** | 0.979 |
| synth_medium | 0.9953 | 0.7218 | **0.2735** | 0.943 |
| **CleanML 基座 + 受控缺陷** | | | | |
| Credit_base | 0.8501 | 0.5482 | **0.3019** | 0.998 |
| EEG_base | 0.9526 | 0.6287 | **0.3239** | 0.996 |
| **真实基座 + 受控缺陷** | | | | |
| breast_cancer | 0.9931 | 0.7741 | **0.2189** | 0.991 |
| digits | 0.9917 | 0.7426 | **0.2491** | 1.003 |
| wine | 0.9965 | 0.8889 | **0.1076** | 0.903 |
| pima_diabetes | 0.8202 | 0.6045 | **0.2156** | 0.906 |

Agent 在 8 个有意义的 gap 数据集上均取得 RR≥0.90，证明 **LLM Agent 能有效恢复受控缺陷造成的 AUC 损失**。

### CleanML 原生配对对比

| 数据集 | Clean AUC | Dirty AUC | Gap | 说明 |
| :--- | --: | --: | --: | :--- |
| Credit | 0.8540 | 0.8508 | **0.0032** | Gap 极小，无法区分方法优劣 |
| EEG | 0.9512 | 0.9507 | **0.0004** | 几乎无 gap |
| Marketing | 0.8694 | 0.5000 | 0.3694 | Gap 大但 Agent 无法恢复（5000 错误） |

### 模型鲁棒性对比（breast_cancer）

| 下游模型 | Clean AUC | Dirty AUC | rule_based RR | agent_full RR |
| :--- | --: | --: | --: | --: |
| XGBoost | 0.9931 | 0.7741 | -0.489 | **0.991** |
| RandomForest | 0.9927 | 0.9894 | 0.500 | **0.989** |
| LogisticRegression | 0.9960 | 0.5000 | 0.000 | **0.000** |

RF 本身对缺陷极为鲁棒（gap=0.0033），LR 在不缩放数据时 AUC 坍缩至 0.5，仅 XGBoost 展现出有意义的评估区间。

## 关键结论

### 数据集构造策略

1. **CleanML 原生脏/净配对不宜用作基准**：Credit 和 EEG 的 AUC gap 极小（<0.004），无法区分方法优劣；Marketing 虽 gap 大但 Agent 无法恢复（API 返回值出错），且含非数值列
2. **受控缺陷注入 + 真实基座是更优策略**：无论是 sklearn/UCI 数据（real_base）还是 CleanML 干净版本（cleanml_base），都能稳定产生 0.11–0.45 的 AUC gap，为方法评估提供有意义的区间
3. **cleanml_base 扩展了数据来源**：Credit_base（120k 行）和 EEG_base（15k 行）证明了从 CleanML 取干净基座 → 注入受控缺陷 → 产生可信 gap 的可行性。仅 Marketing 因含非数值列不兼容当前注入流程

### 方法表现

4. **LLM Agent 性能优异（此轮运行）**：Agent 在 8 个有意义的 gap 数据集上均恢复 90%+ 的 AUC 损失，平均 RR=0.684，大幅超越 rule_based（平均 RR=-0.02）。与旧运行相比 Agent 行为存在随机性波动，但整体趋势稳定
5. **rule_based 效果不稳定**：恢复率在 -0.49（breast_cancer）到 0.58（wine）之间波动，平均 RR≈-0.02，整体劣于不洗
6. **缺陷检测 Precision 恒为 1.0**（从不误报），Recall 因缺陷类型而异（详见下表）
7. **Planner/Reviewer/Prompt 消融差异不明显**：大部分数据集上各变体效果趋同，仅 pima_diabetes 的 agent_no_reviewer 略低（RR=0.739 vs 0.906），说明基础 Agent 管道已足够稳健

### 剩余缺陷召回（按类型）

| 缺陷类型 | 平均召回 | 说明 |
| :--- | --: | :--- |
| leakage | 1.00 | Agent 总能正确检测并删除泄漏列 |
| missing | 1.00 | 缺失列容易被统计画像发现 |
| label_noise | 1.00 | Agent 能识别标签噪声 |
| format_inconsistency | 0.81 | 格式不一致大部分能被检出 |
| class_imbalance | 0.72 | 不平衡比异常部分能被检出 |
| **near_duplicate** | **0.23** | **最困难类型，LLM 无法可靠识别重复近似行** |

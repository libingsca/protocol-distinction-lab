# 研究状态

更新日期：2026-09-08

## 当前主问题

在通信容量足够、充分协议存在且局部训练目标可执行时，任务所需的新区分为何仍可能无法被双方共同生成、获得信用并长期保留？

## 实验进度

| 阶段 | 状态 | 核心结果 | 下一动作 |
|---|---|---|---|
| 历史 A–D | 已封账 | 固定语义接口可学习，但联合形成协议仍失败 | 不再以普通调参延续 |
| CM00 Round 1 | 完成 | 仅发现中性平台，无严格损失屏障 | 扩大精确搜索空间 |
| CM00 R2 | 完成 | 57,528 个候选中无稳健严格屏障 | 主动搜索反例 |
| CM00 R3 | 完成 | 找到码义复用屏障；P1、P2 均被反例否定 | 神经桥接 |
| PB01 | 完成，未完整通过 | 30/30 种子复现损失信用成熟；`T=2` 准确率未跃迁 | 独立确认双时间尺度 |
| PB02 | 完成，主判据 7/7 通过 | 30/30 损失转正；29/30 在 T=4 准确率达标 | 已完成独立审计与报告 |
| CM01-RP | 完成，结构阳性 0/6 | 无实际占用近义消息对；审计通过 | 终止历史改写分支；保留阴性结果 |
| CM02 | 完成，12/12 条件结构阳性 | K=4–7，F=1；独立算法审计通过 | 规划 CM03 受控适应，不外推自主形成 |
| CM03 | 完成，主门槛 0/6，通过审计 | C100 全负；1/6 准确率达标；次要候选优势 6/6 | 终止当前候选/预算迁移分支，保留阴性 |
| CM04 | 完成，主判据通过，审计通过 | E1/E2/E3 达标 28/30、24/30、25/30；全部等价性对照通过 | 规划 CM05 训练反馈选择诊断，不外推自主形成 |
| CM05 | 已规划，未执行 | 待检验少量初始化的损失评分能否跨初始化选择更易学编码 | 冻结随机候选池、无反馈基线和复核种子 |

## 当前优先级

1. PB02 正式结果：`results/pb02/20260908-pb02-independent-confirmation/FINAL_RESULTS_REPORT.md`。
2. 保留宽度 4、线性 receiver 在固定 T=4 的负结果；主实验通过不能外推具体时间点。
3. CM01-RP 已完成：`results/cm01_rp/20260908-cm01-rp-readonly/FINAL_RESULTS_REPORT.md`。六模型均未通过冻结结构门槛，不进入 A/B/C/D。
4. 最新分流决定：CM04 相对可学习性主判据与等价性对照通过，规划但不执行 CM05。详见 [CM04 分流决策](CM04_DECISION.zh-CN.md) 与 [CM05 计划](../experiment-plans/cm05-feedback-selection.zh-CN.md)。CM03 和 CM01-RP 原分支继续终止，CM02 静态结论保留。
5. 历史资产包已补齐、131 项校验通过；12 模型×3 分区最终评价复核通过。
6. CM02 正式结果：`results/cm02/20260908-cm02-target-collision-readonly/FINAL_RESULTS_REPORT.md`；计划为 [CM02 计划](../experiment-plans/cm02-target-collision-diagnostic.zh-CN.md)。12 条件是六模型内的配对条件，不是独立重复。
7. CM03 正式结果：`results/cm03/20260908-cm03-injective-adaptation/FINAL_RESULTS_REPORT.md`；冻结计划为 [CM03 计划](../experiment-plans/cm03-injective-adaptation.zh-CN.md)。完成 30 分支、3,000 次更新；不得以次要随机对照优势替代主门槛失败。
8. CM04 正式结果：`results/cm04/20260908-cm04-code-geometry/FINAL_RESULTS_REPORT.md`；固定计划为 [CM04 计划](../experiment-plans/cm04-code-geometry.zh-CN.md)。bit E0 准确率中位数仅 0.341797，阳性仅限相对优势。
9. CM05 新 run_id：`20260908-cm05-feedback-selection`，已规划未执行。外部双射支架与完整训练表评分必须披露，不将受约束候选搜索称为自主协议形成。

## 冻结的 PB02 参数

- 种子：30–59。
- 任务：`y=(0,1,2,0,1)`。
- 当前协议：`S=(0,1,1,2,1)`。
- 目标协议：`S'=(0,1,2,0,1)`。
- 损失时间点：`T_loss=2`。
- 决策时间点：`T_decision=4`。

完整通过门槛见 [PB02 实验计划](../experiment-plans/pb02-dual-timescale-confirmation.zh-CN.md)。

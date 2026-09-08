# Protocol Distinction Lab

研究任务所需的“新区分”如何在模块化智能系统中被提出、获得信用、形成协议并被长期保留。

本仓库是一个以反例、结构判据和最小机制实验为核心的可证伪研究项目。它由最初的 “MoE 能否升级为心智社会” 问题逐步收缩而来；历史 A–D 四条主线已经封账，当前不再通过增加专家、训练轮数或常规调参延续旧路线。

## 当前结论

截至 2026-09-08：

- CM00 Round 1：发现准确率中性平台，但未发现无并列的严格损失屏障。
- CM00 R2：在 57,528 个候选改写中仍未发现严格屏障，CM01 暂停。
- CM00 R3：在“5 状态、3 消息、3 类别”空间找到稳健反例，确认有限槽位中的**码义复用屏障**。
- PB01：30/30 个种子复现 `C0_CE < 0 < C2_CE`，但 `T=2` 的准确率门槛未通过；总判定为未完整通过。
- PB02：独立种子 30–59 的主判据 7/7 通过；30/30 损失转正、29/30 在 T=4 准确率达标。宽度 4 和线性对照在 T=4 失败，具体时间点不普遍。
- CM01-RP：六个历史模型只读诊断完成，按冻结标准结构阳性 0/6；不进入 A/B/C/D 改写。原始资产包已补齐并校验，历史最终评价 36 项通过。

详见 [研究状态](docs/project/STATUS.zh-CN.md) 和 [项目架构与路线图](docs/project/architecture-and-roadmap.zh-CN.md)。

## 目录

```text
.
├── src/
│   ├── cm00/                    # 精确表格协议搜索（Round 1、R2、R3）
│   └── pb01/                    # 神经桥接实验
├── tests/                       # 独立审计与单元测试
├── docs/
│   ├── project/                 # 架构、状态、证据账本、复现边界
│   ├── preregistrations/        # 预注册文档
│   └── experiment-plans/        # 各轮后续实验计划
├── experiments/
│   └── pb02_dual_timescale/     # 双时间尺度确认实验说明
├── results/                     # 已冻结、可审计的历史结果
│   ├── cm00/
│   └── pb01/
└── evidence_readonly/
    └── checkpoint_archives/     # 原始阶段归档及 SHA-256
```

`results/` 和 `evidence_readonly/` 中已有内容视为只读证据。新运行必须写入新的、唯一的运行目录，不得覆盖历史结果。

## 快速开始

要求 Python 3.10 或更高版本。历史 PB01 环境使用 Python 3.12.13、NumPy 2.3.5。

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -e .
python -m unittest discover -s tests -v
```

运行实验时显式指定新的输出目录：

```bash
python -m cm00.run --output outputs/cm00/my-run
python -m cm00.run_r2 --output outputs/cm00/my-r2-run
python -m cm00.run_r3 --output outputs/cm00/my-r3-run
python -m pb01.run --output outputs/pb01/my-pb01-run
```

PB01 会执行枚举和 30 个神经随机种子，耗时显著高于单元测试。

## 研究纪律

1. 先冻结问题、配置、种子、门槛与源码哈希，再运行正式实验。
2. 区分“历史实测”“当前推导”“待检验假说”和“研究建议”。
3. 不用验证/测试标签选择候选，不把 oracle 诊断描述为自主协议形成。
4. 不覆盖已冻结结果；修正实验必须使用新 `run_id` 并记录原因。
5. 负结果、失败门槛和事后分析均保留，不用新指标替换预注册结论。

## 已知边界

当前仓库包含现阶段 CM00/PB01 的源码、结果和阶段归档，但不包含最初两个输入压缩包的完整展开内容，也不是历史 A–D 全项目复现包。尤其缺少 `OMITTED_JOINT02.json` 所记录的 4,815 个完整历史重放文件。历史 12 个 checkpoint 与 8,192 状态评价资产已于 2026-09-08 以原始诊断 ZIP 补入只读证据目录。详见 [证据账本](docs/project/EVIDENCE_LEDGER.zh-CN.md)。

## 许可证

尚未选择开源许可证。上传 GitHub 前请根据是否允许他人使用、修改和再发布本项目，另行添加 `LICENSE`。

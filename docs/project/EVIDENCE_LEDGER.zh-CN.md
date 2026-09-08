# 证据账本与资产边界

## 仓库内的权威资产

| 资产 | 位置 | 状态 |
|---|---|---|
| CM00 最新累计源码 | `src/cm00/` | 可继续开发；历史结果引用的源码哈希不可改写 |
| PB01 源码 | `src/pb01/` | 本轮保持不变 |
| PB02 源码与审计 | `src/pb02/` | 正式运行哈希已保存 |
| PB02 正式结果 | `results/pb02/20260908-pb02-independent-confirmation/` | 主判据与审计通过；保留架构负结果 |
| 单元测试 | `tests/` | 可扩展 |
| CM00 三轮正式结果 | `results/cm00/<run_id>/` | 只读 |
| PB01 正式结果 | `results/pb01/20260908-pb01/` | 只读 |
| CM01-RP 正式结果 | `results/cm01_rp/20260908-cm01-rp-readonly/` | 结构阳性 0/6，审计通过 |
| 历史最小诊断资产包 | `evidence_readonly/joint02_minimal_assets/20260908/` | 新补齐原包；131 项校验通过 |
| 四个阶段原始归档 | `evidence_readonly/checkpoint_archives/` | 只读；由 `SHA256SUMS` 校验 |
| 总架构、预注册和实验计划 | `docs/` | 文档演进须通过新文件或明确修订记录 |

外层重名的 `FINAL_RESULTS_REPORT.md`、`FINAL_RESULTS_REPORT(1).md`、`(2)`、`(3)` 与各运行目录中的报告逐字节相同，因此仓库只保留运行目录内的权威副本。

## 上游资产补齐记录（2026-09-08）

此前校验过的两个输入包为：

- `研究阅读与提示词包.zip`：18 项清单校验通过。
- `后续诊断最小实验资产.zip`：131 项清单校验通过。

后者原本包含 12 个 joint02 模型、训练数据、完整 8,192 状态评价数据、冻结源码、历史账本、`evaluate_frozen.py` 和 `OMITTED_JOINT02.json`。此前未进入仓库；2026-09-08 用户提供原包后，已按原始字节归档至 `evidence_readonly/joint02_minimal_assets/20260908/后续诊断最小实验资产.zip`。131 项文件大小与 SHA-256、ZIP CRC 校验通过，包哈希为 `f7f6c249f40edf82e61aca68be30091f8a57ca0a6d0c0bf07cf8f13028c3849c`。本地解压副本位于 outputs，原始归档仍只读。

此外，最小实验资产包本身明确省略了 4,815 个全历史重放文件，也不包含前三条历史主线的模型数据。即便未来补入该包，本项目仍只能称为“冻结模型诊断包”，不能称为历史 A–D 全项目复现包。

## 证据保护规则

1. 不修改或覆盖 `results/` 下已有运行目录。
2. 不解包后重新压缩 `evidence_readonly/checkpoint_archives/`；用 SHA-256 判断原件是否改变。
3. 对历史报告的勘误以新文档记录，不静默编辑原报告。
4. 新正式运行必须保存 `config.lock.json`、`environment.json`、`source_hashes.json` 和正式报告。
5. 任何缺失资产都按“未提供”记录，不能用推断生成的替代品冒充原始证据。

CM01-RP 在训练资产白名单下完成只读诊断后，单独执行历史最终评价；12 模型×3 分区共 36 项检查通过。这仅复核最终状态，不补足缺失的全训练轨迹。

# CM01-RP 只读诊断

首轮：`20260908-cm01-rp-readonly`。冻结方案见 `docs/experiment-plans/cm01-rp-frozen-20260908.zh-CN.md`。

历史依赖：Python 3.11.16、NumPy 1.26.4、PyTorch 2.2.2。请使用独立环境；不要替换 PB02 的 NumPy 2.3.5 环境。

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m cm01_rp.run --assets /path/to/02_实验资产 --output outputs/cm01_rp/<new_run_id>
PYTHONPATH=src python -m cm01_rp.audit --output outputs/cm01_rp/<new_run_id>
```

`--output` 必须为不存在的新目录，且在资产目录之外。先从原始 ZIP 解压至本地 outputs 下，并核对 PACKAGE_MANIFEST。
首轮原始包归档在 `evidence_readonly/joint02_minimal_assets/20260908/`。
runner 在模型载入前冻结配置、源码哈希和输入清单，限制诊断进程只能读取冻结 core.py、六个最终模型及三份训练数据。资产完整性校验只读取文件字节作哈希；不解析评价标签。
独立审计从保存的响应重新计算概率、TV、JS、占用、覆盖和结构判定。

诊断完成后，才可在单独进程执行包内已审阅的 evaluate_frozen.py，输出到本轮目录。这是历史最终状态复核，不能称为完整训练重放或新的测试集确认。
本轮 0/6 模型达标；没有训练或生成改写候选。任何后续不同判据必须另立计划和 run_id，保留本轮阴性结果。

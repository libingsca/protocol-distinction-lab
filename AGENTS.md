# AGENTS.md

面向 AI 编码代理的项目约束。

## 当前任务

下一项实验是 PB02。先阅读：

1. `docs/project/STATUS.zh-CN.md`
2. `docs/experiment-plans/pb02-dual-timescale-confirmation.zh-CN.md`
3. `results/pb01/20260908-pb01/FINAL_RESULTS_REPORT.md`

## 不可破坏的约束

- `results/` 与 `evidence_readonly/` 中已有文件视为只读证据。
- 不用事后指标替换预注册主判据。
- 不读取验证/测试标签选择训练候选。
- 不把 oracle、参数注入或完整任务表搜索描述为自主协议形成。
- 修正实验使用新的 `run_id`，不得覆盖旧结果。
- 保留负结果、失败门槛、配置锁、环境记录和源码哈希。

## 工程约定

- Python 包位于 `src/`，测试位于 `tests/`。
- 本地试验输出写入 `outputs/<experiment>/<run_id>/`。
- 运行测试：`python -m unittest discover -s tests -v`。
- 提交前检查 `git diff --check`，并确认没有缓存、虚拟环境或临时输出进入 Git。

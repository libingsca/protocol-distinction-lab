# PB02：双时间尺度独立确认

实现入口：`src/pb02/run.py`；独立结果审计：`src/pb02/audit.py`。

```bash
python -m unittest discover -s tests -v
python -m pb02.run --output outputs/pb02/<new_run_id>
python -m pb02.audit --output outputs/pb02/<new_run_id>
```

输出目录必须不存在。运行前写入配置、源码哈希和环境记录，逐种子保存完整 0–32 步曲线。
主架构为宽度 8 的 tanh receiver；宽度 4、16 与直接线性 receiver 仅作次要对照。

## 实现口径（首轮运行时固定）

- 不修改 PB01 源码，仅复用其 receiver、训练函数及同距离对照池定义；不读取 PB01 逐种子结果。
- 主目标严格固定为计划中的 `(0,1,2,0,1)`。PB01 曾按 receiver 枚举消息排列，本轮不做该搜索。
- 判据 7 的计划未写聚合方式，沿用 PB01 中位数比较口径，同时报告逐种子同时超过两种对照的数量；两种对照都必须通过。
- 首次损失转正定义为 `C_CE>0`；首次决策跃迁定义为 D−C 准确率至少 0.10；逐整数步记录到 32，未发生记为 null。
- 线性模型为 `one-hot @ W + b`，W 使用标准差 0.3 的正态初始化，b 为 0；预训练和适配设置与主模型相同。
- 控制池的任务信息计算使用这项受控玩具任务的训练标签，不存在验证/测试标签选择；固定 sender 改写并非自主协议形成。
- 实验通过判定与审计通过判定分离；负结果仍须保存与报告。

冻结计划：`docs/experiment-plans/pb02-dual-timescale-confirmation.zh-CN.md`。

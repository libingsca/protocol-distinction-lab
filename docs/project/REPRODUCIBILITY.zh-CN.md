# 复现与多电脑同步指南

## 首次上传 GitHub

先在 GitHub 创建一个空仓库，不要勾选自动生成 README、`.gitignore` 或许可证。然后在本项目根目录执行：

```bash
git init -b main
git add -A
git commit -m "chore: initialize protocol distinction lab"
git remote add origin git@github.com:<owner>/<repository>.git
git push -u origin main
```

如果本地目录已经初始化，只需从 `git add -A` 开始。许可证尚未决定，公开仓库发布前应单独确认。

## 首次克隆

```bash
git clone <repository-url>
cd protocol-distinction-lab
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -e .
python -m unittest discover -s tests -v
```

若需要尽量接近 2026-09-08 的 PB01 环境：

```bash
python -m pip install -r requirements/frozen-2026-09-08.txt
```

## 多电脑工作约定

开始工作前：

```bash
git switch main
git pull --ff-only
git switch -c experiment/<short-name>
```

完成一个可核查的小阶段后：

```bash
git add -A
git commit -m "experiment: describe the change"
git push -u origin experiment/<short-name>
```

正式结果先在实验分支生成和审阅，再合并到 `main`。不要让两台电脑同时向同一个未同步分支写入大型结果文件。

## 运行目录命名

建议格式：`YYYYMMDD-<experiment>-<purpose>`，例如：

```text
outputs/pb02/20260909-pb02-independent-confirmation/
```

`outputs/` 是本地草稿输出，默认不进入 Git。通过全部审计且报告完成后，再把需要长期保存的结果复制到：

```text
results/<experiment>/<run_id>/
```

## 结果进入 Git 的最小清单

- 冻结配置；
- 环境与依赖版本；
- 源码哈希；
- 原始逐项/逐种子结果；
- 聚合摘要；
- 审计报告；
- 正式结果报告。

大于 GitHub 普通文件限制的 checkpoint 或数据集，不应直接提交。需要时使用 Git LFS 或独立版本化数据存储，并在证据账本中记录不可变标识和校验和。

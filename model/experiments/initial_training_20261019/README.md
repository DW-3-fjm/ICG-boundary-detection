# 初次训练 PR 实验记录与复现入口

本目录补齐此前仅保存在本机的实验过程，面向算法同学回答：运行了哪些代码、如何准备数据、怎样选权重、分数如何计算。历史运行日期为 **2026-10-19**，仓库补录日期为 **2026-10-20**。

**历史结果来自本地真实图片；本目录不含患者图片、LabelMe 标注、mask、预测图、视频或权重。原始医疗数据依照根 README 的规定线下交接，不上传 GitHub。**

## 先读哪些文件

| 文件 | 用途 |
|---|---|
| [DATASET_CARD.md](DATASET_CARD.md) | 数据规模、标签含义、分组和原始数据交接要求 |
| [split_manifest.csv](split_manifest.csv) | 25 张 Z 图的匿名序号、阶段、来源组与固定划分 |
| [results.json](results.json) | 历史实测汇总、全部逐图指标及评价范围 |
| [training_history.csv](training_history.csv) | 54 轮训练与验证曲线的原始数值 |
| [initialization.json](initialization.json) | 重新初始化的证据及原始记录哈希口径 |
| [validation.json](validation.json) | 本次补录的测试、私有数据准备一致性及发布范围检查 |
| [historical_scripts/](historical_scripts/) | 本轮实际使用的本地包装脚本公开副本，仅供核对流程 |
| [../../configs/initial_z_holdout.yaml](../../configs/initial_z_holdout.yaml) | 路径可配置的公开训练配置，无原电脑绝对路径 |
| [../../scripts/initial_training.py](../../scripts/initial_training.py) | 可移植的数据准备、从零训练及原图评估入口 |

## 两轮结果的范围

| 运行 | 数据与用途 | 结果含义 |
|---|---|---|
| 2026-10-12 少量及全量拟合 | 50 张 Q/Z/H 均用于训练；无独立验证和测试 | 最长拟合运行 Dice 0.9896，属于训练集拟合，不是泛化准确率 |
| 2026-10-19 Z 主任务留出 | 25 张 Z；按 5 个来源组整体划分，训练 15、验证 5、测试 5 | 同一患者内来源组留出，测试 Dice 0.5729；不代表独立患者或视频性能 |

本目录完整记录第二轮。Z 是显像中阶段的主任务；Q/H 暂留作后续辅助对照，L 是器械标注。前景为用户确认的多边形内部，不能依据绿色亮区自动反转内外。

## 当时具体运行了什么

历史核心代码版本为 `3c1a9d20be0b5696c544beeab3fcc987fe771eba`，原运行名为 `z_holdout_20261019_01`，代码分支为 `codex/review-training-readiness`。

1. `inspect_z_split.py` 读取本地转换映射，筛选 Z 原图，检查像素去重、SSIM 与 ORB/RANSAC 外观匹配，并生成人工复核图。
2. `prepare_z_holdout.py` 固定来源画面组和 seed=42，输出 fullres 原尺寸及 resized 训练副本，冻结划分和本地配置。
3. `run_z_holdout.py` 读取配置，直接创建 `Trainer(config)` 并调用 `trainer.train()`；与仓库 `train.py` 使用同一个训练器。它没有调用另一个自定义网络。
4. 编排脚本加载验证集选定的 `best_model.pt`，使用 `ICGPredictor.predict_image()` 和 `Evaluator.evaluate_pairs()` 在原图上逐图评价。汇报 PDF 读取保存的结果，排版脚本不参与训练或打分。

历史副本位于 `historical_scripts/`，原位置是仓库外工作目录的 `tmp/`。这些副本保留原运行逻辑和目录假设，**不要从归档目录直接执行**。公开副本替换了原电脑的数据路径；本地编排脚本也曾做日志句柄和重复绘图清理，没有修改已冻结的训练配置、数据划分或结果，也未因此重跑测试。

新入口把这套流程迁移到仓库中，路径由 YAML/命令行控制。下面的命令属于可移植复现流程；历史分数不是执行新入口后伪造或重复评价得到的结果。

## 可移植复现命令

从仓库根目录运行，`python` 指已按 [训练准备文档](../../TRAINING_READY_CN.md) 安装依赖的解释器。本机已验证的训练版本是 PyTorch `2.7.1+cu128`；CPU 可用于合成测试。原始数据需先按项目规则线下交接。

### 1 准备本地数据

已有本轮转换结果时，使用包含 `train/images/`、`train/masks/` 和 `conversion.private.json` 的私有 fullres 目录：

```powershell
python model/scripts/initial_training.py prepare --converted-root model/data/preliminary_20261012_01/fullres --output model/data/initial_training_20261019
```

只有原始 LabelMe 文件时，可以从私下交接的原始目录重建：

```powershell
# 将示例 source 路径改为已交接的本地目录；保留原目录结构。
python model/scripts/initial_training.py prepare --source D:/private/icg_labelme --output model/data/initial_training_20261019
```

原始转换先对 Q/Z/H 编匿名序号，整张排除 L，再按公开清单仅选择 Z。这样保持历史 50 张目标图的编号，不等于把 Q/H 放入本轮训练。原目录排序改变可能改变编号；如映射核对失败，先对齐线下映射，不要手改清单以强行通过。

转换和准备目录必须是新目录；所有患者内容及私有映射只写入被忽略的 `model/data/`。原始多边形和源文件不被修改。

### 2 检查数据

```powershell
python model/scripts/check_data.py --config model/configs/initial_z_holdout.yaml --output model/logs/initial_data_check.json
```

此处不把 `source_batch` 冒充患者 `group_id`。检查应显示 `patient_split_verified: false`；本轮只具备来源组隔离，不能通过改组名声称独立患者验证。

### 3 从零训练

```powershell
python model/scripts/initial_training.py train --config model/configs/initial_z_holdout.yaml --device cuda
```

共享 `Trainer` 与 `train.py`，不加载旧权重、不接受续训。已有日志或 checkpoint 时拒绝覆盖；新实验应复制 YAML 并配置新的日志、权重和预测目录。训练读取 resized/train 和 resized/val，验证选择最佳权重，不读取测试图。

### 4 原图评价

```powershell
python model/scripts/initial_training.py evaluate --config model/configs/initial_z_holdout.yaml --split train
python model/scripts/initial_training.py evaluate --config model/configs/initial_z_holdout.yaml --split val
# 模型与超参数固定后执行。测试账本会拒绝第二次测试，不要删除账本后调参重测。
python model/scripts/initial_training.py evaluate --config model/configs/initial_z_holdout.yaml --split test
```

新入口明确读取 fullres 原图，模型内部仍缩放到 512×288，再把 mask 映射回原图，与原图 mask 比较。它保存逐图指标、预测 mask 和 checkpoint/划分校验记录。

直接用 `scripts/evaluate.py` 搭配这份训练 YAML 会读取 `resized/test`；所得分数和距离不是本目录历史报告的原图口径。需要原图指标时使用上面的新入口，或明确创建指向 fullres 的独立评估配置。像素距离均没有毫米标定。

## 历史配置与结果

U-Net，首层 32 通道，4 次下采样，dropout=0.1；输入 512×288 RGB。Dice+CE，AdamW，学习率 0.0003，余弦退火，batch=5，seed=42，AMP。上限 120 轮，patience=30，min_delta=0.001。仅训练集使用翻转、轻旋转、亮度/对比度和噪声增强。

原运行第 54 轮早停，按验证 Dice 选择第 24 轮，测试在选择后评价 1 次。验证选择时的输入分辨率 Dice 为 0.8001；原图验证 Dice 为 0.8008，两种口径不能混用。

| 集合 | 图片 / 来源组 | 平均 Dice | 平均 IoU | 平均逐图 HD95 原图像素 |
|---|---:|---:|---:|---:|
| 训练 | 15 / 3 | 0.8119 | 0.7043 | 967.13 |
| 验证 | 5 / 1 | 0.8008 | 0.6693 | 897.70 |
| 测试 | 5 / 1 | 0.5729 | 0.4097 | 1325.73 |

测试平均 Precision=0.4685、Recall=0.7670，出现明显前景扩大、远处碎片及边界偏移。不能只展示重合度，或把训练集 0.9896 当作实际成功率。历史原图尺寸为 3840×2160。

## 验证与限制

```powershell
python -m pytest model/tests -q -p no:cacheprovider --basetemp model/logs/pytest_initial_training
```

2026-10-20 验证：完整 pytest **66 项通过**，其中新增 8 项初次训练工作流测试。新增测试仅使用合成图，覆盖 LabelMe 到准备数据、来源组泄漏、错误阶段/路径、跨图重复、私有映射不匹配、禁止覆盖、从零训练、原尺寸评价和一次性测试账本。它们证明软件流程，不产生临床性能证据。

新工具在本地私有数据上生成的 25 张图与 mask，在原图及缩小图两种版本上共 100 个文件的解码像素与历史准备结果一致；公开配置的训练超参数也与原配置一致。本次只补录流程、检查数据准备及运行合成测试，没有重新训练临床模型或重复评价临床测试集。详见 [validation.json](validation.json)。

验证/测试各只有 1 个来源组，5 张相关图片不是 5 个独立病例。全部保守视为同一患者；本次还是在先前全量拟合与可视化之后进行的回顾性划分。没有连续视频、消退后参考边界或独立患者，不能声称动态跟踪有效或具有临床准确率。

患者原图、标注、映射及权重的交接见 [数据集说明](DATASET_CARD.md)。原始日志含本机路径，留在本地；公开的是匿名数值曲线和逐图指标。公开 CSV 和可移植 YAML 与原始私有文件的字节不同，`initialization.json` 中的原始哈希不能直接用于校验这两份公开文件。

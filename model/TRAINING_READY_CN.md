# 代码审查与本机训练准备

审查日期：2026-10-12。基线：`develop@6ce9fe6e933de512cb4f8271a9a6792d8c431837`。
审查修改位于 `codex/review-training-readiness` 分支；集成目标为 `develop`，不直接改动 `main`。

## 初次训练资料补录 2026-10-20

2026-10-19 已完成以 Z 为主的来源组留出实验。训练入口、公开配置、匿名 15/5/5 划分、逐图结果与复现步骤见 [初次训练实验目录](experiments/initial_training_20261019/README.md)。本文较早日期的“尚未训练”描述属于准备阶段，不表示当前状态。患者数据、原始标注、私有映射、预测图和权重仍不上传 GitHub。

## 初步训练后的状态补充（2026-10-12）

- 用户已确认 Q/Z/H 为同一目标在显像前/中/后，目标以现有多边形内部为准；L 为器械，不能并入目标前景。仅器械标注的图片缺少目标监督，单独保留，不当作目标区域为空的负样本。
- 本地已完成真实图片的区域 mask 转换、等比 RGB 输入的少量样本检查和全部可用目标标注的拟合训练，生成权重、逐图原分辨率指标、阶段汇总与预测对比。所有训练图片仍归为同一组，没有独立患者验证/测试；下面审查阶段“尚未训练”的描述是此前状态。
- 训练后的排查发现：在最近邻放大的二值轮廓上采样双线性放大的概率，会让边界筛选受输出分辨率影响。已改为采样与 mask 一致的模型原生像素概率，保留原阈值；概率图仍以双线性插值返回。此修正不修改分割 mask，也不把启发式分数变成校准置信度。
- 补充高/低阈值与整数/非整数上缩放及下缩放的回归检查后，58 项测试通过。患者数据、私有清单、训练日志、预测图与权重均留在 gitignore 排除的本地目录。
- 下一步需要独立患者与连续视频，验证消退后的漂移、失效与恢复。图像边缘造成的闭合轮廓和器械遮挡仍是尚未解决的显示/跟踪问题。

## 本机是否够用

本机读取得到：RTX 5070 Ti Laptop GPU，显存 12,227 MiB；Core Ultra 9 275HX，24 个逻辑处理器；系统内存约 31.4 GiB（通常称 32 GB）；检查时 C 盘可用约 195 GiB，内存可用约 12.6 GiB。可用资源会随其他程序变化。

对当前二维 U-Net 基线、静态图像分割、单模型消融和视频离线演示，本机适合承担初期训练。当前默认输入是 256×256，四次下采样，首层 64 通道，batch=8。新增 `configs/laptop.yaml` 使用首层 32 通道、batch=8、混合精度，作为更保守的起点。512×512 时先从 batch=2 或 4 试起，再实测调整。

图像从 256×256 增至 512×512 时像素数变为四倍，激活和计算负担明显增加，不能仅靠缩小图片解决边界细节问题。32 通道与 64 通道模型也应在相同患者划分下比较精度。显存不足先降 batch，训练耗时过长再评估是否需要远程设备；目前没有必须租服务器的证据。

笔记本训练时接电并使用合适的散热/性能模式；测试时记录功耗限制及其他 GPU 程序。长期速度会受散热影响。已取得标注文件，但尚无真实数据上的单 epoch 时间、收敛时间或模型性能结论。

## 审查发现与修改

| 级别 | 原问题及影响 | 本地处理 |
|---|---|---|
| P1 | `main@ab576212` 缺少 `icg_pipeline/data`，包入口仍导入它，无法直接运行 | 本轮基于包含该模块的 develop；应通过验证后按仓库规则合入 develop，再由负责人整合 main |
| P1 | 默认 YAML 的 `1e-4` 等在 PyYAML 中读作字符串，优化器/调度器/早停不能按数值运行 | 改成 `1.0e-4` 等明确浮点写法，并用真实配置构造优化器验证 |
| P1 | 注意力模型的通道、参数顺序和分辨率不匹配；双线性分支拼接通道错误 | 修正 gate 输入及尺寸、补充上采样通道投影，四种组合均检查反向传播 |
| P1 | 训练和独立评估都会构造 train/val/test 全部加载器 | 增加按需加载入口；训练只读取 train/val，评估只读取所选集合 |
| P1 | 数据增强概率、角度及亮度等配置不生效，训练中固定翻转和 90 度旋转 | 按 YAML 控制配对几何变换及图像亮度/对比度/噪声；验证和测试禁用增强；不丢弃尾批 |
| P1 | 评估先缩小人工标注再测边界距离，原分辨率误差被改变；保留全部概率图占内存 | 按原图逐张预测及评价，保留标量结果和文件对应关系，距离单位明确为原图像素 |
| P1 | 汇总距离时剔除无穷值，空预测等失败样本会被隐藏 | 整体统计保留失败影响，另报非有限值数量和明确标注的有限子集均值；JSON 非有限值写为 null |
| P2 | HD95/表面距离构造全部点对距离矩阵，内存随两条边界点数乘积增长 | 改用 KD-tree 最近邻查询，保留原有距离定义，并与矩阵参考结果核对 |
| P2 | CUDA 不可用时静默改用 CPU，混合精度又未按设备控制 | 显式请求 CUDA 时失败即报错；CPU 需主动指定；AMP 仅在 CUDA 上开启 |
| P2 | 续训未保存最优历史、早停、混合精度和随机状态，缺少最新权重文件 | 新增 `last_model.pt`，保存并恢复上述状态，兼容读取旧格式中已有状态 |
| P2 | 视频结果长期保留所有帧的全分辨率掩码/概率图 | 默认只保留状态与轮廓元数据；需要大数组时显式设置 `retain_frame_arrays=True`；补充含读取与叠加/编码的逐帧耗时 |
| P2 | 尚无下载后可直接执行的数据检查与本机负载测试 | 新增 `check_data.py` 和 `benchmark.py`；前者检查配对、尺寸、编码、跨集合完全重复和可选患者分组 |
| P1 | LabelMe JSON 不能直接作为现有训练器的 mask 输入 | 新增显式标签映射、患者分组 CSV 驱动的转换工具；保留原尺寸、优先读取内嵌图像、拒绝图像不一致，支持整张排除未知标签样本 |
| P1 | 彩色覆盖图、多类别 mask 或 16 位文件可能被灰度解码后静默合并 | 训练、预检查、推理和评估统一验证位深/通道；二值 mask 只允许背景 0 和单一前景值 |
| P1 | 单患者图片分成 train/val/test 容易被误称为独立患者评估 | 新增 `--train-only` 转换及 `--overfit` 训练检查；日志、checkpoint、TensorBoard 和 Evaluation 明确标记训练集拟合；续训禁止混用两类历史 |
| P2 | 多个相同 stem 的 mask 可能任意选中；最优 checkpoint 的 top-k 历史晚一轮 | 拒绝歧义文件配对，支持大写扩展名；先更新 top-k 历史再保存最优 checkpoint |
| P1 | 大尺寸输入的 FP16 Dice 概率求和可能超过 65504，导致损失计算失真 | Dice 的 softmax 与像素归约使用 FP32；增加 288×512 半精度输入的损失/梯度回归测试 |

## 安装与验证

RTX 50 系列需要支持其架构的 CUDA 版 PyTorch。这里选用可复现的 `torch==2.7.1` / CUDA 12.8，而不是笼统执行 `pip install torch`。PyTorch 官方 [2.7 发布说明](https://pytorch.org/blog/pytorch-2-7/)说明了 Blackwell 和 CUDA 12.8 支持；[官方版本安装表](https://pytorch.org/get-started/previous-versions/)列出对应安装命令。这是本次选定的验证环境，不表示最新版本。

在仓库根目录执行；已安装的本地环境位于仓库根目录 `.venv`。以下 `python` 指用于创建环境的 Python 3.12：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r model/requirements.txt
.\.venv\Scripts\python.exe -m pytest model/tests -q
.\.venv\Scripts\python.exe model/scripts/benchmark.py --config model/configs/laptop.yaml --output model/logs/benchmark_laptop.json
```

显卡驱动能够运行对应 CUDA wheel 时，不必为这些预编译操作另外安装完整 CUDA Toolkit。环境验证要包含实际 GPU 前向和反向计算，仅看到 `nvidia-smi` 或 `torch.cuda.is_available()` 不足以确认所有算子可运行。

## 数据下载后怎么开始

1. 将患者数据放在本地私有目录；可以使用 `model/data/`，该目录已被 gitignore 排除。数据、真实患者对应表、权重和结果图不要提交到 GitHub。
2. 确认图像模态、前景含义和 mask 格式。现代码只支持 8 位图像、背景为 0 的二值区域 mask。LabelMe 的闭合区域多边形可用新增工具转换，操作见 [LabelMe 数据准备与单患者训练检查](LABELME_TRAINING_CN.md)。开放边界线、彩色覆盖图和 16 位原始近红外图需要另外适配。
3. 以患者为组划分 train/val/test；同一患者多次手术也放在同一组。准备私有 CSV，列名为 `split,filename,group_id`，每个图片一行。group_id 使用私有匿名标识；文件放在数据目录，勿提交。
4. 把 `configs/laptop.yaml` 复制为 `configs/laptop.local.yaml`，填写实际数据路径、通道和独立实验输出目录。两个 `in_channels` 必须一致。每次实验使用新的 checkpoint/log 目录，避免覆盖已有记录。
5. 运行检查，确认配对和分组，再抽看图像/标注。先在少数训练样本上检查学习和过拟合能力，然后开始完整训练。测试集不用于选阈值、后处理、模型或 epoch。

数据目录约定：

```text
model/data/
  train/images/a.png    train/masks/a.png
  val/images/b.png      val/masks/b.png
  test/images/c.png     test/masks/c.png
  groups.csv
```

从仓库根目录执行：

```powershell
.\.venv\Scripts\python.exe model/scripts/check_data.py --config model/configs/laptop.local.yaml --manifest model/data/groups.csv --output model/logs/data_check.json
.\.venv\Scripts\python.exe model/scripts/train.py --config model/configs/laptop.local.yaml --device cuda
# 中断后续训：epochs 表示计划的总轮数，而不是额外增加的轮数
.\.venv\Scripts\python.exe model/scripts/train.py --config model/configs/laptop.local.yaml --device cuda --resume model/checkpoints/laptop_baseline/last_model.pt
# 模型及超参数固定后，进行独立测试
.\.venv\Scripts\python.exe model/scripts/evaluate.py --config model/configs/laptop.local.yaml --model model/checkpoints/laptop_baseline/best_model.pt --split test --output model/logs/laptop_baseline/test_results.json
```

若省略分组清单，检查工具只能确认文件级条件，会明确返回 `patient_split_verified: false`。它不能通过文件名或像素哈希识别所有相邻帧泄漏。mask 语义、采集设备差异和人工标注质量仍需团队确认。

## 效率测量如何解释

`benchmark.py` 不需要真实数据，使用已在 GPU 上的随机张量，计时包含前向、损失、反向、梯度裁剪和 AdamW 更新。它报告平均/中位批次耗时、图片吞吐量、峰值 allocated/reserved 显存。解码、数据增强、传输、验证、保存权重及视频光流不在这个计时内，因此不是实际训练总速度，也不是端到端实时 FPS。

训练部分耗时的粗略下界可写为：`epochs × ceil(训练图片数 / batch_size) × 平均批次秒数`。实际总时长还需加入验证、I/O、初始化、保存以及散热降频影响。数据到位后，完整跑几轮再决定正式计划。

本机实测（2026-10-12，PyTorch 2.7.1+cu128，混合精度，5 次预热 + 20 次计时）：

| 配置 | 参数量 | 平均批次耗时 | 训练计算吞吐量 | 峰值 allocated / reserved |
|---|---:|---:|---:|---:|
| 256×256，首层 32 通道，batch=8 | 7,762,498 | 0.086 秒 | 92.7 张/秒 | 0.97 / 1.17 GiB |
| 原默认：256×256，首层 64 通道，batch=8 | 31,036,546 | 0.247 秒 | 32.4 张/秒 | 2.13 / 2.42 GiB |
| 512×512，首层 32 通道，batch=4 | 7,762,498 | 0.176 秒 | 22.7 张/秒 | 1.83 / 2.30 GiB |

这是短时随机张量测试，不能直接换算成真实训练总时长。显存统计不含桌面及其他应用占用，也不保证其他输入/优化器同样占用。详细 JSON 保存在本地 `model/logs/benchmark_*.json`，已排除版本控制。

## 本轮验证和新发现的数据格式

- 本次回归测试：50 项通过，覆盖模型四种配置的反向传播、增强配置、仅所需集合加载、失败边界汇总、断点随机状态、分组检查、LabelMe 转换、未知标签排除、单患者训练检查和大尺寸 FP16 Dice。
- PyTorch CUDA 矩阵前向/反向实测通过；三组完整 U-Net 训练计算基准通过。
- 使用 20 张合成图片完成 CUDA 混合精度两轮训练、checkpoint 保存、原图分辨率 CLI 评估及数据预检查。合成数据只验证软件，不是肺段训练数据或性能证据。
- 本次追加：9 张合成 RGB 图像完成 LabelMe 清单生成、转换、分组预检查、CUDA 混合精度训练及续训至第 3 轮、独立合成 test 集 2 张原分辨率 CLI Evaluation 和单图推理；又用同一合成来源检查仅训练集模式（288×512 输入），训练并续训至第 2 轮，Evaluation 明确标为 `training_fit_only`。两套流程均不使用患者数据。
- 实测复现 Windows 中文路径下 `cv2.imwrite` 失败，改用 OpenCV 内存编码 + Python 文件写入，并同步修改训练数据读取、推理和合成图片工具。
- 后续用户提供的目录含 55 张 PNG、55 份 LabelMe JSON 和两个 RAR；全部 JSON 尺寸标为 3840×2160，共 60 个 polygon，50 张单形状、5 张双形状。标签出现次数为 Q=5、Z=26、H=21、L=8。11 个标注目录各含 5 份 JSON；这并不能证明恰好有 11 位患者。
- 当前 `ICGDataset` 接受已经转换的二值 PNG mask；新增转换工具，但尚未正式转换真实标注或训练。用户已说明 Q/Z/H 分别为吲哚菁绿显像前/中/后，L 含义未知，当前数据为同一患者的不同标注。时相标签不等于分割类别；仍需确认多边形本身的目标结构和各时相标注是否同义。坐标边缘的浮点误差在转换中予以容忍。
- 原图是 16:9，直接压成正方形会改变几何比例；数据适配时应比较保持比例的矩形输入/letterbox 与当前 resize，同时保证图像、标注和回映射一致。已核对到的 30 张外部原图为 8 位三通道，不能据此擅自断定灰度一定优于 RGB。
- 部分 JSON 的 `imagePath` 在当前目录布局中不能直接配对，不能只靠同目录、同文件名猜测；LabelMe 可读取 JSON 内嵌的 `imageData`。转换时必须核对内嵌图与外部图是否一致。

LabelMe 使用单独的 `.venv-labelme` 环境查看图像/标注；安装不改变上述训练环境。JSON 读取与多边形栅格化可以独立于 GUI 实现，见 [LabelMe 官方格式读取示例](https://github.com/wkentaro/labelme/blob/main/examples/utils.py)。

## 仍然需要确认

- 已找到 55 份标注；按用户当前说明，它们来自同一患者。Q/Z/H 是时相，L 暂不使用；多边形的前景医学含义仍需确认。当前只有单患者训练检查的条件，没有独立患者验证/测试或收敛结果。
- 目前的区域外轮廓并不自动等于需要显示的目标分界线；目标区域可能被画面裁切，外轮廓含画面边缘。视频采用 Farneback 光流，也没有实现资料中 TapNet-Siamese 网络。
- HD95 延续原代码“两个方向的 95 分位数取最大值”定义；跨文献对比需核对定义与分辨率。空边界距离按未定义处理，不能把 null 当作 0。
- 配置中的 `monitor_metric` / `monitor_mode` 目前只支持验证 Dice 最大化，传入其他值会明确报错。`use_wandb`、`log_interval` 等可选项未形成完整功能。
- 旧 checkpoint 缺失的随机/早停等状态无法事后还原；续训要维持模型结构与数据处理一致，新 checkpoint 才能恢复新增状态。不同硬件和版本不承诺位级一致。
- 以上是研究软件审查与训练准备，不是已验证的临床效果。

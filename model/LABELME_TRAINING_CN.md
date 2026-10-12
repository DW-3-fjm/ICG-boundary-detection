# LabelMe 数据准备与单患者训练检查

目前已确认：Q、Z、H 分别表示吲哚菁绿显像前、中、后，三个阶段标记同一目标，以现有多边形内部为前景；L 为器械标注。Q/Z/H 是时相信息，不建立三个分割类别。特别是显像前的标注，不能假设来自可见荧光，也不能按绿色亮区自动翻转多边形内外。当前数据按用户说明保守归为同一组，先进行训练集拟合检查。

当前代码仍是二类分割：背景 + 一个经确认的前景。先做单患者训练集拟合检查，验证读取、损失、保存和预测流程，再获取其他患者用于验证和测试。训练集 Dice 高也不能证明跨患者有效。

## 1. 准备私有清单

以下命令从仓库根目录执行，使用已安装的 `.venv`；新机器的环境安装见 [训练准备说明](TRAINING_READY_CN.md)。实际目录、患者分组 CSV、图片、标注和权重均留在本地。

```powershell
# 把路径替换为真实的 LabelMe JSON 根目录；包含 L 的整个样本都会被排除。
.\.venv\Scripts\python.exe model/scripts/convert_labelme.py --input "D:/private/labelme_source" --exclude-labels L --template model/data/labelme_split.csv
```

CSV 列为 `annotation,split,group_id`。工具仅列出相对 JSON 路径，不推测患者身份或划分。当前单患者检查时，所有 `split` 填 `train`，所有 `group_id` 填同一个匿名标识，如 `patient_001`。不要使用真实姓名。之后如获得多患者数据，应按患者分组划分；同一患者的不同手术、时相和相邻帧不能跨集合。

## 2. 转换已确认的区域多边形

本项目使用 Q/Z/H 合并前景的示例；其他数据集仍须先确认各标签表示同一目标。L 是独立器械任务的监督，现有仅含 L 的图片没有目标区域标注，暂从目标区域训练中整张排除。

```powershell
.\.venv\Scripts\python.exe model/scripts/convert_labelme.py --input "D:/private/labelme_source" --manifest model/data/labelme_split.csv --output model/data/labelme_single_patient --foreground-labels Q Z H --exclude-labels L --train-only
```

转换行为：

- 优先读取 JSON 内嵌 `imageData`，没有内嵌图时按 `imagePath` 读取；不通过重名图片猜测对应关系。如可找到外部图，必须与内嵌图的解码像素一致。
- 原始文件不变；输出目录必须是新目录，不能与源目录重叠。输出使用唯一文件名，避免不同文件夹内的 `1.json` 相互覆盖。
- 坐标按最近整数取整，裁切到有效像素，填充多边形内部和边缘；多个前景多边形取并集。容忍图像外缘及微小浮点误差，拒绝明显越界、退化图形、线段、孔洞和未明确映射的标签。
- `--exclude-labels L` 排除含 L 的**整个样本**；不会把缺少目标标注的区域默认为背景。L 不能合并进目标前景；也不能仅凭器械标注把整张图片作为目标不存在的负样本。后续联合训练需完整目标标注或能够屏蔽缺失监督的损失，不能直接把 L 添加为当前二类模型的第三类。
- 输出原尺寸 PNG 图像、0/255 单通道 mask、私有 `groups.csv` 和转换记录。转换记录保留时相标签及来源映射供核查，但不把时相作为模型类别。

抽查转换的 mask，确认区域方向、边缘与原标注一致，再开始训练。

## 3. 配置与训练集检查

复制 `model/configs/laptop.yaml` 为 `model/configs/single_patient.local.yaml`。修改：

```yaml
data:
  train_data_dir: data/labelme_single_patient/train
  # RGB 时 data 和 model 的 in_channels 都设为 3。
  # 3840×2160 原图可先用保持 16:9 的 [288, 512]；尺寸顺序为 H,W。
  image_size: [288, 512]
  in_channels: 3
  batch_size: 2
  augmentations: {}  # 首次过拟合检查先禁用增强
model:
  in_channels: 3
training:
  epochs: 20
  checkpoint_dir: checkpoints/single_patient_fit
  scheduler_params:
    T_max: 20
    eta_min: 1.0e-6
logging:
  log_dir: logs/single_patient_fit
inference:
  model_path: checkpoints/single_patient_fit/best_model.pt
  output_dir: predictions/single_patient_fit
```

上面是**需要修改的字段示意，不是可独立使用的完整配置**；保留复制文件中的其他字段。每次实验用新输出目录。初次先检查少数样本能否过拟合，再扩大到全部可用的已确认标注。

```powershell
.\.venv\Scripts\python.exe model/scripts/check_data.py --config model/configs/single_patient.local.yaml --manifest model/data/labelme_single_patient/groups.csv --splits train --output model/logs/single_patient_fit/data_check.json
.\.venv\Scripts\python.exe model/scripts/train.py --config model/configs/single_patient.local.yaml --device cuda --overfit
# 续训必须继续带 --overfit；epochs 表示总轮数。
.\.venv\Scripts\python.exe model/scripts/train.py --config model/configs/single_patient.local.yaml --device cuda --overfit --resume model/checkpoints/single_patient_fit/last_model.pt
.\.venv\Scripts\python.exe model/scripts/evaluate.py --config model/configs/single_patient.local.yaml --model model/checkpoints/single_patient_fit/best_model.pt --device cuda --split train --output model/logs/single_patient_fit/train_fit.json
```

`--overfit` 用无增强的训练图片计算 train-fit 分数；不要求 val/test 目录。日志与 TensorBoard 使用 `Train-fit`/`train_fit` 标记，checkpoint 标记 `overfit_diagnostic`；不同模式的续训历史不能混用。Evaluation 在原图分辨率计算前景 Dice/IoU 和边界距离，结果标记 `training_fit_only`。这类结果用于检查学习与标注适配，不是独立验证或测试。

## 4. 有多患者数据后

重新生成清单并填入按患者分组的 train/val/test，转换时去掉 `--train-only`，用新的实验目录正常训练，去掉 `--overfit`。转换工具拒绝跨集合患者和跨集合完全相同图片。`check_data.py` 的 `patient_split_verified` 仅表示检查给定清单的覆盖与分组互斥，不能证明填写的患者身份正确。

验证集选择模型、阈值和后处理；测试集仅在方案固定后评价。评估输出也不自行认定患者独立性，必须结合真实的患者清单核对。仅有 L 器械标注的图片保持与目标区域任务分开，可为后续遮挡处理单独准备；不能声称现有代码已经实现器械遮挡恢复。

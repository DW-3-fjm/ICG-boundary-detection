# Historical workflow snapshot for review, not the portable CLI. See ../README.md.
"""Freeze a Z-only, appearance-group held-out pilot without claiming patient independence."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
from collections import Counter

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'ICG-boundary-detection/model'
RUN='z_holdout_20261019_01'
OUT=ROOT/'output'/RUN
LOG=MODEL/'logs'/RUN
DATA=MODEL/'data'/RUN
SOURCE=MODEL/'data/preliminary_20261012_01/fullres/train'
OUT.mkdir(parents=True,exist_ok=True)
LOG.mkdir(parents=True,exist_ok=True)
assert not DATA.exists(), 'Refuse to change an existing frozen split'
audit=json.loads((OUT/'similarity_audit.json').read_text(encoding='utf-8'))
z=sorted([r for r in audit['records'] if r['phase']=='Z'],key=lambda r:r['filename'])
assert len(z)==25 and audit['z_exact_unique']==25
groups=sorted({r['source_batch'] for r in z})
assert len(groups)==5
# Sample counts and group identities are fixed without using masks or previous model scores.
order=list(np.random.default_rng(42).permutation(groups))
assignment={g:('train' if i<3 else 'val' if i==3 else 'test') for i,g in enumerate(order)}
records=[]
hashes={}
for r in z:
    split=assignment[r['source_batch']]
    row=dict(r,split=split,patient_group='single_group_user_reported')
    name=r['filename']
    image=Image.open(SOURCE/'images'/name).convert('RGB')
    mask=Image.open(SOURCE/'masks'/name).convert('L')
    assert image.size==mask.size==(3840,2160)
    raw=np.array(image)
    binary=(np.array(mask)>0).astype(np.uint8)*255
    assert 0<np.count_nonzero(binary)<binary.size
    digest=hashlib.sha256(raw.tobytes()).hexdigest()
    assert digest not in hashes,'Exact image duplicate'
    hashes[digest]=split
    row['decoded_image_sha256']=digest
    row['source_mask_sha256']=hashlib.sha256(binary.tobytes()).hexdigest()
    for folder in ['images','masks']:
        full=DATA/'fullres'/split/folder/name
        full.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(SOURCE/folder/name,full)
    small=cv2.resize(raw,(512,288),interpolation=cv2.INTER_AREA)
    small_mask=cv2.resize(binary,(512,288),interpolation=cv2.INTER_NEAREST)
    for folder,array in [('images',small),('masks',small_mask)]:
        target=DATA/'resized'/split/folder/name
        target.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(array).save(target)
    records.append(row)
counts=dict(Counter(r['split'] for r in records))
assert counts=={'train':15,'val':5,'test':5},counts
for g in groups:
    assert len({r['split'] for r in records if r['source_batch']==g})==1
frozen=json.dumps(records,sort_keys=True,separators=(',',':')).encode()
manifest_hash=hashlib.sha256(frozen).hexdigest()
cross=[p for p in audit['pairs'] if assignment[next(r['source_batch'] for r in z if r['filename']==p['a'])]
       !=assignment[next(r['source_batch'] for r in z if r['filename']==p['b'])]]
summary={'run':RUN,'task':'Z-primary static target-boundary segmentation baseline; dynamic tracking not evaluated',
         'counts':counts,'group_assignment':assignment,'split_seed':42,'model_seed':42,
         'evaluation_scope':'within_patient_source_batch_holdout_NOT_independent_patient',
         'patient_split_verified':False,'patient_group':'single_group_user_reported',
         'foreground':'Existing polygon interior, never invert based on green fluorescence',
         'excluded_from_this_run':{'Q':5,'H':20,'L':5},
         'split_basis':'All 25 Z images; anonymous source-folder batches retained intact after raw-image visual inspection; seeded 3/1/1 group assignment; no prior predictions or scores used',
         'hash_audit':{'exact_duplicates_across_splits':0,'decoded_unique_images':25},
         'similarity_audit':{'cross_split_pairs':len(cross),'max_ssim':max(p['ssim'] for p in cross),
            'max_affine_inliers':max(p['affine_inliers'] for p in cross),
            'limits':'SSIM/ORB are appearance checks only; not proof of independent patients or nonadjacent time frames; no video timeline available'},
         'split_manifest_sha256':manifest_hash,'records':records}
(OUT/'split_manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
(LOG/'split_manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
with (OUT/'split_manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)

base=yaml.safe_load((MODEL/'configs/laptop.yaml').read_text(encoding='utf-8'))
base['data'].update({f'{s}_data_dir':str(DATA/'resized'/s) for s in ['train','val','test']})
base['data'].update(image_size=[288,512],in_channels=3,batch_size=5,num_workers=0,
                    annotation='Z-only; confirmed existing polygon foreground; Q/H reserved as auxiliary, L excluded',
                    augmentations={'horizontal_flip':.5,'vertical_flip':0.,'rotation':8,
                                   'brightness':.05,'contrast':.1,'gaussian_noise':.005})
base['model'].update(in_channels=3,base_filters=32,dropout=.1)
base['training'].update(epochs=120,seed=42,learning_rate=3e-4,overfit_only=False,save_top_k=1,
                        checkpoint_dir=str(MODEL/'checkpoints'/RUN),
                        scheduler_params={'T_max':120,'eta_min':1e-6},
                        early_stopping={'patience':30,'min_delta':.001})
base['logging'].update(log_dir=str(LOG/'training'),use_tensorboard=False)
base['inference'].update(model_path=str(MODEL/'checkpoints'/RUN/'best_model.pt'),output_dir=str(MODEL/'predictions'/RUN))
base['experiment']={'scope':summary['evaluation_scope'],'split_manifest_sha256':manifest_hash,
                     'fresh_initialization':True,'resume':None,'primary_phase':'Z',
                     'test_policy':'Evaluate exactly once after checkpoint selected by validation Dice; do not tune to test results'}
config=MODEL/'configs'/f'{RUN}.local.yaml'
assert not config.exists()
config.write_text(yaml.safe_dump(base,sort_keys=False,allow_unicode=True),encoding='utf-8')

# A labeled contact sheet records all images and their frozen destinations.
im=Image.new('RGB',(2560,1930),'#eef3f7');d=ImageDraw.Draw(im)
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',25)
d.text((24,18),'Z 阶段固定划分：训练 15 / 验证 5 / 测试 5；同一来源画面组不跨集合',fill='#173347',font=font)
d.text((24,58),'仅为同一患者内的来源组留出实验；不是独立患者验证。Q/H 暂不参与本轮。',fill='#607787',font=font)
colors={'train':'#087F80','val':'#A36B00','test':'#7754AD'}
for row_idx,g in enumerate(groups):
    for j,r in enumerate(sorted([r for r in records if r['source_batch']==g],key=lambda r:r['filename'])):
        x=j*512;y=130+row_idx*355
        im.paste(Image.open(SOURCE/'images'/r['filename']).resize((500,281),Image.Resampling.LANCZOS),(x,y))
        d.text((x+6,y+290),f"{r['filename'][:-4]}  {g}  {r['split']}",fill=colors[r['split']],font=font)
im.save(OUT/'Z图片_训练验证测试划分.png')
note='''# Z 阶段主任务：重新初始化训练

项目目标：荧光快速消退过程中的目标边界识别与保持。Z 是主任务数据，Q/H 是后续辅助数据。
现有数据只有静态图片，因此本轮只评估 Z 阶段静态区域分割与边界误差，不声称验证了动态跟踪。
目标按用户确认的现有多边形内部定义，不以荧光绿色区域自动改变前景方向。

25 张 Z 图片来自 5 个来源画面组。人工查看全部原图，并计算原图像素哈希、灰度 SSIM、ORB/RANSAC 外观匹配。
相近画面保留在同一来源组内，使用固定 seed=42 整组分为 3/1/1：训练 15、验证 5、测试 5。
没有根据标注形状、既有预测分数或测试表现挑选测试图片；没有只挑容易图片。
没有删除组内帧，因为相近帧已整体隔离；组内 5 张图片不能视为 5 个独立临床样本。
来源文件夹组不是患者身份；按此前用户说明，全部保守视为同一患者。本实验不是独立患者测试。
没有连续帧时间信息，外观差异与相似性指标不能证明时间独立。

本轮不混入 Q 5 张、H 20 张或 L 5 张；保留供后续受控辅助实验。
新建模型、优化器、调度器；不加载此前 50 张拟合权重，不恢复训练。
训练增强仅用于训练集合：轻微旋转、翻转、亮度、对比度和噪声；验证/测试不增强。
验证集用于选择最佳 checkpoint 和早停，测试只在训练结束后评估一次，不参与选模型或阈值调整。
固定 120 轮上限、早停 patience=30；训练前冻结配置和划分。
网络输入 512×288 RGB；最终在保留的 3840×2160 原图上评价 Dice、IoU 和边界距离。
本次划分是在既有全量拟合及可视化之后建立的回顾性留出实验；仍需要新收集的独立患者与连续视频作后续前瞻性验证。
'''
(OUT/'实验设计与限制.md').write_text(note,encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='records'},indent=2,ensure_ascii=False))
print('CONFIG',config)

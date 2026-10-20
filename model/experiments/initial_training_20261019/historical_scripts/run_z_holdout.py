# Historical workflow snapshot for review, not the portable CLI. See ../README.md.
"""Run exactly one fresh Z-only baseline, select on validation, then evaluate the frozen test once."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import contextlib
import csv
import hashlib
import json
import logging
import math
import os
import re
import sys
import time
import traceback

os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'ICG-boundary-detection/model'
RUN='z_holdout_20261019_01'
LOG=MODEL/'logs'/RUN
OUT=ROOT/'output'/RUN
DATA=MODEL/'data'/RUN
CONFIG=MODEL/'configs'/f'{RUN}.local.yaml'
sys.path.insert(0,str(MODEL/'src'))

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from icg_pipeline import Trainer
from icg_pipeline.utils import load_config
from icg_pipeline.inference import ICGPredictor
from icg_pipeline.eval import Evaluator
from icg_pipeline.data import read_source_image, read_binary_mask, write_image

torch.set_num_threads(4)
TZ=timezone(timedelta(hours=8))
def now():return datetime.now(TZ).isoformat(timespec='seconds')
def safe(x):
    if isinstance(x,dict):return {k:safe(v) for k,v in x.items()}
    if isinstance(x,list):return [safe(v) for v in x]
    if isinstance(x,(float,np.floating)):return float(x) if math.isfinite(x) else None
    if isinstance(x,np.integer):return int(x)
    return x
def dump(path,x):
    temp=path.with_suffix('.writing')
    temp.write_text(json.dumps(safe(x),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)
STATE=LOG/'run_status.json'
assert not STATE.exists(),'Refuse to run an experiment twice'
status={'state':'starting','run':RUN,'started_at':now(),'fresh_initialization':True,'resume':None,
        'test_evaluations_started':0,'test_evaluations_completed':0}
dump(STATE,status)
start=time.perf_counter()
try:
    config=load_config(str(CONFIG))
    manifest=json.loads((OUT/'split_manifest.json').read_text(encoding='utf-8'))
    assert not config['training']['overfit_only']
    assert config['experiment']['split_manifest_sha256']==manifest['split_manifest_sha256']
    checkpoint=Path(config['inference']['model_path'])
    assert not checkpoint.exists(),'Do not reuse existing weights'
    with (LOG/'train.console.log').open('x',encoding='utf-8') as console:
        with contextlib.redirect_stdout(console),contextlib.redirect_stderr(console):
            trainer=Trainer(config)
            assert trainer.current_epoch==0 and not trainer.optimizer.state
            assert len(trainer.train_loader.dataset)==15 and len(trainer.val_loader.dataset)==5
            train_names={Path(p[0]).name for p in trainer.train_loader.dataset.pairs}
            val_names={Path(p[0]).name for p in trainer.val_loader.dataset.pairs}
            test_names={r['filename'] for r in manifest['records'] if r['split']=='test'}
            assert not (train_names&val_names or train_names&test_names or val_names&test_names)
            digest=hashlib.sha256()
            for key,value in trainer.model.state_dict().items():
                digest.update(key.encode());digest.update(value.detach().cpu().numpy().tobytes())
            provenance={'initialized_at':now(),'model_initialized_from_scratch':True,'checkpoint_loaded_before_training':False,
                        'optimizer_initial_state_entries':len(trainer.optimizer.state),'initial_epoch':trainer.current_epoch,
                        'initial_model_state_sha256':digest.hexdigest(),'config_sha256':hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
                        'split_manifest_sha256':manifest['split_manifest_sha256'],'torch_version':str(torch.__version__),
                        'cuda_device':torch.cuda.get_device_name(0),'seed':config['training']['seed'],
                        'train_files':sorted(train_names),'validation_files':sorted(val_names),'test_files_never_loaded_for_training':sorted(test_names)}
            dump(LOG/'initialization_provenance.json',provenance)
            status.update(state='training',stage='fresh_initialization_train_val',train_images=15,val_images=5,test_images=5)
            dump(STATE,status)
            trainer.train()
            completed_epochs=trainer.current_epoch+1
            del trainer
            # setup_logging captures redirected stdout; detach that handler before closing it.
            for handler in list(logging.getLogger().handlers):
                if getattr(handler,'stream',None) is console:
                    logging.getLogger().removeHandler(handler)
                    handler.close()
    torch.cuda.empty_cache()
    selected=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert selected['training_mode']=='train_val'
    best_epoch=int(selected['epoch'])+1
    best_val_input=float(selected['metrics']['dice'])
    del selected
    status.update(state='evaluating',completed_epochs=completed_epochs,best_epoch=best_epoch,best_validation_dice_input_resolution=best_val_input)
    dump(STATE,status)
    predictor=ICGPredictor(config)
    assert predictor.checkpoint_training_mode=='train_val'
    evaluator=Evaluator(config)
    results={}
    # No training or hyperparameter changes occur after the checkpoint is frozen.
    for split in ['train','val','test']:
        rows=[r for r in manifest['records'] if r['split']==split]
        if split=='test':
            assert status['test_evaluations_started']==0
            status['test_evaluations_started']=1
        status['stage']=f'evaluate_{split}_original_resolution'
        dump(STATE,status)
        def pairs():
            for row in rows:
                name=row['filename']
                image=read_source_image(DATA/'fullres'/split/'images'/name)
                gt=read_binary_mask(DATA/'fullres'/split/'masks'/name)
                prediction=predictor.predict_image(image)
                write_image(str(MODEL/'predictions'/RUN/split/'masks_original_resolution'/name),prediction['mask'])
                yield {'mask':prediction['mask'],'image_path':name},gt
        result=evaluator.evaluate_pairs(pairs())
        result.update(split=split,sample_count=len(rows),primary_phase='Z',checkpoint_epoch=best_epoch,
                      completed_epochs=completed_epochs,distance_units='original_image_pixels',original_size=[3840,2160],
                      evaluation_scope='training_fit_only' if split=='train' else manifest['evaluation_scope'],
                      patient_split_verified=False,split_manifest_sha256=manifest['split_manifest_sha256'])
        dump(LOG/f'{split}_evaluation.json',result)
        results[split]=result
        if split=='test':status['test_evaluations_completed']=1
        dump(STATE,status)

    # Retain every test image, including failures, in a compact comparison sheet.
    test_results={r['image_path']:r for r in results['test']['per_sample']}
    f=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',25)
    sheet=Image.new('RGB',(2048,5*640+120),'#eef3f7');draw=ImageDraw.Draw(sheet)
    draw.text((16,14),'Z 留出测试：全部 5 张；同一患者内来源组隔离，非独立患者验证',fill='#173347',font=f)
    draw.text((16,57),'左：人工标注（绿）    右：人工标注（绿）+ 新模型预测（橙）',fill='#607787',font=f)
    for i,row in enumerate([r for r in manifest['records'] if r['split']=='test']):
        name=row['filename']; y=120+i*640
        raw=read_source_image(DATA/'fullres/test/images'/name)
        gt=read_binary_mask(DATA/'fullres/test/masks'/name)
        pred=read_binary_mask(MODEL/'predictions'/RUN/'test/masks_original_resolution'/name)
        raw=cv2.resize(raw,(1024,576),interpolation=cv2.INTER_AREA)
        gt=cv2.resize(gt,(1024,576),interpolation=cv2.INTER_NEAREST)
        pred=cv2.resize(pred,(1024,576),interpolation=cv2.INTER_NEAREST)
        reference=raw.copy();comparison=raw.copy()
        cs,_=cv2.findContours((gt>0).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(reference,cs,-1,(0,255,120),3);cv2.drawContours(comparison,cs,-1,(0,255,120),3)
        cs,_=cv2.findContours((pred>0).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(comparison,cs,-1,(40,165,255),3)
        sheet.paste(Image.fromarray(cv2.cvtColor(reference,cv2.COLOR_BGR2RGB)),(0,y))
        sheet.paste(Image.fromarray(cv2.cvtColor(comparison,cv2.COLOR_BGR2RGB)),(1024,y))
        m=test_results[name]
        draw.text((14,y+588),f"{name}   Dice={m['dice']:.4f}   IoU={m['iou']:.4f}   HD95={m['hausdorff_95']:.1f}px",fill='#173347',font=f)
    sheet.save(OUT/'Z留出测试_全部预测对比.png')

    pattern=re.compile(r'Epoch (\d+)/(\d+) \| Train Loss: ([\d.]+) \| Train Dice: ([\d.]+) \| Val Loss: ([\d.]+) \| Val Dice: ([\d.]+) \| Val IoU: ([\d.]+) \| LR: ([\deE+.-]+)')
    history=[]
    for m in pattern.finditer((LOG/'train.console.log').read_text(encoding='utf-8')):
        ep,total,tl,td,vl,vd,vi,lr=m.groups()
        history.append(dict(epoch=int(ep),train_loss=float(tl),train_dice_augmented=float(td),val_loss=float(vl),val_dice=float(vd),val_iou=float(vi),lr=float(lr)))
    assert len(history)==completed_epochs
    with (OUT/'training_history.csv').open('w',encoding='utf-8-sig',newline='') as fcsv:
        w=csv.DictWriter(fcsv,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
    compact={s:{'count':r['sample_count'],'dice':r['aggregated']['mean_dice'],'iou':r['aggregated']['mean_iou'],
                 'hd95_original_pixels':r['aggregated']['mean_hausdorff_95']} for s,r in results.items()}
    final={'state':'complete','run':RUN,'completed_at':now(),'completed_epochs':completed_epochs,'best_epoch':best_epoch,
           'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
           'fresh_initialization':True,'initial_model_state_sha256':provenance['initial_model_state_sha256'],
           'test_evaluations_completed':status['test_evaluations_completed'],'scope':manifest['evaluation_scope'],
           'split_manifest_sha256':manifest['split_manifest_sha256'],'metrics_original_resolution':compact,
           'elapsed_seconds':time.perf_counter()-start,'limitations':['one held-out source batch per validation/test split, not five independent cases',
             'all treated as one patient','no continuous video','retrospective split after prior full-dataset fit; new subjects still required']}
    dump(LOG/'results_summary.json',final);dump(OUT/'results_summary.json',final)
    lines=['# Z 阶段主任务：重新初始化训练结果','',
           '本轮以 Z 为主，Q/H 暂存作后续辅助实验。目标为既有多边形内部；没有自动反转前景。','',
           '## 划分与训练','',
           '25 张 Z 图按 5 个来源画面组完整划分：训练 15、验证 5、测试 5；未将组内相似帧拆到不同集合。',
           '来源组不是患者组，按用户说明全部视为同一患者；这是患者内部来源组留出实验。',
           f'模型、优化器从零初始化；未加载旧权重。实际训练 {completed_epochs} 轮，按验证集 Dice 选第 {best_epoch} 轮权重。',
           '训练前固定划分与配置，测试集在选好权重后评价 1 次；未用测试结果修改超参数或阈值。','',
           '## 原图分辨率评价','',
           '| 集合 | 图片 | Dice | IoU | HD95 / 原图像素 |','|---|---:|---:|---:|---:|']
    for s,label in [('train','训练'),('val','验证'),('test','测试')]:
        m=compact[s];hd='不可有限汇总' if m['hd95_original_pixels'] is None else f"{m['hd95_original_pixels']:.2f}"
        lines.append(f"| {label} | {m['count']} | {m['dice']:.4f} | {m['iou']:.4f} | {hd} |")
    lines+=['','网络输入为 512×288 RGB，指标在 3840×2160 原图上逐图计算后平均。HD95 没有毫米标定。','',
            '## 如何理解','',
            '这些分数与此前 50 张训练集拟合的 0.9896 Dice 不属于同一种评估，不能据此直接评价新算法提升或下降。',
            '验证与测试各只有 1 个来源画面组，组内的 5 张相关图片不能视为 5 个独立临床样本；分数不稳定且不能代表跨患者泛化。',
            '这是静态目标分割的基线。动态边界识别、荧光消退后的传播、漂移与器械遮挡恢复仍需连续视频验证。',
            '若后续加入 Q/H，必须先确认对应来源/片段关系，禁止把留出画面组相应的辅助帧放入训练。',
            '此次是已看过全量拟合与可视化之后的回顾性划分；未来应采集未见过的患者与完整消退视频，建立新的外部测试。','',
            '完整记录：split_manifest.json、similarity_audit.json、training_history.csv、results_summary.json；初始化证据保存在模型日志目录。']
    (OUT/'重新训练结果报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    status.update(final,stage='complete');dump(STATE,status)
    print(json.dumps(safe(final),ensure_ascii=False,indent=2))
except BaseException as exc:
    status.update(state='failed',failed_at=now(),error=repr(exc),traceback=traceback.format_exc())
    dump(STATE,status)
    raise

# Historical workflow snapshot for review, not the portable CLI. See ../README.md.
from pathlib import Path
import json, sys, hashlib
from collections import defaultdict, Counter
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'ICG-boundary-detection/model'
OLD=MODEL/'data/preliminary_20261012_01/fullres'
OUT=ROOT/'output/z_holdout_20261019_01'
OUT.mkdir(parents=True,exist_ok=True)
mapping=json.loads((OLD/'conversion.private.json').read_text(encoding='utf-8'))['source_mapping']
parents=sorted({str(Path(r['annotation']).parent) for r in mapping})
groups={p:f'batch_{i+1:02d}' for i,p in enumerate(parents)}
records=[]
for r in mapping:
    records.append({'filename':r['filename'],'phase':r['labels'][0],
                    'source_batch':groups[str(Path(r['annotation']).parent)]})
print(json.dumps({'batches':{g:dict(Counter(r['phase'] for r in records if r['source_batch']==g)) for g in groups.values()}},indent=2))
z=[r for r in records if r['phase']=='Z']
canvas=Image.new('RGB',(2560,5*350+100),'#edf2f7')
d=ImageDraw.Draw(canvas);f=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',24)
d.text((25,20),'Z phase: raw images and anonymous source batches; no model scores used',fill='#173347',font=f)
orb=cv2.ORB_create(nfeatures=1200)
features={};thumbs={};hashes={}
for i,r in enumerate(z):
    p=OLD/'train/images'/r['filename']
    im=Image.open(p).convert('RGB')
    hashes[r['filename']]=hashlib.sha256(im.tobytes()).hexdigest()
    small=np.array(im.resize((512,288),Image.Resampling.LANCZOS))
    gray=cv2.cvtColor(small,cv2.COLOR_RGB2GRAY)
    kp,desc=orb.detectAndCompute(gray,None)
    features[r['filename']]=(kp,desc)
    thumbs[r['filename']]=cv2.resize(gray,(256,144)).astype(np.float32)
    x=(i%5)*512;y=100+(i//5)*350
    canvas.paste(im.resize((500,281),Image.Resampling.LANCZOS),(x,y))
    d.text((x+8,y+290),f"{r['filename'][:-4]} | {r['source_batch']}",fill='#173347',font=f)
canvas.save(OUT/'Z图片_划分前检查.png')

def ssim(a,b):
    ma=cv2.GaussianBlur(a,(11,11),1.5);mb=cv2.GaussianBlur(b,(11,11),1.5)
    va=cv2.GaussianBlur(a*a,(11,11),1.5)-ma*ma
    vb=cv2.GaussianBlur(b*b,(11,11),1.5)-mb*mb
    cov=cv2.GaussianBlur(a*b,(11,11),1.5)-ma*mb
    return float(np.mean(((2*ma*mb+6.5025)*(2*cov+58.5225))/((ma*ma+mb*mb+6.5025)*(va+vb+58.5225))))

pairs=[]
matcher=cv2.BFMatcher(cv2.NORM_HAMMING)
for i,a in enumerate(z):
    for b in z[i+1:]:
        ka,da=features[a['filename']];kb,db=features[b['filename']]
        candidates=matcher.knnMatch(da,db,k=2) if da is not None and db is not None else []
        good=[p[0] for p in candidates if len(p)==2 and p[0].distance<.75*p[1].distance]
        inliers=0
        if len(good)>=8:
            src=np.float32([ka[m.queryIdx].pt for m in good]);dst=np.float32([kb[m.trainIdx].pt for m in good])
            _,mask=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=3)
            if mask is not None:inliers=int(mask.sum())
        pairs.append({'a':a['filename'],'b':b['filename'],'same_source_batch':a['source_batch']==b['source_batch'],
                      'ssim':ssim(thumbs[a['filename']],thumbs[b['filename']]),'orb_matches':len(good),'affine_inliers':inliers,
                      'inlier_fraction':inliers/max(1,min(len(ka),len(kb)))})
audit={'records':records,'z_exact_unique':len(set(hashes.values())),'z_count':len(z),'pairs':pairs,
       'method':'raw appearance only: grayscale SSIM at 256x144; ORB+affine RANSAC at 512x288; source batch grouping is not patient identity'}
(OUT/'similarity_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
for key in ['ssim','affine_inliers']:
    print(key,'cross-batch top',json.dumps(sorted([p for p in pairs if not p['same_source_batch']],key=lambda p:p[key],reverse=True)[:8],indent=2))

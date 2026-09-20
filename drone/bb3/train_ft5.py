"""Fine-tune a detector on recorded validation-city views (frames <= 150) mixed with synthetic views (gpu5 layout).
Like bb3/train_ft.py, with every path an argument so it runs on any pod.
Usage: python train_ft5.py NAME BASE_WEIGHTS SYNTH_DIR VC_DIR REPEAT EPOCHS OUT_PROJECT [LR0] [key=val ...]
  SYNTH_DIR: make_synth.py gen output (data.yaml, images/{train,val}); VC_DIR: build_views.py output.
  Class names come from SYNTH_DIR/data.yaml (the upstream order). The val split is VC_DIR/images/val when it
  exists (temporal fold), else the synthetic val split."""
import glob
import os
import sys

from ultralytics import YOLO

name, base, synth, vc, rep, ep, proj = sys.argv[1:8]
rep, ep = int(rep), int(ep)
lr0 = float(sys.argv[8]) if len(sys.argv) > 8 else 0.002
extra = dict(a.split('=', 1) for a in sys.argv[9:])
for k, v in list(extra.items()):
    try:
        extra[k] = float(v) if '.' in v else int(v)
    except ValueError:
        pass
names = open(f'{synth}/data.yaml').read().split('names:')[1]
val = f'{vc}/images/val' if glob.glob(f'{vc}/images/val/*.png') else f'{synth}/images/val'
os.makedirs(proj, exist_ok=True)
y = f'{proj}/data_{name}.yaml'
with open(y, 'w') as f:
    f.write(f'path: /\ntrain:\n  - {synth}/images/train\n')
    for _ in range(rep):
        f.write(f'  - {vc}/images/train\n')
    f.write(f'val: {val}\nnames:{names}')
kw = dict(data=y, epochs=ep, imgsz=1280, batch=8, lr0=lr0, lrf=0.2, warmup_epochs=1, optimizer='SGD', momentum=0.9,
          project=proj, name=name, exist_ok=True, workers=5, device=0, cos_lr=True,
          mosaic=1.0, close_mosaic=3, degrees=0, fliplr=0.5, flipud=0.5, scale=0.3, hsv_h=0.01, hsv_s=0.4, hsv_v=0.3,
          plots=False, patience=100)
kw.update(extra)   # any key=val on the command line overrides a default (workers, batch, ...)
m = YOLO(base)
m.train(**kw)
print('DONE', f'{proj}/{name}/weights/best.pt')

"""Download generic aerial background tiles (LoveDA, 1024x1024 @0.3 m, urban+rural) for synthesis.
They contain none of our 16 classes, so they are clean negatives (cars, roofs, roads).
    python fetch_bg.py /root/dd/loveda 700          (training: tiles 0..699)
    python fetch_bg.py /root/dd/loveda_scene 40 2522 (synthetic test scenes only)"""
import os, sys
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download
out, n = sys.argv[1], int(sys.argv[2])
start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
os.makedirs(out, exist_ok=True)
def get(i):
    split = "val" if i >= 2522 else "train"
    fn = f"urban:rural {split} images/{i}.png"
    try:
        p = hf_hub_download("chloechia/loveda", fn, repo_type="dataset")
        dst = os.path.join(out, f"loveda_{split}_{i}.png")
        if not os.path.exists(dst):
            os.symlink(p, dst)
        return 1
    except Exception as e:
        return 0
with ThreadPoolExecutor(8) as ex:
    print("ok", sum(ex.map(get, range(start, start + n))))

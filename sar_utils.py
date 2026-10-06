"""Data: read the selected classes, resize to 224x224 grayscale uint8, cache per split."""
import os, numpy as np, torch
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from config import DATASETS, CACHE_DIR

for _a, _t in [('float', float), ('int', int), ('object', object), ('bool', bool)]:
    if not hasattr(np, _a):   # only add missing aliases (old repo code); never overwrite numpy's own types
        setattr(np, _a, _t)

IMG_SIZE = 224

def _read(path):
    img = Image.open(path).convert('L').resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    return np.asarray(img, dtype=np.uint8)

def _load_split(name, split):
    root, classes = DATASETS[name]['root'], DATASETS[name]['classes']
    cache = os.path.join(CACHE_DIR, f'{name}_{split}.pt')
    if os.path.exists(cache):
        d = torch.load(cache, weights_only=True)
        if d['classes'] == classes:
            return d['x'], d['y']
    paths, labels = [], []
    for i, c in enumerate(classes):
        folder = os.path.join(root, split, c)
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg')))
        paths += [os.path.join(folder, f) for f in files]
        labels += [i] * len(files)
    with ThreadPoolExecutor(8) as ex:
        imgs = list(ex.map(_read, paths))
    x = torch.from_numpy(np.stack(imgs)).unsqueeze(1)
    y = torch.tensor(labels, dtype=torch.long)
    tmp = cache + '.tmp'; torch.save({'x': x, 'y': y, 'classes': classes}, tmp); os.replace(tmp, cache)
    return x, y

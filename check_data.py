"""Verifies the cluster data matches the counts used so far, then builds the 224x224 cache."""
import os
from config import DATASETS
import sar_utils as U
EXPECTED = {
    'opensarship': {'train': [8484, 2920, 227, 222], 'val': [1060, 365, 28, 27], 'test': [1062, 365, 29, 29]},
    'fusarship':   {'train': [1354, 628, 218, 118],  'val': [169, 78, 27, 14],   'test': [170, 79, 28, 16]},
}
ok = True
for name, cfg in DATASETS.items():
    for split, exp in EXPECTED[name].items():
        got = [len([f for f in os.listdir(os.path.join(cfg['root'], split, c))
                    if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg'))]) for c in cfg['classes']]
        flag = 'OK' if got == exp else 'MISMATCH'
        ok &= got == exp
        print(f"{name:<12} {split:<5} {got}  expected {exp}  {flag}")
if not ok:
    raise SystemExit("Counts differ from the Colab data. Stop and check before training.")
for name in DATASETS:
    for split in ['train', 'val', 'test']:
        x, y = U._load_split(name, split)
        print(f"cached {name}/{split}: {tuple(x.shape)}")
print("Data OK.")

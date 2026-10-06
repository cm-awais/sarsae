"""LoRA learning-rate check: same runs at different lr_lora, kept apart from the main results."""
import sys, os
import gpu_train as G
from config import WORK
LRS = [1e-4, 5e-4, 1e-3]
lr = LRS[int(sys.argv[1])]
G.HP['lr_lora'] = lr
G.OUT = f'{WORK}/runs_lrcheck/lr{lr:g}'; os.makedirs(G.OUT, exist_ok=True)
for d in ['fusarship', 'opensarship']:
    for m in ['sarjepa', 'dinov2_b']:
        G.run(m, d, 'c06_lora_wce', 0)
print(f"lr {lr:g} done")
"""Loads every model on the GPU, checks dims, LoRA, init equality and one training step on real images."""
import torch
import sar_utils as U
from smoke_test import smoke_test
x_all, y_all = U._load_split('opensarship', 'test')
idx = torch.cat([(y_all == c).nonzero()[:2, 0] for c in range(4)])
names = ['sarjepa', 'saratrx', 'dofa_b', 'dinov2_b', 'clip_b', 'dinov3_b_web', 'dinov3_l_web', 'dinov3_l_sat']
rep = smoke_test(names, x_all[idx].float() / 255.0, y_all[idx], 4, 'cuda')
assert all(r['status'] == 'OK' for r in rep), "smoke test failed"
print("SMOKE TEST PASSED")

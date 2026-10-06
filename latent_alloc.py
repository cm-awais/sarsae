"""Step 5: latent allocation. Does the SAE give tail classes their own latents?

For each SAE (configs 3, 7, 8 staged; 9, 10 joint), on the TRAIN set:
  - a latent is alive if it fires at least once
  - its class = the class with the highest MEAN activation (size-neutral: a small class can win)
  - A_c = share of alive latents assigned to class c
  - reference: with equal allocation every class gets 1/K, so the 2 tail classes together get 0.5
Config 4 uses the same SAE as config 3, so it is not repeated.
Joint configs (9, 10) need a forward pass through the LoRA backbone -> GPU.
Writes work/analysis/latent_alloc.csv and a summary.
"""
import os, json, argparse, torch, torch.nn.functional as F, numpy as np, pandas as pd
import sar_utils as U
import fm_registry as R
import gpu_train as G
from sae_staged import load_split, OUT as STAGED
from gpu_train import TopKSAE, HP
from config import WORK, RUNS_DIR, DATASETS

OUT = f'{WORK}/analysis'; os.makedirs(OUT, exist_ok=True)
MODELS = ['sarjepa', 'saratrx', 'dofa_b', 'dinov2_b', 'clip_b', 'dinov3_b_web', 'dinov3_l_web', 'dinov3_l_sat']
STAGED_CFG = {'c03_sae_ce': 'frozen', 'c07_lora_sae_ce': 'c05_lora_ce', 'c08_lora_sae_wce': 'c06_lora_wce'}
JOINT_CFG = ['c09_joint_learned_ce', 'c10_joint_learned_wce']
TAIL = {'opensarship': ['Dredging', 'Fishing'], 'fusarship': ['Bulk', 'Tanker']}

def allocation(z, y, classes):
    z, y = z.float().cpu(), y.cpu()
    alive = (z > 0).any(0)
    means = torch.stack([z[y == k].mean(0) for k in range(len(classes))])     # (K, m) mean activation per class
    owner = means[:, alive].argmax(0)
    n_alive = int(alive.sum())
    counts = torch.bincount(owner, minlength=len(classes)).float()
    freq = torch.bincount(y, minlength=len(classes)).float()
    return n_alive, (counts / max(n_alive, 1)).tolist(), (freq / freq.sum()).tolist()

def staged_codes(model, dataset, config, seed, device):
    rid = f'{model}__{dataset}__{config}__s{seed}'
    p = os.path.join(STAGED, f'{rid}_sae.pt')
    if not os.path.exists(p): return None
    x, y = load_split(model, dataset, STAGED_CFG[config], seed, 'train')
    sd = torch.load(p, map_location=device, weights_only=True)
    d, m = sd['enc.weight'].shape[1], sd['enc.weight'].shape[0]
    sae = TopKSAE(d, m, HP['topk']).to(device); sae.load_state_dict(sd); sae.eval()
    with torch.no_grad():
        return sae.encode(x.to(device)), y

@torch.no_grad()
def joint_codes(model, dataset, config, seed, device, bs=128):
    rdir = os.path.join(RUNS_DIR, f'{model}__{dataset}__{config}__s{seed}')
    if not os.path.exists(os.path.join(rdir, 'result.json')): return None
    K = len(DATASETS[dataset]['classes'])
    net = G.Net(R.load_backbone(model, device), K, joint=True).to(device)
    net.load_state_dict(torch.load(os.path.join(rdir, 'best.pt'), map_location=device, weights_only=True), strict=False)
    net.eval()
    x, y = U._load_split(dataset, 'train'); zs = []
    for i in range(0, len(y), bs):
        xb = x[i:i + bs].to(device).float() / 255.0
        with torch.autocast('cuda', dtype=torch.float16, enabled=device == 'cuda'):
            e = net.backbone(xb).float()
        zs.append(net.sae.encode(F.normalize(e, dim=-1)).cpu())
    del net; torch.cuda.empty_cache() if device == 'cuda' else None
    return torch.cat(zs), y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=MODELS)
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    path = f'{OUT}/latent_alloc.csv'
    done = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=['run_id'])
    rows = done.to_dict('records')
    for s in a.seeds:
        for d, cfg in DATASETS.items():
            for m in a.models:
                for c in list(STAGED_CFG) + JOINT_CFG:
                    rid = f'{m}__{d}__{c}__s{s}'
                    if rid in set(done.run_id): continue
                    out = staged_codes(m, d, c, s, a.device) if c in STAGED_CFG else joint_codes(m, d, c, s, a.device)
                    if out is None:
                        print(f'{rid:<45} missing, skipped', flush=True); continue
                    n_alive, share, freq = allocation(*out, cfg['classes'])
                    tail_idx = [cfg['classes'].index(t) for t in TAIL[d]]
                    tail_share, tail_freq = sum(share[i] for i in tail_idx), sum(freq[i] for i in tail_idx)
                    uniform = len(tail_idx) / len(cfg['classes'])          # share if every class got equal latents
                    row = {'run_id': rid, 'model': m, 'dataset': d, 'config': c, 'seed': s, 'alive_latents': n_alive,
                           'total_latents': out[0].shape[1], 'tail_latent_share': tail_share, 'tail_data_share': tail_freq,
                           'tail_equal_share': uniform, 'tail_vs_equal': tail_share / uniform,
                           **{f'latents_{k}': v for k, v in zip(cfg['classes'], share)},
                           **{f'data_{k}': v for k, v in zip(cfg['classes'], freq)}}
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(path, index=False)          # saved after every run: resumable
                    print(f"{rid:<45} alive {n_alive:>5} | tail latents {tail_share:.2f} | equal share {uniform:.2f} | "
                          f"tail data {tail_freq:.2f}", flush=True)
    df = pd.DataFrame(rows)
    pd.set_option('display.width', 200)
    print("\nTail latent share (mean over models and seeds). tail_vs_equal = 1: tail classes get as many latents "
          "as head classes; < 1: tail under-allocated; tail_data_share = tail share of training images, for reference:")
    t = df.groupby(['dataset', 'config'])[['alive_latents', 'tail_data_share', 'tail_equal_share', 'tail_latent_share',
                                           'tail_vs_equal']].mean().round(3)
    print(t.to_string()); t.to_csv(f'{OUT}/latent_alloc_summary.csv')

if __name__ == '__main__':
    main()
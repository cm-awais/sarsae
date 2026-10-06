"""Configs 3, 4, 7, 8: staged SAE. (1) TopK SAE trained without labels on embeddings, (2) linear head on the codes.

  c03_sae_ce       frozen embeddings          head: CE
  c04_sae_wce      frozen embeddings          head: weighted CE
  c07_lora_sae_ce  LoRA embeddings from c05   head: CE
  c08_lora_sae_wce LoRA embeddings from c06   head: weighted CE

The SAE is unsupervised, so configs 3 and 4 share one SAE per (model, dataset, seed).
Results: work/runs_staged/<run_id>.json (+ sae.pt, head.pt, test logits), table in work/results_staged.csv
"""
import os, json, argparse, copy, torch, torch.nn as nn, torch.nn.functional as F, numpy as np
import embeddings as E
import sar_utils as U
from gpu_train import TopKSAE, HP, metrics, _atomic_json
from config import WORK, RUNS_DIR

OUT = f'{WORK}/runs_staged'
HP_SAE  = dict(lr=1e-3, batch_size=256, max_epochs=100, patience=25, sae_mult=HP['sae_mult'], topk=HP['topk'])
HP_HEAD = dict(lr=1e-2, weight_decay=0.01, batch_size=256, max_epochs=300, patience=50, grad_clip=1.0)  # = frozen.py
CONFIGS_S = {  # name: (embedding source, weighted CE?)
    'c03_sae_ce':       ('frozen', False),
    'c04_sae_wce':      ('frozen', True),
    'c07_lora_sae_ce':  ('c05_lora_ce', False),
    'c08_lora_sae_wce': ('c06_lora_wce', True),
}
MODELS = ['sarjepa', 'saratrx', 'dofa_b', 'dinov2_b', 'clip_b', 'dinov3_b_web', 'dinov3_l_web', 'dinov3_l_sat']

def load_split(model, dataset, source, seed, split):
    if source == 'frozen':
        e, y = E.load_embeddings(model, dataset, split)
    else:
        d = torch.load(os.path.join(RUNS_DIR, f'{model}__{dataset}__{source}__s{seed}', f'emb_{split}.pt'),
                       weights_only=True)
        e, y = d['e'].float(), d['y']
    return F.normalize(e, dim=-1), y          # unit norm, as in the joint configs

def _rec(sae, x):
    return ((sae(x)[1] - x) ** 2).sum(-1).mean().item()

def train_sae(xtr, xva, seed, device):
    """Unsupervised. Early stopping on val reconstruction error."""
    torch.manual_seed(seed)
    d = xtr.shape[1]
    sae = TopKSAE(d, HP_SAE['sae_mult'] * d, HP_SAE['topk']).to(device)
    opt = torch.optim.AdamW(sae.parameters(), lr=HP_SAE['lr'], weight_decay=0.0)
    best, best_ep, bad, best_state, hist = float('inf'), -1, 0, None, []
    for ep in range(HP_SAE['max_epochs']):
        perm = torch.randperm(len(xtr), generator=torch.Generator().manual_seed(seed * 1000 + ep)).to(device)
        sae.train()
        for i in range(0, len(perm), HP_SAE['batch_size']):
            xb = xtr[perm[i:i + HP_SAE['batch_size']]]
            loss = ((sae(xb)[1] - xb) ** 2).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sae.renorm()
        sae.eval()
        with torch.no_grad():
            vr = _rec(sae, xva)
        hist.append({'epoch': ep, 'val_rec': vr})
        if vr < best:
            best, best_ep, bad, best_state = vr, ep, 0, copy.deepcopy(sae.state_dict())
        else:
            bad += 1
            if bad >= HP_SAE['patience']:
                break
    sae.load_state_dict(best_state); sae.eval()
    return sae, {'sae_best_epoch': best_ep, 'sae_epochs_run': len(hist), 'sae_val_rec': best, 'sae_history': hist}

def train_head(ztr, ytr, zva, yva, weighted, classes, seed, device):
    K = len(classes)
    cw = None
    if weighted:
        counts = torch.bincount(ytr, minlength=K).float()
        cw = counts.sum() / (K * counts)
    torch.manual_seed(seed)
    head = nn.Linear(ztr.shape[1], K).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=HP_HEAD['lr'], weight_decay=HP_HEAD['weight_decay'])
    best, best_ep, bad, best_state, hist = -1.0, -1, 0, None, []
    for ep in range(HP_HEAD['max_epochs']):
        perm = torch.randperm(len(ytr), generator=torch.Generator().manual_seed(seed * 1000 + ep)).to(device)
        head.train()
        for i in range(0, len(perm), HP_HEAD['batch_size']):
            j = perm[i:i + HP_HEAD['batch_size']]
            loss = F.cross_entropy(head(ztr[j]), ytr[j], weight=cw)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), HP_HEAD['grad_clip']); opt.step()
        head.eval()
        with torch.no_grad():
            tf1 = metrics(ytr.cpu().numpy(), head(ztr).argmax(1).cpu().numpy(), classes)['macro_f1']
            vf1 = metrics(yva.cpu().numpy(), head(zva).argmax(1).cpu().numpy(), classes)['macro_f1']
        hist.append({'epoch': ep, 'train_macro_f1': tf1, 'val_macro_f1': vf1})
        if vf1 > best:
            best, best_ep, bad, best_state = vf1, ep, 0, copy.deepcopy(head.state_dict())
        else:
            bad += 1
            if bad >= HP_HEAD['patience']:
                break
    head.load_state_dict(best_state); head.eval()
    return head, {'val_macro_f1': best, 'train_macro_f1_at_best': hist[best_ep]['train_macro_f1'],
                  'best_epoch': best_ep, 'epochs_run': len(hist), 'history': hist}

def run_model_dataset_seed(model, dataset, seed, device):
    """All four staged configs for one (model, dataset, seed); reuses the frozen SAE for c03 and c04."""
    classes = U.DATASETS[dataset]['classes']
    sae_cache = {}
    for config, (source, weighted) in CONFIGS_S.items():
        rid = f'{model}__{dataset}__{config}__s{seed}'
        path = os.path.join(OUT, f'{rid}.json')
        if os.path.exists(path):
            print(f'{rid:<45} done, skipped', flush=True); continue
        if source != 'frozen' and not os.path.exists(
                os.path.join(RUNS_DIR, f'{model}__{dataset}__{source}__s{seed}', 'emb_test.pt')):
            print(f'{rid:<45} MISSING source run {source}, skipped', flush=True); continue
        (xtr, ytr), (xva, yva), (xte, yte) = [[t.to(device) for t in load_split(model, dataset, source, seed, s)]
                                              for s in ['train', 'val', 'test']]
        if source not in sae_cache:
            sae_cache[source] = train_sae(xtr, xva, seed, device)
        sae, sae_info = sae_cache[source]
        with torch.no_grad():
            ztr, zva, zte = sae.encode(xtr), sae.encode(xva), sae.encode(xte)
        head, head_info = train_head(ztr, ytr, zva, yva, weighted, classes, seed, device)
        with torch.no_grad():
            logits = head(zte).cpu()
        yt, pred = yte.cpu().numpy(), logits.argmax(1).numpy()
        res = {'run_id': rid, 'model': model, 'dataset': dataset, 'config': config, 'seed': seed,
               'embedding_source': source, **{f'sae_{k}': v for k, v in HP_SAE.items()},
               **{f'head_{k}': v for k, v in HP_HEAD.items()},
               **{k: v for k, v in head_info.items() if k != 'history'},
               **{k: v for k, v in sae_info.items() if k != 'sae_history'},
               **{f'test_{k}': v for k, v in metrics(yt, pred, classes).items()},
               'test_rec_error': _rec(sae, xte), 'train_rec_error': _rec(sae, xtr),
               'dead_latent_frac': float(1 - (zte > 0).any(0).float().mean()),
               'dead_latent_frac_train': float(1 - (ztr > 0).any(0).float().mean()),
               'head_history': head_info['history'], 'sae_history': sae_info['sae_history']}
        torch.save(sae.state_dict(), os.path.join(OUT, f'{rid}_sae.pt'))
        torch.save(head.state_dict(), os.path.join(OUT, f'{rid}_head.pt'))
        np.save(os.path.join(OUT, f'{rid}_test_logits.npy'), logits.numpy())
        _atomic_json(res, path)
        print(f"{rid:<45} test F1 {res['test_macro_f1']:.4f} | val {res['val_macro_f1']:.4f} | "
              f"train {res['train_macro_f1_at_best']:.4f} | rec {res['test_rec_error']:.3f} | "
              f"dead {res['dead_latent_frac']:.2f}", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=MODELS)
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for s in a.seeds:
        for d in ['fusarship', 'opensarship']:
            for m in a.models:
                run_model_dataset_seed(m, d, s, a.device)
    import pandas as pd
    rows = [json.load(open(os.path.join(OUT, f))) for f in sorted(os.listdir(OUT)) if f.endswith('.json')]
    df = pd.DataFrame(rows).drop(columns=['head_history', 'sae_history'], errors='ignore')
    df.to_csv(f'{WORK}/results_staged.csv', index=False)
    pd.set_option('display.width', 200)
    print("\nTest macro-F1, mean ± std over seeds:")
    print(df.groupby(['dataset', 'model', 'config'])['test_macro_f1'].agg(['mean', 'std', 'count']).round(4).to_string())

if __name__ == '__main__':
    main()
"""Configs 1 and 2: linear head on frozen embeddings (CE vs weighted CE). Seconds per run.

Results: work/runs_frozen/<run_id>.json (+ test logits), table in work/results_frozen.csv
"""
import os, json, argparse, copy, torch, torch.nn as nn, torch.nn.functional as F, numpy as np
import embeddings as E
import sar_utils as U
from gpu_train import metrics, _atomic_json
from config import WORK

OUT = f'{WORK}/runs_frozen'
# HPF = dict(lr=1e-3, weight_decay=0.01, batch_size=256, max_epochs=100, patience=25, grad_clip=1.0)
HPF = dict(lr=1e-2, weight_decay=0.01, batch_size=256, max_epochs=300, patience=50, grad_clip=1.0)
CONFIGS_F = {'c01_frozen_ce': False, 'c02_frozen_wce': True}
MODELS = ['sarjepa', 'saratrx', 'dofa_b', 'dinov2_b', 'clip_b', 'dinov3_b_web', 'dinov3_l_web', 'dinov3_l_sat']

@torch.no_grad()
def _f1(head, e, y, classes):
    return metrics(y.cpu().numpy(), head(e).argmax(1).cpu().numpy(), classes)['macro_f1']

def run_head(model, dataset, config, seed, device='cuda'):
    rid = f'{model}__{dataset}__{config}__s{seed}'
    path = os.path.join(OUT, f'{rid}.json')
    if os.path.exists(path):
        return json.load(open(path))
    classes = U.DATASETS[dataset]['classes']; K = len(classes)
    etr, ytr = [t.to(device) for t in E.load_embeddings(model, dataset, 'train')]
    eva, yva = [t.to(device) for t in E.load_embeddings(model, dataset, 'val')]
    ete, yte = [t.to(device) for t in E.load_embeddings(model, dataset, 'test')]
    cw = None
    if CONFIGS_F[config]:
        counts = torch.bincount(ytr, minlength=K).float()
        cw = counts.sum() / (K * counts)
    torch.manual_seed(seed)
    head = nn.Linear(etr.shape[1], K).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=HPF['lr'], weight_decay=HPF['weight_decay'])
    best, best_ep, bad, best_state, hist = -1.0, -1, 0, None, []
    for ep in range(HPF['max_epochs']):
        perm = torch.randperm(len(ytr), generator=torch.Generator().manual_seed(seed * 1000 + ep)).to(device)
        head.train()
        for i in range(0, len(perm), HPF['batch_size']):
            j = perm[i:i + HPF['batch_size']]
            loss = F.cross_entropy(head(etr[j]), ytr[j], weight=cw)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), HPF['grad_clip']); opt.step()
        head.eval()
        tf1, vf1 = _f1(head, etr, ytr, classes), _f1(head, eva, yva, classes)
        hist.append({'epoch': ep, 'train_macro_f1': tf1, 'val_macro_f1': vf1})
        if vf1 > best:
            best, best_ep, bad, best_state = vf1, ep, 0, copy.deepcopy(head.state_dict())
        else:
            bad += 1
            if bad >= HPF['patience']:
                break
    head.load_state_dict(best_state); head.eval()
    with torch.no_grad():
        logits = head(ete).cpu()
    np.save(os.path.join(OUT, f'{rid}_test_logits.npy'), logits.numpy())
    yt, pred = yte.cpu().numpy(), logits.argmax(1).numpy()
    res = {'run_id': rid, 'model': model, 'dataset': dataset, 'config': config, 'seed': seed,
           'val_macro_f1': best, 'train_macro_f1_at_best': hist[best_ep]['train_macro_f1'],
           'best_epoch': best_ep, 'epochs_run': len(hist), **HPF,
           **{f'test_{k}': v for k, v in metrics(yt, pred, classes).items()},
           'history': hist}
    torch.save(best_state, os.path.join(OUT, f'{rid}_head.pt'))
    _atomic_json(res, path)
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=MODELS)
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    import pandas as pd
    rows = []
    for m in a.models:
        for d in ['fusarship', 'opensarship']:
            for c in CONFIGS_F:
                for s in a.seeds:
                    r = run_head(m, d, c, s, a.device); rows.append(r)
                    print(f"{r['run_id']:<45} test F1 {r['test_macro_f1']:.4f} | val {r['val_macro_f1']:.4f} | "
                          f"train {r['train_macro_f1_at_best']:.4f} | best ep {r['best_epoch']}", flush=True)
    df = pd.DataFrame(rows).drop(columns=['history'])
    df.to_csv(f'{WORK}/results_frozen.csv', index=False)
    pd.set_option('display.width', 200)
    print("\nTest macro-F1, mean ± std over seeds:")
    t = df.groupby(['dataset', 'model', 'config'])['test_macro_f1'].agg(['mean', 'std']).round(4)
    print(t.to_string())

if __name__ == '__main__':
    main()
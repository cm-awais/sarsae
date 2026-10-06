"""GPU training for configs 5, 6 (LoRA + head) and 9, 10 (joint LoRA + SAE + head, learned loss weights).

Every run is resumable: a checkpoint is written to Drive after each epoch, and finished runs are skipped.
Configs 5 and 6 also save their LoRA embeddings, so configs 7 and 8 can later run on CPU.
"""
import os, gc, json, time, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from peft import LoraConfig, get_peft_model
from sklearn.metrics import precision_recall_fscore_support, balanced_accuracy_score, confusion_matrix
import sar_utils as U
import fm_registry as R

from config import RUNS_DIR as OUT, WORK
CONFIGS = {  # name: (joint SAE?, weighted CE?)
    'c05_lora_ce':          (False, False),
    'c06_lora_wce':         (False, True),
    'c09_joint_learned_ce': (True,  False),
    'c10_joint_learned_wce':(True,  True),
}
HP = dict(lora_r=8, lora_alpha=16, lora_dropout=0.1, lr_lora=1e-4, lr_other=1e-3, weight_decay=0.01,
          max_epochs=50, patience=10, grad_clip=1.0, sae_mult=4, topk=32, batch_size=64)

# ---------------- utilities ----------------
def _atomic_save(obj, path):
    tmp = path + '.tmp'; torch.save(obj, tmp); os.replace(tmp, path)

def _atomic_json(obj, path):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f: json.dump(obj, f, indent=1)
    os.replace(tmp, path)

def metrics(y_true, y_pred, classes):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(classes))), zero_division=0)
    out = {'macro_f1': float(f.mean()), 'balanced_acc': float(balanced_accuracy_score(y_true, y_pred)),
           'accuracy': float((np.asarray(y_true) == np.asarray(y_pred)).mean())}
    for i, c in enumerate(classes):
        out[f'f1_{c}'], out[f'precision_{c}'], out[f'recall_{c}'] = float(f[i]), float(p[i]), float(r[i])
    return out

# ---------------- model ----------------
class TopKSAE(nn.Module):
    """z = TopK(ReLU(W_enc (x - b_dec) + b_enc)),  x_hat = W_dec z + b_dec.  Input x is unit-normalized."""
    def __init__(self, d, m, k):
        super().__init__()
        self.k, self.b_dec = k, nn.Parameter(torch.zeros(d))
        self.enc, self.dec = nn.Linear(d, m), nn.Linear(m, d, bias=False)
        with torch.no_grad():
            self.dec.weight.copy_(F.normalize(torch.randn(d, m), dim=0))
            self.enc.weight.copy_(self.dec.weight.t()); self.enc.bias.zero_()
    def encode(self, x):
        a = F.relu(self.enc(x - self.b_dec))
        v, i = a.topk(self.k, dim=-1)
        return torch.zeros_like(a).scatter_(-1, i, v)
    def forward(self, x):
        z = self.encode(x)
        return z, self.dec(z) + self.b_dec
    @torch.no_grad()
    def renorm(self):                      # keep decoder columns unit-norm
        self.dec.weight.copy_(F.normalize(self.dec.weight, dim=0))

class Net(nn.Module):
    def __init__(self, backbone, num_classes, joint):
        super().__init__()
        d = backbone.spec['dim']
        self.backbone = get_peft_model(backbone, LoraConfig(
            r=HP['lora_r'], lora_alpha=HP['lora_alpha'], lora_dropout=HP['lora_dropout'],
            target_modules=backbone.spec['lora_targets'], bias='none'))
        self.joint = joint
        if joint:
            self.sae = TopKSAE(d, HP['sae_mult'] * d, HP['topk'])
            self.log_var = nn.Parameter(torch.zeros(2))   # s_i = log(sigma_i^2) for [classification, reconstruction]
            self.head = nn.Linear(HP['sae_mult'] * d, num_classes)
        else:
            self.head = nn.Linear(d, num_classes)
    def forward(self, x):
        e = self.backbone(x).float()
        if not self.joint:
            return self.head(e), None, None
        e_bar = F.normalize(e, dim=-1)
        z, e_hat = self.sae(e_bar)
        rec = ((e_hat - e_bar) ** 2).sum(-1).mean()       # normalized reconstruction error
        return self.head(z), rec, z

def joint_loss(l_cls, l_rec, log_var):
    # L = L_cls/(2 s1^2) + L_rec/(2 s2^2) + log s1 + log s2, written with s_i = log sigma_i^2
    return 0.5 * torch.exp(-log_var[0]) * l_cls + 0.5 * torch.exp(-log_var[1]) * l_rec + 0.5 * log_var.sum()

def trainable_state(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}

# ---------------- data on GPU ----------------
def _gpu_split(dataset, split, device):
    x, y = U._load_split(dataset, split)
    return x.to(device), y.to(device)

def _batches(x, y, bs, perm=None):
    idx = perm if perm is not None else torch.arange(len(y), device=y.device)
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        yield x[j].float() / 255.0, y[j]

@torch.no_grad()
def _predict(model, x, y, bs, amp):
    model.eval(); logits, zs = [], []
    for xb, _ in _batches(x, y, bs):
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            out, _, z = model(xb)
        logits.append(out.float().cpu())
        if z is not None: zs.append((z > 0).any(0).cpu())
    active = torch.stack(zs).any(0) if zs else None
    return torch.cat(logits), active

# ---------------- one run ----------------
def run(model_name, dataset, config, seed, device='cuda', bs=None, max_epochs=None, verbose=True):
    run_id = f'{model_name}__{dataset}__{config}__s{seed}'
    rdir = os.path.join(OUT, run_id); os.makedirs(rdir, exist_ok=True)
    if os.path.exists(f'{rdir}/result.json'):
        if verbose: print(f'[{run_id}] done, skipped')
        return json.load(open(f'{rdir}/result.json'))
    joint, weighted = CONFIGS[config]
    max_epochs = max_epochs or HP['max_epochs']
    amp = device == 'cuda'
    classes = U.DATASETS[dataset]['classes']

    xtr, ytr = _gpu_split(dataset, 'train', device)
    xva, yva = _gpu_split(dataset, 'val', device)
    counts = torch.bincount(ytr.cpu(), minlength=len(classes)).float()
    cw = (counts.sum() / (len(classes) * counts)).to(device) if weighted else None

    torch.manual_seed(seed)
    model = Net(R.load_backbone(model_name, device), len(classes), joint).to(device)
    bs = bs or HP['batch_size']
    lora_p = [p for n, p in model.named_parameters() if p.requires_grad and 'lora_' in n]
    other_p = [p for n, p in model.named_parameters() if p.requires_grad and 'lora_' not in n and n != 'log_var']
    groups = [{'params': lora_p, 'lr': HP['lr_lora']}, {'params': other_p, 'lr': HP['lr_other']}]
    if joint: groups.append({'params': [model.log_var], 'lr': HP['lr_other'], 'weight_decay': 0.0})
    opt = torch.optim.AdamW(groups, weight_decay=HP['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=amp)

    state = dict(epoch=0, best_f1=-1.0, best_epoch=-1, bad=0, history=[], train_time=0.0)
    ck = f'{rdir}/last.pt'
    if os.path.exists(ck):                                           # resume
        c = torch.load(ck, map_location=device, weights_only=False)
        model.load_state_dict(c['model'], strict=False); opt.load_state_dict(c['opt'])
        scaler.load_state_dict(c['scaler']); state = c['state']
        if verbose: print(f'[{run_id}] resumed at epoch {state["epoch"]}')

    while state['epoch'] < max_epochs and state['bad'] < HP['patience']:
        ep, t0 = state['epoch'], time.time()
        torch.manual_seed(seed * 1000 + ep)                          # deterministic per epoch, also after resume
        perm = torch.randperm(len(ytr), generator=torch.Generator().manual_seed(seed * 1000 + ep)).to(device)
        model.train(); tot = n = 0; tr_pred, tr_true = [], []
        for xb, yb in _batches(xtr, ytr, bs, perm):
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                logits, rec, _ = model(xb)
            tr_pred.append(logits.detach().argmax(1).cpu()); tr_true.append(yb.cpu())
            l_cls = F.cross_entropy(logits.float(), yb, weight=cw)
            loss = joint_loss(l_cls, rec, model.log_var) if joint else l_cls
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g['params']], HP['grad_clip'])
            scaler.step(opt); scaler.update()
            if joint: model.sae.renorm()
            tot += loss.item() * len(yb); n += len(yb)
        vlog, _ = _predict(model, xva, yva, 2 * bs, amp)
        vf1 = metrics(yva.cpu().numpy(), vlog.argmax(1).numpy(), classes)['macro_f1']
        tf1 = metrics(torch.cat(tr_true).numpy(), torch.cat(tr_pred).numpy(), classes)['macro_f1']
        state['train_time'] += time.time() - t0
        state['history'].append({'epoch': ep, 'loss': tot / n, 'train_macro_f1': tf1, 'val_macro_f1': vf1,
                                 'sec': time.time() - t0})
        if vf1 > state['best_f1']:
            state.update(best_f1=vf1, best_epoch=ep, bad=0)
            _atomic_save(trainable_state(model), f'{rdir}/best.pt')
        else:
            state['bad'] += 1
        state['epoch'] = ep + 1
        _atomic_save({'model': trainable_state(model), 'opt': opt.state_dict(),
                      'scaler': scaler.state_dict(), 'state': state}, ck)
        if verbose:
            print(f'[{run_id}] ep {ep:3d} | loss {tot / n:.4f} | train F1 {tf1:.4f} | val F1 {vf1:.4f} | best {state["best_f1"]:.4f} '
                  f'@{state["best_epoch"]} | {time.time() - t0:.0f}s')

    # ---- final evaluation with the best weights ----
    model.load_state_dict(torch.load(f'{rdir}/best.pt', map_location=device, weights_only=True), strict=False)
    del xtr, ytr
    xte, yte = _gpu_split(dataset, 'test', device)
    tlog, active = _predict(model, xte, yte, 2 * bs, amp)
    pred = tlog.argmax(1).numpy(); yt = yte.cpu().numpy()
    np.save(f'{rdir}/test_logits.npy', tlog.numpy()); np.save(f'{rdir}/test_labels.npy', yt)
    res = {'run_id': run_id, 'model': model_name, 'dataset': dataset, 'config': config, 'seed': seed,
           'val_macro_f1': state['best_f1'], 'best_epoch': state['best_epoch'], 'epochs_run': state['epoch'],
           'train_time_sec': round(state['train_time'], 1), 'batch_size': bs, **HP,
           **{f'test_{k}': v for k, v in metrics(yt, pred, classes).items()},
                      'confusion_matrix': confusion_matrix(yt, pred, labels=list(range(len(classes)))).tolist(),
           'history': state['history']}
    if joint:
        res['dead_latent_frac'] = float(1 - active.float().mean())
        res['sigma_cls'], res['sigma_rec'] = [float(v) for v in torch.exp(0.5 * model.log_var.detach().cpu())]
        model.eval(); recs = []
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            for xb, _ in _batches(xte, yte, 2 * bs):
                recs.append(model(xb)[1].float().item() * len(xb))
        res['test_rec_error'] = sum(recs) / len(yte)
    if not joint:                                                    # LoRA embeddings for configs 7 and 8
        model.eval()
        for split in ['train', 'val', 'test']:
            x, y = _gpu_split(dataset, split, device); es = []
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                for xb, _ in _batches(x, y, 2 * bs):
                    es.append(model.backbone(xb).float().cpu())
            _atomic_save({'e': torch.cat(es).half(), 'y': y.cpu()}, f'{rdir}/emb_{split}.pt')
            del x, y
    _atomic_json(res, f'{rdir}/result.json')
    if os.path.exists(ck): os.remove(ck)                             # optimizer state no longer needed
    print(f'[{run_id}] TEST macro-F1 {res["test_macro_f1"]:.4f} | best epoch {state["best_epoch"]} | '
          f'{state["train_time"] / 60:.1f} min')
    del model, opt; gc.collect()
    if device == 'cuda': torch.cuda.empty_cache()
    return res

def run_queue(models, datasets, configs, seeds, device='cuda', **kw):
    for s in seeds:
        for m in models:
            for d in datasets:
                for c in configs:
                    try:
                        run(m, d, c, s, device=device, **kw)
                    except Exception as ex:
                        print(f'FAILED {m}/{d}/{c}/s{s}: {type(ex).__name__}: {ex}')
                        gc.collect(); torch.cuda.empty_cache() if device == 'cuda' else None

def collect_results():
    import pandas as pd
    rows = [json.load(open(os.path.join(OUT, r, 'result.json'))) for r in sorted(os.listdir(OUT))
            if os.path.exists(os.path.join(OUT, r, 'result.json'))]
    df = pd.DataFrame(rows)
    if len(df): df.drop(columns=['confusion_matrix', 'history'], errors='ignore').to_csv(f'{WORK}/results_gpu.csv', index=False)
    return df

"""Step 1: extract frozen embeddings once per (model, dataset, split) and save them to Drive.

Saved file: {EMB_DIR}/{model}_{dataset}_{split}.pt with keys
    e (N, dim) float32, y (N,) long, model, dataset, split, dim, classes
Existing files are skipped, so the job can be stopped and restarted at any time.
"""
import os, gc, time, torch
import sar_utils as U
import fm_registry as R

from config import EMB_DIR

def emb_path(model, dataset, split):
    return os.path.join(EMB_DIR, f'{model}_{dataset}_{split}.pt')

def _atomic_save(obj, path):
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)          # a crash never leaves a half-written file

@torch.no_grad()
def _encode(bb, x_uint8, device, batch_size, amp):
    out = []
    for i in range(0, len(x_uint8), batch_size):
        x = x_uint8[i:i + batch_size].to(device).float() / 255.0
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            e = bb(x)
        out.append(e.float().cpu())
    return torch.cat(out)

def extract_all(models, datasets=('opensarship', 'fusarship'), splits=('train', 'val', 'test'),
                device='cuda', batch_size=64, amp=None):
    """Loads each model once, encodes every missing (dataset, split), saves, frees memory."""
    os.makedirs(EMB_DIR, exist_ok=True)
    amp = (device == 'cuda') if amp is None else amp
    for m in models:
        todo = [(d, s) for d in datasets for s in splits if not os.path.exists(emb_path(m, d, s))]
        if not todo:
            print(f"[{m}] all embeddings exist, skipped"); continue
        t0 = time.time()
        bb = R.load_backbone(m, device)
        for d, s in todo:
            x, y = U._load_split(d, s)
            e = _encode(bb, x, device, batch_size, amp)
            assert e.shape == (len(y), bb.spec['dim']), f"bad shape {tuple(e.shape)}"
            assert torch.isfinite(e).all(), f"non-finite values in {m}/{d}/{s}"
            if R.PRETRAINED:   # identical embeddings for all images = weights did not load properly
                assert e.std(0).mean() > 1e-3, f"{m}/{d}/{s}: embeddings barely vary across images"
            _atomic_save({'e': e, 'y': y, 'model': m, 'dataset': d, 'split': s,
                          'dim': bb.spec['dim'], 'classes': U.DATASETS[d]['classes']}, emb_path(m, d, s))
            print(f"[{m}] {d}/{s}: {tuple(e.shape)} | mean norm {e.norm(dim=1).mean():.2f}")
        print(f"[{m}] done in {time.time() - t0:.0f}s")
        del bb; gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()

def load_embeddings(model, dataset, split):
    d = torch.load(emb_path(model, dataset, split), weights_only=True)
    return d['e'], d['y']

def status(models, datasets=('opensarship', 'fusarship'), splits=('train', 'val', 'test')):
    """Prints which embedding files exist."""
    for m in models:
        have = [f"{d[:4]}/{s}" for d in datasets for s in splits if os.path.exists(emb_path(m, d, s))]
        print(f"{m:<14} {len(have)}/{len(datasets) * len(splits)}")

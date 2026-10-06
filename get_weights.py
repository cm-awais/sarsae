"""Downloads every model's weights once into the shared cache (run on the login node)."""
import os, glob, shutil
from config import SARJEPA_CKPT
import fm_registry as R
if not os.path.exists(SARJEPA_CKPT):                      # SAR-JEPA lives in a 13.7 GB Kaggle archive
    import kagglehub
    root = kagglehub.model_download('liweijie19/sar-jepa/pyTorch/default')
    hits = [p for p in glob.glob(os.path.join(root, '**', '*.pth'), recursive=True)
            if 'sar-jepa' in os.path.relpath(p, root).lower() and 'gaus' not in os.path.relpath(p, root).lower()]
    assert len(hits) == 1, hits
    shutil.copy(hits[0], SARJEPA_CKPT); print("SAR-JEPA ->", SARJEPA_CKPT)
for name in R.REGISTRY:                                    # triggers HF / torch.hub downloads into the cache
    m, spec = R.REGISTRY[name]()
    print(f"{name}: ok, dim {spec['dim']}")
    del m

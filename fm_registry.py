"""Foundation-model registry: every model takes the shared (B,1,224,224) [0,1] batch and returns (B, dim)."""
import os, sys, importlib.util, torch, torch.nn as nn, torch.nn.functional as F

from config import REPOS, SARJEPA_CKPT
PRETRAINED = True                        # False only for offline tests
REGISTRY = {}

def register(name):
    def wrap(fn):
        REGISTRY[name] = fn
        return fn
    return wrap

def _import_file(mod_name, path, extra_path=None):
    if extra_path and extra_path not in sys.path:
        sys.path.append(extra_path)          # append, never insert: SAR-JEPA has a profile.py that shadows stdlib
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def _load_matching(model, sd, tag):
    sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
    target = model.state_dict()
    sd = {k: v for k, v in sd.items() if k in target and v.shape == target[k].shape}
    msg = model.load_state_dict(sd, strict=False)
    print(f"[{tag}] loaded {len(sd)}/{len(target)} tensors | missing: {msg.missing_keys}")
    bad = [k for k in msg.missing_keys if not k.startswith(('fc_norm', 'head'))]
    if bad:   # anything beyond the fresh pooling norm / head means pretrained weights did not load
        raise RuntimeError(f"[{tag}] pretrained weights missing for: {bad[:6]}")

class Adapter(nn.Module):
    """Resize, set channels, normalize, call the model. LoRA attaches to modules named 'qkv' inside."""
    def __init__(self, model, spec):
        super().__init__()
        self.model, self.spec = model, spec
        c = spec['in_chans']
        mean = torch.tensor(spec['mean'] or [0.0] * c).view(1, c, 1, 1)
        std  = torch.tensor(spec['std']  or [1.0] * c).view(1, c, 1, 1)
        self.register_buffer('mean', mean, persistent=False)
        self.register_buffer('std', std, persistent=False)

    def forward(self, x):
        s = self.spec
        if x.shape[-1] != s['img_size']:
            x = F.interpolate(x, size=s['img_size'], mode='bicubic', align_corners=False)
        if s['in_chans'] > 1:
            x = x.expand(-1, s['in_chans'], -1, -1)
        x = (x - self.mean) / self.std
        if s.get('wave') is not None:                       # DOFA needs wavelengths
            return self.model.forward_features(x, s['wave'])
        return self.model(x)

def load_backbone(name, device):
    model, spec = REGISTRY[name]()
    spec.setdefault('lora_targets', ['qkv'])
    return Adapter(model, spec).to(device).eval()

# ---------------- timm models ----------------
def _timm(name, model_id):
    @register(name)
    def _load():
        import timm
        m = timm.create_model(model_id, pretrained=PRETRAINED, num_classes=0, img_size=224)
        cfg = m.pretrained_cfg
        return m, dict(dim=m.num_features, in_chans=3, img_size=224,
                       mean=list(cfg['mean']), std=list(cfg['std']))

_timm('dinov2_b',      'vit_base_patch14_dinov2.lvd142m')
_timm('clip_b',        'vit_base_patch16_clip_224.openai')
_timm('dinov3_b_web',  'vit_base_patch16_dinov3.lvd1689m')
_timm('dinov3_l_web',  'vit_large_patch16_dinov3.lvd1689m')
_timm('dinov3_l_sat',  'vit_large_patch16_dinov3.sat493m')

# ---------------- SAR-JEPA ----------------
@register('sarjepa')
def _sarjepa():
    mod = _import_file('models_vit_rp', f'{REPOS}/SAR-JEPA/Pretraining/models_vit_rp.py',
                       f'{REPOS}/SAR-JEPA/Pretraining')
    m = mod.vit_base_patch16(num_classes=0, global_pool=True)
    if PRETRAINED:
        ck = torch.load(SARJEPA_CKPT, map_location='cpu', weights_only=False)
        _load_matching(m, ck.get('model', ck), 'sarjepa')
    return m, dict(dim=768, in_chans=1, img_size=224, mean=None, std=None)

# ---------------- SARATR-X (HiViT-B) ----------------
@register('saratrx')
def _saratrx():
    mod = _import_file('saratrx_hivit', f'{REPOS}/SARATR-X/classification/model/hivit.py')
    in_chans, sd = 3, None
    if PRETRAINED:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download('waterdisappear/SARATR-X', 'weight/186K_all/checkpoint-800.pth')
        ck = torch.load(p, map_location='cpu', weights_only=False)
        sd = {k.replace('module.', '', 1): v for k, v in ck.get('model', ck).items()}
        in_chans = sd['patch_embed.proj.weight'].shape[1]      # read channel count from checkpoint
    # Pretraining used rpe=False (models_hivit_mae.py); the repo's classification hivit_base hard-codes rpe=True
    from functools import partial
    m = mod.HiViT(embed_dim=512, depths=[2, 2, 20], num_heads=8, stem_mlp_ratio=3., mlp_ratio=4., rpe=False,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6), in_chans=in_chans, num_classes=0)
    if sd is not None:
        _load_matching(m, sd, 'saratrx')
    return m, dict(dim=m.num_features, in_chans=in_chans, img_size=224, mean=None, std=None)

# ---------------- DOFA (ViT-B) ----------------
DOFA_URL = 'https://huggingface.co/earthflow/DOFA/resolve/main/DOFA_ViT_base_e100.pth'
@register('dofa_b')
def _dofa():
    mod = _import_file('dofa_v1', f'{REPOS}/DOFA/dofa_v1.py', f'{REPOS}/DOFA')
    m = mod.vit_base_patch16(num_classes=0, global_pool=True)
    if PRETRAINED:
        sd = torch.hub.load_state_dict_from_url(DOFA_URL, map_location='cpu')
        _load_matching(m, sd, 'dofa_b')
    # S1 pretraining used 2 channels, wavelength 3.75 each, uint8-scale mean/std (DOFA repo, waves.json)
    return m, dict(dim=768, in_chans=2, img_size=224, wave=[3.75, 3.75],
                   mean=[166.36275909 / 255, 88.45542715 / 255], std=[64.83126309 / 255, 43.07350145 / 255])

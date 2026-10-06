"""All paths in one place. Edit DATA_ROOT once; everything else lives under WORK."""
import os
ROOT      = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get('SARSAE_DATA', '/CHANGE/ME/cvpr_earth_vision/data/classificiation')
WORK      = os.environ.get('SARSAE_WORK', f'{ROOT}/work')   # cache, weights, embeddings, runs, results
REPOS     = f'{ROOT}/repos'                                 # SAR-JEPA, SARATR-X, DOFA clones

DATASETS = {
    'opensarship': {'root': f'{DATA_ROOT}/c_opensar_splited', 'classes': ['Cargo', 'Tanker', 'Dredging', 'Fishing']},
    'fusarship':   {'root': f'{DATA_ROOT}/c_fusar_splited',   'classes': ['Cargo', 'Fishing', 'Bulk', 'Tanker']},
}
CACHE_DIR  = f'{WORK}/cache'
WEIGHTS    = f'{WORK}/weights'
EMB_DIR    = f'{WORK}/embeddings'
RUNS_DIR   = f'{WORK}/runs'
SARJEPA_CKPT = f'{WEIGHTS}/sarjepa_checkpoint-200.pth'
for _d in [WORK, CACHE_DIR, WEIGHTS, EMB_DIR, RUNS_DIR]:
    os.makedirs(_d, exist_ok=True)

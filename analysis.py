"""Step 4: one table for all 10 configs, per-class recall, and paired factor comparisons.

Reads work/results_frozen.csv (c01, c02), work/results_staged.csv (c03, c04, c07, c08), work/runs/*/result.json (c05, c06, c09, c10).
Writes everything to work/analysis/. CPU only, seconds.
"""
import os, json, glob, itertools
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from config import WORK, RUNS_DIR, DATASETS

OUT = f'{WORK}/analysis'; os.makedirs(OUT, exist_ok=True)
MAIN = ['sarjepa', 'saratrx', 'dofa_b', 'dinov2_b', 'clip_b', 'dinov3_b_web']   # ViT-B class
PROBE = ['dinov3_l_web', 'dinov3_l_sat']
CONFIGS = ['c01_frozen_ce', 'c02_frozen_wce', 'c03_sae_ce', 'c04_sae_wce', 'c05_lora_ce', 'c06_lora_wce',
           'c07_lora_sae_ce', 'c08_lora_sae_wce', 'c09_joint_learned_ce', 'c10_joint_learned_wce']
SHORT = {c: c[:3] for c in CONFIGS}
TAIL = {'opensarship': ['Dredging', 'Fishing'], 'fusarship': ['Bulk', 'Tanker']}

# Each comparison: (name, config A, config B), effect = B - A
PAIRS = [('LoRA (CE)', 'c01_frozen_ce', 'c05_lora_ce'), ('LoRA (WCE)', 'c02_frozen_wce', 'c06_lora_wce'),
         ('SAE on frozen (CE)', 'c01_frozen_ce', 'c03_sae_ce'), ('SAE on frozen (WCE)', 'c02_frozen_wce', 'c04_sae_wce'),
         ('SAE on LoRA (CE)', 'c05_lora_ce', 'c07_lora_sae_ce'), ('SAE on LoRA (WCE)', 'c06_lora_wce', 'c08_lora_sae_wce'),
         ('Joint vs staged (CE)', 'c07_lora_sae_ce', 'c09_joint_learned_ce'),
         ('Joint vs staged (WCE)', 'c08_lora_sae_wce', 'c10_joint_learned_wce'),
         ('Joint vs LoRA only (CE)', 'c05_lora_ce', 'c09_joint_learned_ce'),
         ('Joint vs LoRA only (WCE)', 'c06_lora_wce', 'c10_joint_learned_wce'),
         ('WCE vs CE: frozen', 'c01_frozen_ce', 'c02_frozen_wce'), ('WCE vs CE: SAE', 'c03_sae_ce', 'c04_sae_wce'),
         ('WCE vs CE: LoRA', 'c05_lora_ce', 'c06_lora_wce'), ('WCE vs CE: LoRA+SAE', 'c07_lora_sae_ce', 'c08_lora_sae_wce'),
         ('WCE vs CE: joint', 'c09_joint_learned_ce', 'c10_joint_learned_wce')]

def load_all():
    parts = []
    for f in ['results_frozen.csv', 'results_staged.csv']:
        p = os.path.join(WORK, f)
        if os.path.exists(p): parts.append(pd.read_csv(p))
        else: print(f"WARNING: {f} not found")
    rows = [json.load(open(p)) for p in glob.glob(os.path.join(RUNS_DIR, '*', 'result.json'))]
    if rows: parts.append(pd.DataFrame(rows).drop(columns=['history', 'confusion_matrix'], errors='ignore'))
    df = pd.concat(parts, ignore_index=True)
    df = df[df.config.isin(CONFIGS)]
    dup = df.duplicated(['model', 'dataset', 'config', 'seed']).sum()
    assert dup == 0, f"{dup} duplicate runs"
    return df

def tail_recall(df):
    out = pd.Series(np.nan, index=df.index)
    for d, cls in TAIL.items():
        m = df.dataset == d
        out[m] = df.loc[m, [f'test_recall_{c}' for c in cls]].mean(axis=1)
    return out

def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - rank) * p[i])); adj[i] = run
    return adj

def main():
    df = load_all()
    df['tail_recall'] = tail_recall(df)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)

    # ---- completeness ----
    cnt = df.groupby(['dataset', 'config']).size().unstack(0)
    print("Runs per config (expected 24 = 8 models x 3 seeds):"); print(cnt.to_string(), "\n")

    # ---- Table 1: mean ± std test macro-F1 ----
    g = df.groupby(['dataset', 'model', 'config'])['test_macro_f1'].agg(['mean', 'std', 'count']).reset_index()
    g.to_csv(f'{OUT}/table_macro_f1_long.csv', index=False)
    for d in DATASETS:
        t = g[g.dataset == d].copy()
        t['cell'] = t['mean'].map('{:.3f}'.format) + ' ± ' + t['std'].map('{:.3f}'.format)
        wide = t.pivot(index='model', columns='config', values='cell').reindex(index=MAIN + PROBE, columns=CONFIGS)
        wide.columns = [SHORT[c] for c in wide.columns]
        wide.to_csv(f'{OUT}/table_macro_f1_{d}.csv')
        print(f"Test macro-F1, {d} (mean ± std over 3 seeds):"); print(wide.to_string(), "\n")
        avg = g[(g.dataset == d) & g.model.isin(MAIN)].groupby('config')['mean'].mean().reindex(CONFIGS)
        print(f"  mean over the 6 ViT-B models: " + ' | '.join(f"{SHORT[c]} {v:.3f}" for c, v in avg.items()), "\n")

    # ---- Table 2: per-class recall (mean over 6 ViT-B models and 3 seeds) ----
    for d, cfg in DATASETS.items():
        cols = [f'test_recall_{c}' for c in cfg['classes']] + ['tail_recall', 'test_macro_f1', 'test_balanced_acc']
        t = df[(df.dataset == d) & df.model.isin(MAIN)].groupby('config')[cols].mean().reindex(CONFIGS).round(3)
        t.index = [SHORT[c] for c in t.index]; t.columns = [c.replace('test_', '') for c in t.columns]
        t.to_csv(f'{OUT}/table_per_class_{d}.csv')
        print(f"Per-class recall, {d} (tail = {', '.join(TAIL[d])}; mean over 6 ViT-B models x 3 seeds):")
        print(t.to_string(), "\n")

    # ---- Table 3: paired factor comparisons ----
    rows = []
    for d in DATASETS:
        for metric in ['test_macro_f1', 'tail_recall']:
            for name, a, b in PAIRS:
                x = df[(df.dataset == d) & df.model.isin(MAIN)].pivot_table(
                    index=['model', 'seed'], columns='config', values=metric)
                if a not in x or b not in x: continue
                diff_seed = (x[b] - x[a]).dropna()
                diff_model = diff_seed.groupby('model').mean()            # unit of analysis: model (n = 6)
                p = wilcoxon(diff_model).pvalue if len(diff_model) >= 5 and diff_model.abs().sum() > 0 else np.nan
                rows.append({'dataset': d, 'metric': metric, 'comparison': name, 'A': SHORT[a], 'B': SHORT[b],
                             'mean_diff': diff_model.mean(), 'min_diff': diff_model.min(), 'max_diff': diff_model.max(),
                             'models_improved': f"{(diff_model > 0).sum()}/{len(diff_model)}",
                             'runs_improved': f"{(diff_seed > 0).sum()}/{len(diff_seed)}", 'p_wilcoxon': p})
    comp = pd.DataFrame(rows)
    for (d, metric), idx in comp.groupby(['dataset', 'metric']).groups.items():
        comp.loc[idx, 'p_holm'] = holm(comp.loc[idx, 'p_wilcoxon'].fillna(1.0))
    comp.to_csv(f'{OUT}/table_factor_comparisons.csv', index=False)
    for (d, metric), t in comp.groupby(['dataset', 'metric']):
        print(f"Paired comparisons, {d}, {metric} (B - A, per model averaged over seeds, n = 6 models):")
        print(t.drop(columns=['dataset', 'metric']).round(4).to_string(index=False), "\n")

    # ---- Table 4: domain probe (ViT-L sat - web) ----
    x = df[df.model.isin(PROBE)].pivot_table(index=['dataset', 'config', 'seed'], columns='model', values='test_macro_f1')
    if set(PROBE) <= set(x.columns):
        x['sat_minus_web'] = x['dinov3_l_sat'] - x['dinov3_l_web']
        t = x.groupby(['dataset', 'config'])['sat_minus_web'].agg(['mean', 'std']).round(3).unstack(0)
        t.index = [SHORT[c] for c in t.index]
        t.to_csv(f'{OUT}/table_domain_probe.csv')
        print("Domain probe: DINOv3-L satellite minus web, test macro-F1 (mean ± std over seeds):"); print(t.to_string())
    print(f"\nAll tables saved to {OUT}/")

if __name__ == '__main__':
    main()
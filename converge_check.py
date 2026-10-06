"""Are the frozen heads (configs 1, 2) converged? Fits a fully converged logistic regression
on the same embeddings, raw and standardized, and compares with the frozen.py results.
CPU only, a few minutes. Usage: python converge_check.py
"""
import warnings, time
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import embeddings as E
from config import WORK
warnings.filterwarnings('ignore')

MODELS = ['sarjepa', 'saratrx', 'dofa_b', 'dinov2_b', 'clip_b', 'dinov3_b_web', 'dinov3_l_web', 'dinov3_l_sat']
frozen = pd.read_csv(f'{WORK}/results_frozen.csv')
rows = []
for d in ['fusarship', 'opensarship']:
    for m in MODELS:
        (etr, ytr), (ete, yte) = [[t.numpy() for t in E.load_embeddings(m, d, s)] for s in ['train', 'test']]
        for mode in ['raw', 'standardized']:
            Xtr, Xte = etr, ete
            if mode == 'standardized':
                sc = StandardScaler().fit(etr); Xtr, Xte = sc.transform(etr), sc.transform(ete)
            t0 = time.time()
            clf = LogisticRegression(max_iter=3000, C=1.0).fit(Xtr, ytr)
            rows.append({'dataset': d, 'model': m, 'features': mode,
                         'train_f1': f1_score(ytr, clf.predict(Xtr), average='macro'),
                         'test_f1': f1_score(yte, clf.predict(Xte), average='macro'),
                         'converged': clf.n_iter_.max() < 3000, 'sec': round(time.time() - t0, 1)})
            r = rows[-1]
            print(f"{d:<12} {m:<13} {mode:<12} train {r['train_f1']:.3f} | test {r['test_f1']:.3f} | "
                  f"converged {r['converged']} | {r['sec']}s", flush=True)
lr = pd.DataFrame(rows)
ours = (frozen[frozen.config == 'c01_frozen_ce'].groupby(['dataset', 'model'])
        [['train_macro_f1_at_best', 'test_macro_f1']].mean().rename(columns={
            'train_macro_f1_at_best': 'head_train_f1', 'test_macro_f1': 'head_test_f1'}))
table = lr.pivot_table(index=['dataset', 'model'], columns='features', values=['train_f1', 'test_f1']).round(3)
table.columns = [f'logreg_{a}_{b[:3]}' for a, b in table.columns]
table = table.join(ours.round(3))
lr.to_csv(f'{WORK}/converge_check.csv', index=False)
pd.set_option('display.width', 220)
print("\nComparison (head = our config 1 head, mean of 3 seeds):")
print(table.to_string())
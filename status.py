"""Progress overview: done / running / failed / waiting."""
import os, glob
from config import RUNS_DIR
import worker as W
runs = [W._rid(*r) for r in W.run_list(W.ALL_MODELS, ['fusarship', 'opensarship'], list(W.G.CONFIGS), [0, 1, 2])]
done = [r for r in runs if W._done(r)]
running = [r for r in runs if os.path.exists(W._lock(r)) and r not in done]
failed = [r for r in runs if W._failed(r) and r not in done]
print(f"done {len(done)} | running {len(running)} | failed {len(failed)} | waiting {len(runs) - len(done) - len(running) - len(failed)} | total {len(runs)}")
for r in running: print("  running:", r)
for r in failed:  print("  FAILED:", r, "->", os.path.join(RUNS_DIR, r + '.failed'))

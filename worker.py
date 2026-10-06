"""One worker per GPU. Workers share the run list and claim runs with lock files, so no run is done twice.

Usage: python worker.py            (all 192 GPU runs)
       python worker.py --models sarjepa --datasets fusarship --seeds 0 --max-epochs 2   (quick test)
"""
import os, sys, time, socket, argparse, subprocess, traceback
import gpu_train as G
from config import RUNS_DIR

ALL_MODELS = ['sarjepa', 'saratrx', 'dofa_b', 'dinov2_b', 'clip_b', 'dinov3_b_web', 'dinov3_l_web', 'dinov3_l_sat']
JOB = os.environ.get('SLURM_JOB_ID', f'local{os.getpid()}')

def run_list(models, datasets, configs, seeds):
    # seed 0 for everything first, FUSAR (fast) before OpenSARShip, ViT-B before ViT-L
    return [(m, d, c, s) for s in seeds for d in datasets for m in models for c in configs]

def _rid(m, d, c, s): return f'{m}__{d}__{c}__s{s}'
def _done(rid):       return os.path.exists(os.path.join(RUNS_DIR, rid, 'result.json'))
def _failed(rid):     return os.path.exists(os.path.join(RUNS_DIR, f'{rid}.failed'))
def _lock(rid):       return os.path.join(RUNS_DIR, f'{rid}.lock')

def _alive(job):
    """Is the job holding a lock still running? Unknown -> assume alive (never steal by mistake)."""
    if job.startswith('local'):
        return job == JOB or os.path.exists(f'/proc/{job[5:]}')
    try:
        out = subprocess.run(['squeue', '-h', '-j', job], capture_output=True, text=True, timeout=30)
        return out.returncode != 0 or bool(out.stdout.strip())
    except Exception:
        return True

def claim(rid):
    path = _lock(rid)
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f'{JOB} {socket.gethostname()} {time.ctime()}'.encode()); os.close(fd)
            return True
        except FileExistsError:
            try:
                holder = open(path).read().split()[0]
            except Exception:
                return False
            if _alive(holder):
                return False
            print(f'[{JOB}] stale lock on {rid} (job {holder} is gone), taking over', flush=True)
            try: os.remove(path)
            except FileNotFoundError: pass
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=ALL_MODELS)
    ap.add_argument('--datasets', nargs='+', default=['fusarship', 'opensarship'])
    ap.add_argument('--configs', nargs='+', default=list(G.CONFIGS))
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--max-epochs', type=int, default=None)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--retry-failed', action='store_true')
    a = ap.parse_args()
    runs = run_list(a.models, a.datasets, a.configs, a.seeds)
    print(f'[{JOB}] {len(runs)} runs in the list', flush=True)
    while True:
        did_something = False
        for m, d, c, s in runs:
            rid = _rid(m, d, c, s)
            if _done(rid) or (_failed(rid) and not a.retry_failed) or not claim(rid):
                continue
            did_something = True
            try:
                if _done(rid): continue                      # finished by another worker meanwhile
                print(f'[{JOB}] START {rid}', flush=True)
                G.run(m, d, c, s, device=a.device, max_epochs=a.max_epochs)
                if os.path.exists(os.path.join(RUNS_DIR, f'{rid}.failed')):
                    os.remove(os.path.join(RUNS_DIR, f'{rid}.failed'))
            except Exception:
                tb = traceback.format_exc(); print(f'[{JOB}] FAILED {rid}\n{tb}', flush=True)
                open(os.path.join(RUNS_DIR, f'{rid}.failed'), 'w').write(tb)
            finally:
                try: os.remove(_lock(rid))
                except FileNotFoundError: pass
        if not did_something:
            break
    left = [_rid(*r) for r in runs if not _done(_rid(*r))]
    print(f'[{JOB}] nothing left to claim. Not finished: {len(left)} (running elsewhere or failed)', flush=True)
    G.collect_results()

if __name__ == '__main__':
    main()

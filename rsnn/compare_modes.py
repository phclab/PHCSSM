# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright National Yang Ming Chiao Tung University, Taiwan.
# Creator Po-Han Chiang. Patent pending.
# Noncommercial use only. For commercial licensing see LICENSE.
"""compare_modes.py
================================
Inference-only sanity check that the trained PHCSSM weights also work in
step-by-step RSNN form - the form an RSNN-capable neuromorphic chip would
execute. Both modes share the same ALIF + STP equations, so for the same
weights the spike trains and neuron responses should match. STDP only runs
during training; this script never touches it.

What each mode does on the test set:
  - PHCSSM : K transmission iterations, each running a JAX scan over the
             full T sequence (training-time forward path).
  - RSNN   : feeds the input one timestep at a time (t = 0, 1, ..., T-1).
             At each t, the synapse -> neuron loop produces spike[t] from
             spike[t-1]'s feedback. After the full timeline finishes, the
             accumulated spikes/voltages are passed to readout to produce
             the prediction. No transmission loop, so mstep is undefined
             here (reported NaN). This is the deployment-time forward path
             on an RSNN-capable chip.

Per checkpoint we report (no loss, no training):
  - test accuracy (PHCSSM vs RSNN - agreement implies a faithful port)
  - PHCSSM mstep (from compute_effective_steps); RSNN: n/a
  - wall time on the test set (each mode)
  - peak Python process RSS (CPU memory proxy)
  - JAX live-arrays bytes after PHCSSM (GPU memory proxy)

Usage:
    python compare_modes.py <pth1> [<pth2> ...]
    python compare_modes.py --pattern 'runs/*.pth'
    python compare_modes.py --best PATTERN   # auto-pick highest-epoch pth per (run_tag, seed)

Output: console table + CSV next to script.
"""

import os
import sys
import re
import glob
import time
import csv
import argparse
try:
    import resource
except ImportError:  # not available on Windows
    resource = None
import datetime

import numpy as np

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from loader import load_checkpoint, extract_rsnn_params, load_test_data
import rsnn_forward as rsnn  # forward_sample / readout / encode

import jax
import jax.numpy as jnp
import equinox as eqx


# -- helpers ----------------------------------------------------------------

def _epoch_from_pth(p):
    m = re.search(r"_e(\d+)\.pth$", os.path.basename(p))
    return int(m.group(1)) if m else 0


def _seed_from_pth(p):
    m = re.search(r"Seed(\d+)", os.path.basename(p))
    return int(m.group(1)) if m else None


def _runtag_from_pth(p):
    name = os.path.basename(p)
    m = re.match(r"^(grid_[^_]+)_(.+)_Seed\d+_", name)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    return os.path.splitext(name)[0]


def select_best_pth(paths):
    """Group by (run_tag, seed); keep the highest-epoch .pth in each group."""
    by_key = {}
    for p in paths:
        key = (_runtag_from_pth(p), _seed_from_pth(p))
        if key not in by_key or _epoch_from_pth(p) > _epoch_from_pth(by_key[key]):
            by_key[key] = p
    return sorted(by_key.values())


def peak_rss_mb():
    if resource is None:
        return float("nan")
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # KB -> MB on Linux


def jax_live_bytes():
    try:
        return sum(int(np.prod(a.shape)) * a.dtype.itemsize for a in jax.live_arrays())
    except Exception:
        return 0


# -- PHCSSM evaluation ------------------------------------------------------

def make_phcssm_infer_fn(mode):
    @eqx.filter_jit
    def fn(model, bx):
        if mode in ("osp", "osp+tbptt"):
            logits, all_spikes, _, _ = model.forward(bx, inference=True, mode=mode)
        else:
            logits, all_spikes, _ = model(bx, inference=True)
        return logits, all_spikes
    return fn


def eval_phcssm(model, config, X, y):
    mode = config.get("execution_mode", "nothing")
    bs = int(config.get("batch_size", 32))
    fn = make_phcssm_infer_fn(mode)
    X_jax = jnp.asarray(X)
    y_jax = jnp.asarray(y)
    n = X_jax.shape[0]

    # Warmup compile (excluded from timing)
    fn(model, X_jax[:min(bs, n)])

    correct_dev = jnp.int32(0)
    eff_acc = jnp.float32(0.0)
    n_batches = 0
    t0 = time.perf_counter()
    for i in range(0, n, bs):
        bx = X_jax[i:i + bs]
        by = y_jax[i:i + bs]
        logits, all_spikes = fn(model, bx)
        preds = jnp.argmax(logits, axis=-1)
        correct_dev = correct_dev + (preds == by).sum()
        eff_acc = eff_acc + model.compute_effective_steps(all_spikes)
        n_batches += 1
    correct_dev.block_until_ready()
    elapsed = time.perf_counter() - t0
    acc = float(correct_dev) / n
    mstep = float(eff_acc) / max(n_batches, 1)
    return acc, mstep, elapsed


# -- RSNN evaluation --------------------------------------------------------

def eval_rsnn(model, config, X, y):
    P = extract_rsnn_params(model, config)
    n = len(X)
    correct = 0
    t0 = time.perf_counter()
    for i in range(n):
        pred, _, _ = rsnn.forward_sample(X[i], P, horizon=None)
        if pred == int(y[i]):
            correct += 1
    elapsed = time.perf_counter() - t0
    acc = correct / n
    # RSNN has no transmission loop: it's a single causal pass through time
    # with unbounded recurrence over t, so mstep is not defined here.
    return acc, float("nan"), elapsed


# -- per-checkpoint comparison ----------------------------------------------

def compare_one(pth_path):
    pre_rss = peak_rss_mb()
    pre_jax = jax_live_bytes()

    model, config = load_checkpoint(pth_path)
    test = load_test_data(config, pth_path)
    if test is None:
        raise FileNotFoundError(f"test data not found for {pth_path}")
    X, y = test

    # PHCSSM
    p_acc, p_mstep, p_time = eval_phcssm(model, config, X, y)
    p_rss = peak_rss_mb()
    p_jax = jax_live_bytes()

    # RSNN (after JAX cleanup so RSS measurement is comparable)
    try:
        jax.clear_caches()
    except Exception:
        pass
    r_acc, r_mstep, r_time = eval_rsnn(model, config, X, y)
    r_rss = peak_rss_mb()

    return {
        "pth": os.path.basename(pth_path),
        "run_tag": _runtag_from_pth(pth_path),
        "seed": _seed_from_pth(pth_path),
        "dataset": config.get("dataset_name", "?"),
        "num_transmissions": int(config.get("num_transmissions", 0)),
        "clw": config.get("convergence_loss_weight", 0.0),
        "n_test": len(X),
        # PHCSSM
        "phc_acc": p_acc,
        "phc_mstep": p_mstep,
        "phc_time_s": p_time,
        "phc_rss_mb": p_rss - pre_rss,
        "phc_jax_mb": (p_jax - pre_jax) / 1024.0 / 1024.0,
        # RSNN
        "rsnn_acc": r_acc,
        "rsnn_mstep": r_mstep,
        "rsnn_time_s": r_time,
        "rsnn_rss_mb": r_rss - p_rss,
        # Deltas
        "acc_delta_pp": (p_acc - r_acc) * 100,
        "speed_x": r_time / p_time if p_time > 0 else float("nan"),
    }


# -- CLI --------------------------------------------------------------------

def collect_paths(args):
    paths = []
    if args.paths:
        paths.extend(args.paths)
    if args.pattern:
        paths.extend(glob.glob(args.pattern))
    if not paths:
        # default: best clw60 pth from v6 grid output dirs
        paths = glob.glob(os.path.join(PROJECT_DIR, "runs", "*.pth"))
    if args.best_per_runtag and paths:
        paths = select_best_pth(paths)
    return sorted(paths)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="explicit .pth files")
    ap.add_argument("--pattern", help="glob pattern to add .pth files")
    ap.add_argument("--best-per-runtag", action="store_true",
                    help="when multiple .pth match the same (run_tag, seed), keep highest-epoch")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap number of checkpoints to compare (0 = all)")
    ap.add_argument("--csv", default=None,
                    help="output CSV path (default: compare_modes_<ts>.csv)")
    args = ap.parse_args()

    paths = collect_paths(args)
    if args.limit > 0:
        paths = paths[: args.limit]
    if not paths:
        print("No .pth files matched.", file=sys.stderr)
        return 1

    print("=" * 92)
    print(f"PHCSSM vs RSNN comparison - {len(paths)} checkpoint(s)")
    print(f"Start: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 92)

    rows = []
    for i, p in enumerate(paths, 1):
        print(f"\n[{i}/{len(paths)}] {os.path.basename(p)}")
        try:
            r = compare_one(p)
            rows.append(r)
            print(f"  PHCSSM : acc={r['phc_acc']*100:6.2f}% mstep={r['phc_mstep']:5.2f}/{r['num_transmissions']:>2} "
                  f"time={r['phc_time_s']:6.2f}s  RSS+={r['phc_rss_mb']:6.1f}MB  JAX+={r['phc_jax_mb']:6.1f}MB")
            print(f"  RSNN   : acc={r['rsnn_acc']*100:6.2f}% mstep= n/a (single pass)  "
                  f"time={r['rsnn_time_s']:6.2f}s  RSS+={r['rsnn_rss_mb']:6.1f}MB")
            print(f"  delta  : acc={r['acc_delta_pp']:+5.2f}pp  speed: RSNN/PHCSSM = {r['speed_x']:6.2f}x")
        except Exception as e:
            import traceback
            print(f"  [ERR] {e}")
            traceback.print_exc()

    # -- summary table ----------------------------------------------------
    print("\n" + "=" * 110)
    print("Summary (rows sorted by dataset, run_tag, seed)")
    print("=" * 110)
    header = f"{'dataset':<10} {'run_tag':<35} {'seed':>5} {'phc_acc':>8} {'rsnn_acc':>9} {'dacc_pp':>8} {'phcM/K':>8} {'phc_t':>7} {'rsnn_t':>7} {'rsnn/phc':>9}"
    print(header)
    print("-" * len(header))
    for r in sorted(rows, key=lambda r: (r["dataset"], r["run_tag"], r["seed"] or 0)):
        m_str = f"{r['phc_mstep']:.2f}/{r['num_transmissions']}"
        print(f"{r['dataset']:<10} {r['run_tag'][:35]:<35} {str(r['seed']):>5} "
              f"{r['phc_acc']*100:>7.2f}% {r['rsnn_acc']*100:>8.2f}% {r['acc_delta_pp']:>+7.2f} "
              f"{m_str:>8} {r['phc_time_s']:>6.2f}s {r['rsnn_time_s']:>6.2f}s {r['speed_x']:>8.2f}x")

    # -- CSV --------------------------------------------------------------
    csv_path = args.csv or os.path.join(
        SCRIPT_DIR,
        f"compare_modes_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV saved: {csv_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

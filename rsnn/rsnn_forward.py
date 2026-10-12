#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright National Yang Ming Chiao Tung University, Taiwan.
# Creator Po-Han Chiang. Patent pending.
# Noncommercial use only. For commercial licensing see LICENSE.
"""
rsnn_forward.py
============================
JAX and Equinox implementation of the sequential RSNN forward pass.

Compares a step-by-step RSNN against itself with a *truncated causal feedback
horizon*. Reads checkpoints written by ``phcssm/trainer.py``
(pickle of {params, static, config, ...}) and runs the same RSNN logic that
was used for the JAX-v4 study, but using NumPy throughout - no PyTorch.

Normal RSNN: full causal chain - spike[t] depends on all spike[0..t-1].
Truncated  : every M steps the synaptic feedback state (delay_buf, u_trace, R,
             prev_spike) is reset to its initial value. Neuron-internal
             states (V_f, adapt, V_res) are *not* reset, since PHCSSM also
             integrates these via scan_decay over the full sequence within
             each transmission.

Usage:
    python rsnn_forward.py <pth_file> [--horizons 12,24,48,96]
                                                   [--samples N]
"""

import os
import sys
import argparse
import time
import math
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from loader import (
    load_checkpoint, extract_rsnn_params, load_test_data,
    _sigmoid, _softplus, infer_dataset_from_path,
)


# -- RSNN forward (NumPy step-by-step) ----------------------------------------

def encode(x_np, P):
    """Linear encoder + LayerNorm (eps=1e-5), matches eqx.nn.LayerNorm."""
    h = x_np @ P["enc_w"].T + P["enc_b"]            # [T, H]
    mean = h.mean(axis=-1, keepdims=True)
    var  = h.var(axis=-1, keepdims=True)
    h = (h - mean) / np.sqrt(var + 1e-5) * P["ln_w"] + P["ln_b"]
    return h


def readout(activity, P):
    """Apply readout slicing + strategy + decoder. Returns argmax class id."""
    if P["readout_slices"]:
        ri = np.concatenate([activity[:, s:e] for s, e in P["readout_slices"]], axis=1)
    else:
        ri = activity
    strat = P["readout_strategy"]
    if strat == "ssm":
        rms = np.sqrt((ri ** 2).mean(axis=-1, keepdims=True) + 1e-6)
        r = (ri / rms * P["norm_w"])[-1]
    elif strat == "mean":
        r = ri.mean(axis=0)
    elif strat == "last":
        r = ri[-1]
    elif strat == "sum":
        r = ri.sum(axis=0)
    elif strat == "max":
        r = ri.max(axis=0)
    elif strat in ("max_last", "last_max"):
        r = (ri.max(axis=0) + ri[-1]) / 2.0
    elif strat in ("max_mean", "mean_max"):
        r = (ri.max(axis=0) + ri.mean(axis=0)) / 2.0
    elif strat == "com":
        T = ri.shape[0]
        t_idx = np.arange(T, dtype=np.float32) / max(T - 1, 1)
        w = ri + 1e-8
        r = (w * t_idx[:, None]).sum(axis=0) / w.sum(axis=0)
    elif strat == "weighted":
        T = ri.shape[0]
        weights = np.linspace(0.0, 1.0, T, dtype=np.float32)[:, None]
        r = (ri * weights).sum(axis=0) / (weights.sum() + 1e-8)
    else:
        r = ri[-1]
    logits = r @ P["dec_w"].T + P["dec_b"]
    return int(np.argmax(logits)), logits


def forward_sample(x_np, P, horizon=None, return_voltages=False):
    """
    Step-by-step RSNN forward pass with optional causal-horizon truncation.

    horizon=None : full causal chain.
    horizon=M    : every M steps, reset feedback path (delay buffer, u_trace,
                   R, prev_spike) to its initial values.

    Returns: (pred, all_spikes [T,H], reset_count)
    """
    encoded = encode(x_np, P)
    sensory = encoded * P["input_mask"]
    T, H = encoded.shape

    V_f      = np.zeros(H, dtype=np.float32)
    adapt    = np.zeros(H, dtype=np.float32)
    V_res    = np.zeros(H, dtype=np.float32)
    u_trace  = P["U_full"].copy()
    R        = np.ones(H, dtype=np.float32)
    prev_sp  = np.zeros(H, dtype=np.float32)

    delay = P["delay"]
    if delay > 0:
        delay_buf = np.zeros((delay, H), dtype=np.float32)

    all_spikes   = np.zeros((T, H), dtype=np.float32)
    all_voltages = np.zeros((T, H), dtype=np.float32)

    reset_count = 0

    for t in range(T):
        # -- Horizon-based feedback reset ----------------------------------
        if horizon is not None and t > 0 and t % horizon == 0:
            if delay > 0:
                delay_buf = np.zeros((delay, H), dtype=np.float32)
            prev_sp = np.zeros(H, dtype=np.float32)
            u_trace = P["U_full"].copy()
            R = np.ones(H, dtype=np.float32)
            reset_count += 1

        # -- Synaptic delay ------------------------------------------------
        if delay > 0:
            delayed = delay_buf[0].copy()
            delay_buf = np.roll(delay_buf, -1, axis=0)
        else:
            delayed = prev_sp

        # -- STP facilitation ----------------------------------------------
        u_decay = np.clip(P["alpha_f"] - P["facil_amp"] * delayed, 1e-6, 1.0)
        u_trace = u_decay * u_trace + (1 - P["alpha_f"]) * P["U_full"] + P["facil_amp"] * delayed
        u_trace = np.clip(u_trace, 0.0, 1.0)

        # -- STP depression ------------------------------------------------
        r_decay = np.clip(P["alpha_r"] - u_trace * delayed, 1e-6, 1.0)
        R = r_decay * R + (1 - P["alpha_r"])
        R = np.clip(R, 0.0, 1.0)

        # -- Recurrent synaptic current ------------------------------------
        syn = (delayed * R * u_trace) @ P["W_eff"].T
        I_t = P["drive_alpha"] * sensory[t] + syn

        # -- Neuron dynamics -----------------------------------------------
        V_f    = P["alpha_exc"] * V_f + _softplus(I_t)
        adapt  = P["adapt_alpha"] * adapt + _softplus(_sigmoid(V_f - P["v_th"]))
        v_eff  = P["v_th"] + adapt * P["beta_adapt"]
        sp_pre = (V_f - v_eff > 0).astype(np.float32)
        V_res  = P["alpha_inh"] * V_res + _softplus(sp_pre * P["reset_weight"])
        V_final = V_f - V_res - adapt * P["beta_adapt"]
        spike  = (V_final - P["v_th"] > 0).astype(np.float32)

        all_spikes[t]   = spike
        all_voltages[t] = V_final
        if delay > 0:
            delay_buf[-1] = spike
        prev_sp = spike

    activity = all_voltages if P["readout_source"] == "voltage" else all_spikes
    pred, _ = readout(activity, P)
    if return_voltages:
        return pred, all_spikes, all_voltages, reset_count
    return pred, all_spikes, reset_count


def evaluate(X, y, P, horizon=None):
    N = len(X)
    correct = 0
    preds = []
    for i in range(N):
        pred, _, _ = forward_sample(X[i], P, horizon=horizon)
        preds.append(pred)
        if pred == int(y[i]):
            correct += 1
        if (i + 1) % 50 == 0 or i == N - 1:
            tag = f"M={horizon}" if horizon else "full"
            print(f"\r    RSNN({tag})  {i+1}/{N} acc={correct/(i+1):.3f}",
                  end="", flush=True)
    print()
    return correct / N, preds


# -- Decay-constant report (informational) ------------------------------------

def report_decay(P):
    print("  -- Decay Constants (per-neuron statistics) --")
    for name, vals in [("alpha_exc", P["alpha_exc"]),
                       ("adapt_alpha", P["adapt_alpha"]),
                       ("alpha_f (STP facilitation)", P["alpha_f"]),
                       ("alpha_r (STP recovery)", P["alpha_r"])]:
        if np.ndim(vals) > 0 and np.size(vals) > 1:
            mn = float(np.min(vals)); mx = float(np.max(vals)); mu = float(np.mean(vals))
            h_mu  = math.log(0.01) / math.log(mu) if 0 < mu < 1 else float("inf")
            h_mx  = math.log(0.01) / math.log(mx) if 0 < mx < 1 else float("inf")
            ret_mu = mu ** 12; ret_mx = mx ** 12
            print(f"    {name}:")
            print(f"      min={mn:.4f}  mean={mu:.4f}  max={mx:.4f}")
            print(f"      After 12 steps: mean retains {ret_mu*100:.1f}%, "
                  f"max retains {ret_mx*100:.1f}%")
            print(f"      Memory horizon (to 1%): mean={h_mu:.0f} steps, "
                  f"max={h_mx:.0f} steps")
        else:
            v = float(vals) if np.ndim(vals) == 0 else float(vals.flat[0])
            print(f"    {name}: {v:.4f}")
    print()


# -- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RSNN causal depth comparison (v6 / Equinox checkpoints)")
    parser.add_argument("pth", help="Path to .pth checkpoint (v6 pickle format)")
    parser.add_argument("--horizons", default="12,24,48,96,192",
                        help="Comma-separated causal horizon values to test")
    parser.add_argument("--samples", type=int, default=None,
                        help="Limit number of test samples")
    args = parser.parse_args()

    horizons = [int(h) for h in args.horizons.split(",")]

    print("=" * 70)
    print("  RSNN Causal Depth Comparison (JAX/Equinox checkpoints)")
    print("=" * 70)
    print(f"  Checkpoint: {os.path.basename(args.pth)}")
    print(f"  Horizons to test: {horizons}")
    print()

    model, config = load_checkpoint(args.pth)
    P = extract_rsnn_params(model, config)

    ds = infer_dataset_from_path(args.pth, config)
    K_orig = config.get("num_transmissions", 12)
    print(f"  Dataset: {ds}")
    print(f"  neuron_dim: {P['H']}")
    print(f"  num_transmissions (K): {K_orig}")
    print(f"  synaptic_delay: {P['delay']}")
    print(f"  drive_alpha: {P['drive_alpha']}")
    print(f"  readout: source={P['readout_source']}, "
          f"strategy={P['readout_strategy']}")
    print()

    report_decay(P)

    Xy = load_test_data(config, args.pth, splits=("test",))
    if Xy is None or Xy[0] is None:
        print(f"  ERROR: cannot find test data for dataset {ds}.")
        sys.exit(1)
    X, y = Xy
    if args.samples:
        X, y = X[:args.samples], y[:args.samples]
    T = X.shape[1]
    print(f"  Data: {X.shape}  T={T}  classes={len(set(y.tolist()))}")
    print()

    # Full RSNN baseline
    print("  Running full RSNN (no horizon limit)...")
    t0 = time.time()
    acc_full, preds_full = evaluate(X, y, P, horizon=None)
    dt_full = time.time() - t0
    print(f"    Full RSNN: acc={acc_full*100:.1f}%  ({dt_full:.1f}s)")
    print()

    results = {"full": (acc_full, preds_full)}

    for M in horizons:
        if M >= T:
            print(f"  Horizon M={M} >= T={T}, equivalent to full RSNN. Skipping.")
            results[M] = (acc_full, preds_full)
            continue

        n_resets = (T - 1) // M
        print(f"  Running RSNN with horizon M={M} (resets {n_resets}x per "
              f"sample, covers {M}/{T} = {M/T*100:.1f}% per window)...")
        t0 = time.time()
        acc_m, preds_m = evaluate(X, y, P, horizon=M)
        dt = time.time() - t0
        results[M] = (acc_m, preds_m)

        agree = sum(1 for a, b in zip(preds_full, preds_m) if a == b)
        print(f"    RSNN(M={M}): acc={acc_m*100:.1f}%  "
              f"agreement_with_full={agree}/{len(preds_full)} "
              f"({agree/len(preds_full)*100:.1f}%)  ({dt:.1f}s)")
        print()

    # Summary
    print()
    print("=" * 70); print("  SUMMARY"); print("=" * 70); print()
    print(f"  Dataset: {ds}  T={T}  K={K_orig}  delay={P['delay']}")
    print()
    print(f"  {'Horizon':>10} {'Accuracy':>10} {'d vs Full':>10} "
          f"{'Agreement':>12} {'Resets/sample':>14}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*14}")
    for key in ["full"] + horizons:
        acc, preds = results[key]
        if key == "full":
            print(f"  {'Full':>10} {acc*100:>9.1f}% {'---':>10} "
                  f"{'---':>12} {'0':>14}")
        else:
            M = key
            if M >= T:
                delta_str = "= full"; agree_str = "100.0%"
            else:
                delta = acc - acc_full
                agree = sum(1 for a, b in zip(preds_full, preds) if a == b) / len(preds_full)
                delta_str = f"{delta*100:+.1f}%"
                agree_str = f"{agree*100:.1f}%"
            n_r = max(0, (T - 1) // M)
            print(f"  {'M=' + str(M):>10} {acc*100:>9.1f}% "
                  f"{delta_str:>10} {agree_str:>12} {n_r:>14}")

    if horizons:
        worst_M = min(horizons)
        acc_w, preds_w = results[worst_M]
        disagree = [i for i, (a, b) in enumerate(zip(preds_full, preds_w)) if a != b]
        if disagree:
            print()
            print(f"  -- Per-sample disagreements: Full vs M={worst_M} "
                  f"({len(disagree)} samples) --")
            print(f"  {'Idx':>5} {'y_true':>6} {'Full':>6} "
                  f"{'M=' + str(worst_M):>8} {'Full ok?':>10} {'M ok?':>8}")
            for idx in disagree[:20]:
                yt = int(y[idx])
                pf = preds_full[idx]; pm = preds_w[idx]
                print(f"  {idx:>5} {yt:>6} {pf:>6} {pm:>8} "
                      f"{'O' if pf == yt else 'X':>10} "
                      f"{'O' if pm == yt else 'X':>8}")
            if len(disagree) > 20:
                print(f"  ... and {len(disagree) - 20} more")

    print()
    print("Done.")


if __name__ == "__main__":
    main()

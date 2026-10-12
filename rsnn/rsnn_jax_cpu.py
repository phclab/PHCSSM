"""rsnn_jax_cpu.py
JAX/lax.scan port of NumPy RSNN forward, jit-able to CPU backend.
Goal: same bit-level result as rsnn_forward.forward_sample
(when horizon=None) but faster via JIT.

Usage as module:
    from rsnn_jax_cpu import make_rsnn_forward_cpu
    fwd = make_rsnn_forward_cpu(P)         # pre-compiles for given param shapes
    pred, all_spikes, all_voltages = fwd(x_np)

CLI bench:
    python rsnn_jax_cpu.py <pth>
"""
import os, sys, time, argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR); sys.path.insert(0, PROJECT_DIR)

# Force JAX to also be available on CPU; we still set backend='cpu' on jit.
import jax
import jax.numpy as jnp

from loader import load_checkpoint, extract_rsnn_params, load_test_data
import rsnn_forward as rsnn_np


def _sigmoid(x):
    return 1.0 / (1.0 + jnp.exp(-x))

def _softplus(x):
    return jnp.maximum(x, 0.0) + jnp.log1p(jnp.exp(-jnp.abs(x)))


def _encode(x, P):
    h = x @ P["enc_w"].T + P["enc_b"]
    mean = h.mean(axis=-1, keepdims=True)
    var  = h.var(axis=-1, keepdims=True)
    h = (h - mean) / jnp.sqrt(var + 1e-5) * P["ln_w"] + P["ln_b"]
    return h


def _readout(activity, P):
    """Apply readout slicing + strategy + decoder."""
    if P["readout_slices"]:
        ri = jnp.concatenate([activity[:, s:e] for s, e in P["readout_slices"]], axis=1)
    else:
        ri = activity
    strat = P["readout_strategy"]
    if strat == "mean":
        r = ri.mean(axis=0)
    elif strat == "last":
        r = ri[-1]
    elif strat == "sum":
        r = ri.sum(axis=0)
    elif strat == "max":
        r = ri.max(axis=0)
    elif strat == "ssm":
        rms = jnp.sqrt((ri ** 2).mean(axis=-1, keepdims=True) + 1e-6)
        r = (ri / rms * P["norm_w"])[-1]
    else:
        raise NotImplementedError(f"strategy {strat} not yet ported")
    logits = r @ P["dec_w"].T + P["dec_b"]
    return jnp.argmax(logits), logits


def _make_scan_body(P):
    """Return scan body fn closing over static params."""
    delay = int(P["delay"])
    has_delay = delay > 0
    drive_alpha = float(P["drive_alpha"])

    def body(carry, sensory_t):
        V_f, adapt, V_res, u_trace, R, prev_sp, delay_buf = carry

        if has_delay:
            delayed = delay_buf[0]
            delay_buf = jnp.roll(delay_buf, -1, axis=0)
        else:
            delayed = prev_sp

        u_decay = jnp.clip(P["alpha_f"] - P["facil_amp"] * delayed, 1e-6, 1.0)
        u_trace = u_decay * u_trace + (1 - P["alpha_f"]) * P["U_full"] + P["facil_amp"] * delayed
        u_trace = jnp.clip(u_trace, 0.0, 1.0)

        r_decay = jnp.clip(P["alpha_r"] - u_trace * delayed, 1e-6, 1.0)
        R = r_decay * R + (1 - P["alpha_r"])
        R = jnp.clip(R, 0.0, 1.0)

        syn = (delayed * R * u_trace) @ P["W_eff"].T
        I_t = drive_alpha * sensory_t + syn

        V_f    = P["alpha_exc"] * V_f + _softplus(I_t)
        adapt  = P["adapt_alpha"] * adapt + _softplus(_sigmoid(V_f - P["v_th"]))
        v_eff  = P["v_th"] + adapt * P["beta_adapt"]
        sp_pre = (V_f - v_eff > 0).astype(jnp.float32)
        V_res  = P["alpha_inh"] * V_res + _softplus(sp_pre * P["reset_weight"])
        V_final = V_f - V_res - adapt * P["beta_adapt"]
        spike  = (V_final - P["v_th"] > 0).astype(jnp.float32)

        if has_delay:
            delay_buf = delay_buf.at[-1].set(spike)
        prev_sp = spike

        return (V_f, adapt, V_res, u_trace, R, prev_sp, delay_buf), (spike, V_final)

    return body


def make_rsnn_forward_cpu(P_np):
    """Pre-compile RSNN forward on CPU. Returns fn(x_np) -> (pred, spikes, voltages)."""
    # Convert P to jnp arrays once
    P_jnp = {}
    for k, v in P_np.items():
        if isinstance(v, np.ndarray):
            P_jnp[k] = jnp.asarray(v)
        else:
            P_jnp[k] = v
    delay = int(P_jnp["delay"])
    H = int(P_jnp["W_eff"].shape[0])

    body = _make_scan_body(P_jnp)

    import functools
    @functools.partial(jax.jit, backend='cpu')
    def forward_cpu(x):
        encoded = _encode(x, P_jnp)
        sensory = encoded * P_jnp["input_mask"]
        T = sensory.shape[0]

        V_f      = jnp.zeros(H, dtype=jnp.float32)
        adapt    = jnp.zeros(H, dtype=jnp.float32)
        V_res    = jnp.zeros(H, dtype=jnp.float32)
        u_trace  = P_jnp["U_full"].astype(jnp.float32)
        R        = jnp.ones(H, dtype=jnp.float32)
        prev_sp  = jnp.zeros(H, dtype=jnp.float32)
        delay_buf = jnp.zeros((max(delay, 1), H), dtype=jnp.float32)
        carry0 = (V_f, adapt, V_res, u_trace, R, prev_sp, delay_buf)

        _, (all_spikes, all_voltages) = jax.lax.scan(body, carry0, sensory)

        activity = all_voltages if P_jnp["readout_source"] == "voltage" else all_spikes
        pred, _ = _readout(activity, P_jnp)
        return pred, all_spikes, all_voltages

    def fn(x_np):
        x = jnp.asarray(x_np)
        pred, sp, v = forward_cpu(x)
        return int(pred), np.asarray(sp), np.asarray(v)

    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pth")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--rep", type=int, default=5)
    args = ap.parse_args()

    print(f"Loading {os.path.basename(args.pth)}")
    print(f"  JAX devices: {jax.devices()}  default backend: {jax.default_backend()}")

    model, config = load_checkpoint(args.pth)
    P = extract_rsnn_params(model, config)
    test = load_test_data(config, args.pth)
    if test is None:
        raise FileNotFoundError("test data not found")
    X, y = test
    X = X[:args.n]; y = y[:args.n]
    n = len(X)
    T = X.shape[1]
    print(f"  samples={n} T={T} H={int(P['W_eff'].shape[0])}")

    print("\n[building CPU JIT...]")
    fwd_jit = make_rsnn_forward_cpu(P)
    # warmup
    _ = fwd_jit(X[0])
    _ = fwd_jit(X[0])

    # bench JIT
    times_jit = []
    for _ in range(args.rep):
        t0 = time.perf_counter()
        for i in range(n):
            fwd_jit(X[i])
        times_jit.append((time.perf_counter() - t0) / n)

    # bench NumPy
    rsnn_np.forward_sample(X[0], P)  # warmup
    times_np = []
    for _ in range(args.rep):
        t0 = time.perf_counter()
        for i in range(n):
            rsnn_np.forward_sample(X[i], P)
        times_np.append((time.perf_counter() - t0) / n)

    # accuracy comparison
    pred_jit = np.array([fwd_jit(x)[0] for x in X])
    pred_np = np.array([rsnn_np.forward_sample(x, P)[0] for x in X])
    y_np = np.asarray(y)
    print(f"\nResults (n={n} samples, median over {args.rep} reps):")
    print(f"  RSNN NumPy time/sample: {np.median(times_np)*1e3:7.2f} ms  acc={float((pred_np==y_np).mean())*100:.2f}%")
    print(f"  RSNN JAX-CPU time/sample:{np.median(times_jit)*1e3:7.2f} ms  acc={float((pred_jit==y_np).mean())*100:.2f}%")
    print(f"  pred_agree NumPy<->JAX: {float((pred_np==pred_jit).mean())*100:.2f}%")
    speedup = np.median(times_np) / np.median(times_jit)
    print(f"  speedup (CPU JAX vs NumPy): {speedup:.2f}x")


if __name__ == "__main__":
    main()

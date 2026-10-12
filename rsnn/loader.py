# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright National Yang Ming Chiao Tung University, Taiwan.
# Creator Po-Han Chiang. Patent pending.
# Noncommercial use only. For commercial licensing see LICENSE.
"""
loader.py
=============
Shared helpers for loading Equinox and JAX checkpoints and extracting
parameters in NumPy form for the step-by-step RSNN inference scripts.

A checkpoint is a pickle dict:
    {
        "params": <equinox params partition>,
        "static": <equinox static partition>,
        "config": <python dict>,
        "acc": float, "test_acc": float, "epoch": int, "__version__": str,
    }
The full PHCSSM module is reconstructed via eqx.combine(params, static).

JAX + Equinox must be importable. The phcssm package at the repository
root must also be on the path.
"""

import os
import sys
import re
import pickle
import numpy as np

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)


# -- Pickle compatibility shim for older JAX-saved checkpoints ----------------

def _safe_pickle_load(f):
    """Unpickle data written by an older JAX, then undo the patch.

    Those pickles pass a ``named_shape`` keyword that current JAX no longer
    accepts, so ``ShapedArray.update`` is wrapped for the duration of the load.
    The original is restored in a finally block, so nothing global is left
    altered and the two read sites do not depend on which runs first.
    """
    try:
        import jax._src.core as _jax_core  # noqa
    except Exception:
        return pickle.load(f)

    _orig_update = _jax_core.ShapedArray.update

    def _patched_update(self, *args, **kwargs):
        kwargs.pop("named_shape", None)
        return _orig_update(self, *args, **kwargs)

    _jax_core.ShapedArray.update = _patched_update
    try:
        return pickle.load(f)
    finally:
        _jax_core.ShapedArray.update = _orig_update


# -- Region / readout slice helpers (mirror phcssm.model logic) ----------

def calculate_region_slices(config):
    h = config["neuron_dim"]
    slices = []
    start = 0
    for i, r in enumerate(config["partition_region_ratio"]):
        end = h if i == len(config["partition_region_ratio"]) - 1 else start + int(h * r)
        split = start + int((end - start) * config["region_ei_ratios"][i])
        slices.append((start, split, end))
        start = end
    return slices


def calculate_readout_slices(config):
    h = config["neuron_dim"]
    slices = []
    start = 0
    for i, r in enumerate(config["partition_region_ratio"]):
        end = h if i == len(config["partition_region_ratio"]) - 1 else start + int(h * r)
        mid = start + int((end - start) * config["region_ei_ratios"][i])
        out_t = config["readout_target_types"][i]
        if out_t in ("E", "I", "EI"):
            s2 = start if out_t != "I" else mid
            e2 = mid if out_t == "E" else end
            if e2 > s2:
                slices.append((s2, e2))
        start = end
    return slices


# -- Checkpoint loader --------------------------------------------------------

def load_checkpoint(pth_path):
    """
    Load a .pth (pickle) checkpoint and return (model, config).

    `model` is a fully recombined `phcssm.model.PHCSSM` Equinox module,
    suitable for both attribute access (RSNN parameter extraction) and direct
    forward calls (full PHCSSM inference).
    """
    import equinox as eqx
    from phcssm import model as phc  # noqa  (registers pytree types)

    with open(pth_path, "rb") as f:
        ck = _safe_pickle_load(f)

    if "params" not in ck or "static" not in ck:
        raise KeyError(f"{pth_path}: missing 'params'/'static' keys "
                       f"(found {list(ck.keys())}).  This script expects that format.")

    model = eqx.combine(ck["params"], ck["static"])
    return model, ck["config"]


# -- Extract a NumPy-only parameter bundle for the step-by-step RSNN ----------

def extract_rsnn_params(model, config):
    """
    Pull every array the RSNN needs into a plain NumPy dict.

    Mirrors the parameter set used by the RSNN forward pass in
    ``rsnn_forward.py``.
    """
    def _np(arr):
        return np.asarray(arr).astype(np.float32)

    p = {}
    # Encoder
    p["enc_w"] = _np(model.encoder.linear.weight)   # [H, in_dim]
    p["enc_b"] = _np(model.encoder.linear.bias)     # [H]
    p["ln_w"]  = _np(model.encoder.norm.weight)     # [H]
    p["ln_b"]  = _np(model.encoder.norm.bias)       # [H]

    # NeuronLayer
    p["tau_exc"]         = _np(model.shared_nl.tau_exc).reshape(-1)
    p["tau_inh"]         = _np(model.shared_nl.tau_inh).reshape(-1)
    p["v_th"]            = _np(model.shared_nl.v_th).reshape(-1)
    p["reset_weight"]    = _np(model.shared_nl.reset_weight).reshape(-1)
    p["tau_adapt_param"] = _np(model.shared_nl.tau_adapt_param).reshape(-1)
    p["beta_adapt"]      = float(model.shared_nl.beta_adapt)

    # SynapseLayer
    p["W"]               = _np(model.shared_sl.weight)         # [H, H]
    p["mask"]            = _np(model.shared_sl.mask)           # [H, H]
    p["U_logit"]         = _np(model.shared_sl.U_logit).reshape(-1)
    p["tau_rec_logit"]   = _np(model.shared_sl.tau_rec_logit).reshape(-1)
    p["tau_facil_param"] = _np(model.shared_sl.tau_facil_param).reshape(-1)
    p["facil_amp"]       = float(model.shared_sl.facil_amp)
    p["delay"]           = int(model.shared_sl.delay)

    # Input / readout / decoder
    p["input_mask"]       = _np(model.input_mask)
    p["drive_alpha"]      = float(model.drive_alpha)
    p["readout_source"]   = model.readout_source
    p["readout_strategy"] = model.readout_strategy
    p["readout_slices"]   = [tuple(s) for s in model.readout_slices]
    p["norm_w"]           = _np(model.norm.scale)              # [readout_H]
    p["dec_w"]            = _np(model.dec.weight)              # [out, readout_H]
    p["dec_b"]            = _np(model.dec.bias)                # [out]
    p["H"]                = int(config["neuron_dim"])

    # Pre-derived effective constants (matches v4 RSNN script)
    p["alpha_exc"]   = _sigmoid(p["tau_exc"]) * 0.99
    p["alpha_inh"]   = _sigmoid(p["tau_inh"]) * 0.99
    p["adapt_alpha"] = np.exp(-1.0 / p["tau_adapt_param"])
    p["U_full"]      = _sigmoid(p["U_logit"])
    p["tau_rec_full"]= _softplus(p["tau_rec_logit"]) + 1.0
    p["alpha_r"]     = 1.0 - (1.0 / p["tau_rec_full"])
    p["alpha_f"]     = np.exp(-1.0 / p["tau_facil_param"])
    p["W_eff"]       = p["W"] * p["mask"]
    return p


# -- NumPy nonlinearities (no torch dependency in RSNN path) ------------------

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _softplus(x):
    # Numerically stable: log(1+exp(x)) = max(x,0) + log1p(exp(-|x|))
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


# -- Dataset / seed inference for auto-loading test data ----------------------

DATASET_KEYWORDS = {
    "Heartbeat":            ["heart"],
    "EthanolConcentration": ["ethanol"],
    "SelfRegulationSCP1":   ["scp1"],
    "SelfRegulationSCP2":   ["scp2"],
    "MotorImagery":         ["motor"],
    "EigenWorms":           ["worms", "eigenworm"],
    "SpokenArabicDigits":   ["arabic", "spoken"],
}

DATASET_DISPLAY = {
    "Heartbeat":            "Heartbeat",
    "EthanolConcentration": "EthanolConc.",
    "SelfRegulationSCP1":   "SCP1",
    "SelfRegulationSCP2":   "SCP2",
    "MotorImagery":         "Motor",
    "EigenWorms":           "Worms",
    "SpokenArabicDigits":   "SpokenArabic",
}


def infer_dataset_from_path(pth_path, config=None):
    if config is not None:
        ds = config.get("dataset_name") or config.get("dataset")
        if ds:
            return ds
    tag = os.path.basename(pth_path).lower()
    for ds, kws in DATASET_KEYWORDS.items():
        if any(kw in tag for kw in kws):
            return ds
    return None


def infer_seed_from_path(pth_path, config=None):
    if config is not None and config.get("seed") is not None:
        return str(config["seed"])
    m = re.search(r"Seed(\d+)", os.path.basename(pth_path))
    return m.group(1) if m else None


def _load_pkl(path):
    with open(path, "rb") as f:
        obj = _safe_pickle_load(f)
    return np.array(obj.tolist()) if hasattr(obj, "tolist") else np.array(obj)


def load_test_data(config, pth_path, splits=("test",)):
    """
    Auto-detect data folder and load X_split.pkl / y_split.pkl for each split.

    Returns: dict { split: (X, y) }  or  (X, y) if a single split is requested.
    Returns None if the data folder cannot be found.
    """
    ds = infer_dataset_from_path(pth_path, config)
    seed = infer_seed_from_path(pth_path, config)
    if ds is None:
        return None
    expected_dim = config.get("input_dim") or config.get("in_features")

    pth_dir = os.path.dirname(os.path.abspath(pth_path))
    candidates = []
    if "data_root" in config:
        candidates.append(os.path.join(config["data_root"], ds, str(seed) if seed else ""))
    if "data_dir" in config:
        candidates.append(os.path.join(config["data_dir"], ds, str(seed) if seed else ""))

    roots = [
        os.path.join(pth_dir, "..", "datasets"),
        os.path.join(pth_dir, "..", "..", "datasets"),
        os.path.join(pth_dir, "..", "..", "..", "datasets"),
        os.path.join(PROJECT_DIR, "datasets"),
        os.path.join(PROJECT_DIR, "..", "datasets"),
        os.path.expanduser("~/datasets"),
    ]
    for root in roots:
        if seed:
            candidates.append(os.path.join(root, ds, seed))
        candidates.append(os.path.join(root, ds))

    data_dir = None
    for cand in candidates:
        if cand and os.path.isdir(cand) and os.path.isfile(os.path.join(cand, "X_test.pkl")):
            data_dir = cand
            break
    if data_dir is None and seed is None:
        # Try first available seed under the dataset
        for root in roots:
            base = os.path.join(root, ds)
            if os.path.isdir(base):
                subs = sorted(os.listdir(base))
                for s in subs:
                    if os.path.isfile(os.path.join(base, s, "X_test.pkl")):
                        data_dir = os.path.join(base, s)
                        break
                if data_dir:
                    break
    if data_dir is None:
        return None

    out = {}
    for split in splits:
        xp = os.path.join(data_dir, f"X_{split}.pkl")
        yp = os.path.join(data_dir, f"y_{split}.pkl")
        if not (os.path.isfile(xp) and os.path.isfile(yp)):
            out[split] = None
            continue
        X = _load_pkl(xp).astype(np.float32)
        y = _load_pkl(yp)
        if y.ndim == 2:
            y = np.argmax(y, axis=1)
        y = y.astype(np.int64)
        # Match the trainer: drop a single leading column if input_dim is off-by-one
        if expected_dim is not None and X.shape[-1] == expected_dim + 1:
            X = X[:, :, 1:]
        elif expected_dim is not None and X.shape[-1] > expected_dim:
            diff = X.shape[-1] - expected_dim
            X = X[:, :, diff:]
        out[split] = (X, y)

    if len(splits) == 1:
        return out[splits[0]]
    return out, data_dir

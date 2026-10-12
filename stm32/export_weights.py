# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright National Yang Ming Chiao Tung University, Taiwan.
# Creator Po-Han Chiang. Patent pending.
# Noncommercial use only. For commercial licensing see LICENSE.
"""export_weights.py
======================
Convert a PHCSSM checkpoint (.pth) to a C header that
embeds all RSNN sequential-deployment parameters as `static const float[]`
arrays for inclusion in the Arduino sketch.

Usage:
    python export_weights.py \\
        <path/to/grid_<ds>_H100_..._Seed<N>_..._eM.pth> \\
        -o src/weights/rsnn_weights_<ds>_seed<N>.h

The exported header defines the macros + arrays consumed by `rsnn_forward.cpp`
One header per (dataset, seed) checkpoint.

Notes
-----
* Uses the existing `loader.extract_rsnn_params` (no duplication).
* All weights are exported in FP32 (no quantization).
* readout_strategy is encoded as integer 0..4 (mean/last/sum/max/ssm).
* readout_source is 0 (voltage) or 1 (spike).
* W_eff is already mask-merged in extract_rsnn_params -> stored directly.
* Per-neuron tau parameters are converted to alpha_* form in
  extract_rsnn_params -> use the alpha_* fields directly (no further compute).
"""
from __future__ import annotations
import os, sys, argparse, re
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "rsnn"))

from loader import load_checkpoint, extract_rsnn_params


_STRAT_MAP = {"mean": 0, "last": 1, "sum": 2, "max": 3, "ssm": 4}
_SOURCE_MAP = {"voltage": 0, "spike": 1}


def _fmt_float(x: float) -> str:
    """ASCII-friendly FP32 literal (avoids 1e-300 underflow surprises)."""
    if np.isnan(x): return "nanf"
    if np.isposinf(x): return "INFINITY"
    if np.isneginf(x): return "(-INFINITY)"
    s = f"{float(x):.9g}"
    # Require either decimal point or exponent to make the C float literal
    # well-formed (e.g. "8" alone is int, but "8.0f" is float, and "8f" is
    # a parse error).
    if ("." not in s) and ("e" not in s) and ("E" not in s):
        s += ".0"
    # Ensure float suffix (so the C compiler picks FP32 not promoted double)
    if not s.endswith(("f", "F")):
        s += "f"
    return s


def _emit_1d_int(name: str, arr_list, indent: int = 0) -> str:
    """Emit a 1-D static const int array (used for slice indices)."""
    body = ", ".join(str(int(v)) for v in arr_list)
    pad = " " * indent
    return f"{pad}static const int {name}[{len(arr_list)}] = {{ {body} }};"


def _emit_1d(name: str, arr: np.ndarray, indent: int = 0) -> str:
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    body = ", ".join(_fmt_float(v) for v in arr.tolist())
    pad = " " * indent
    return f"{pad}static const float {name}[{arr.size}] = {{ {body} }};"


def _emit_2d(name: str, arr: np.ndarray, indent: int = 0) -> str:
    arr = np.asarray(arr, dtype=np.float32)
    assert arr.ndim == 2, f"expected 2-D for {name}, got shape {arr.shape}"
    rows, cols = arr.shape
    pad = " " * indent
    lines = [f"{pad}static const float {name}[{rows}][{cols}] = {{"]
    for r in range(rows):
        row_body = ", ".join(_fmt_float(v) for v in arr[r].tolist())
        comma = "," if r < rows - 1 else ""
        lines.append(f"{pad}  {{ {row_body} }}{comma}")
    lines.append(f"{pad}}};")
    return "\n".join(lines)


LICENSE_NOTICE = [
    "SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0",
    "Copyright National Yang Ming Chiao Tung University, Taiwan.",
    "Creator Po-Han Chiang. Patent pending.",
    "Noncommercial use only. For commercial licensing see LICENSE.",
]


def _parse_pth_filename(pth_path: str) -> dict:
    """Pull dataset, seed, epoch, val_acc, test_acc from filename for header."""
    fn = os.path.basename(pth_path)
    out = {"filename": fn}
    m = re.search(r"grid_(\w+)_H100", fn)
    if not m:
        # checkpoints written by this repository's train.py
        m = re.search(r"phcssm_(\w+?)_Seed", fn)
    if m: out["dataset"] = m.group(1)
    m = re.search(r"Seed(\d+)", fn)
    if m: out["seed"] = int(m.group(1))
    m = re.search(r"Seed\d+_([0-9.]+)_([0-9.]+)_e(\d+)\.pth$", fn)
    if m:
        out["val_acc"] = float(m.group(1))
        out["test_acc"] = float(m.group(2))
        out["epoch"] = int(m.group(3))
    return out


def export(pth_path: str, out_path: str, identifier: str = None):
    """Read checkpoint, emit C header to out_path."""
    print(f"Loading checkpoint: {pth_path}")
    model, config = load_checkpoint(pth_path)
    P = extract_rsnn_params(model, config)

    meta = _parse_pth_filename(pth_path)
    if identifier is None:
        ds = meta.get("dataset", "unknown")
        seed = meta.get("seed", 0)
        identifier = f"{ds}_seed{seed}"

    H = int(P["H"])
    enc_w = np.asarray(P["enc_w"], dtype=np.float32)
    D_in = int(enc_w.shape[1])
    dec_w = np.asarray(P["dec_w"], dtype=np.float32)
    N_class, H_prime = int(dec_w.shape[0]), int(dec_w.shape[1])
    delay = int(P["delay"])
    drive_alpha = float(P["drive_alpha"])
    beta_adapt = float(P["beta_adapt"])
    facil_amp = float(P["facil_amp"])
    src = _SOURCE_MAP[P["readout_source"]]
    strat = _STRAT_MAP[P["readout_strategy"]]

    slices = list(P["readout_slices"])
    n_slices = len(slices)

    print(f"  identifier={identifier}, H={H}, D_in={D_in}, "
          f"N_class={N_class}, H_prime={H_prime}, delay={delay}")
    print(f"  drive_alpha={drive_alpha}, beta_adapt={beta_adapt}, "
          f"facil_amp={facil_amp}")
    print(f"  readout_source={P['readout_source']}({src}), "
          f"strategy={P['readout_strategy']}({strat}), slices={slices}")

    guard = f"RSNN_WEIGHTS_{identifier.upper()}_H"

    lines = []
    for _notice in LICENSE_NOTICE:
        lines.append(f"// {_notice}")
    lines.append("")
    lines.append(f"// Auto-generated by export_weights.py")
    lines.append(f"// Source: {meta.get('filename', pth_path)}")
    if "epoch" in meta:
        lines.append(f"// Epoch: {meta['epoch']}, val_acc: {meta.get('val_acc')}, "
                     f"test_acc: {meta.get('test_acc')}")
    lines.append(f"// PHCSSM 1.0.0 RSNN sequential deployment weights, FP32")
    lines.append("")
    lines.append(f"#ifndef {guard}")
    lines.append(f"#define {guard}")
    lines.append("")
    lines.append("#include <math.h>")
    lines.append("")
    lines.append(f'#define RSNN_VARIANT_NAME    "{identifier}"')
    lines.append(f"#define H                    {H}")
    lines.append(f"#define D_IN                 {D_in}")
    lines.append(f"#define N_CLASS              {N_class}")
    lines.append(f"#define HPRIME               {H_prime}")
    lines.append(f"#define DELAY                {delay}")
    lines.append(f"#define DRIVE_ALPHA          {_fmt_float(drive_alpha)}")
    lines.append(f"#define BETA_ADAPT           {_fmt_float(beta_adapt)}")
    lines.append(f"#define FACIL_AMP            {_fmt_float(facil_amp)}")
    lines.append(f"#define READOUT_SOURCE       {src}    // 0=voltage, 1=spike")
    lines.append(f"#define READOUT_STRATEGY     {strat}    // 0=mean,1=last,2=sum,3=max,4=ssm")
    lines.append(f"#define N_SLICES             {n_slices}")
    lines.append("")

    if n_slices > 0:
        starts = [int(s) for s, _ in slices]
        ends = [int(e) for _, e in slices]
        lines.append(_emit_1d_int("SLICE_START", starts))
        lines.append(_emit_1d_int("SLICE_END", ends))
        lines.append("")

    # 2-D arrays
    lines.append(_emit_2d("ENC_W", enc_w))               # [H, D_in]
    lines.append(_emit_2d("W_EFF", P["W_eff"]))          # [H, H]
    lines.append(_emit_2d("DEC_W", dec_w))               # [N_class, H_prime]

    # 1-D arrays
    one_d_keys = [
        ("ENC_B",        P["enc_b"]),
        ("LN_W",         P["ln_w"]),
        ("LN_B",         P["ln_b"]),
        ("INPUT_MASK",   P["input_mask"]),
        ("U_FULL",       P["U_full"]),
        ("ALPHA_F",      P["alpha_f"]),
        ("ALPHA_R",      P["alpha_r"]),
        ("ALPHA_EXC",    P["alpha_exc"]),
        ("ALPHA_INH",    P["alpha_inh"]),
        ("ADAPT_ALPHA",  P["adapt_alpha"]),
        ("V_TH",         P["v_th"]),
        ("RESET_WEIGHT", P["reset_weight"]),
        ("DEC_B",        P["dec_b"]),
    ]
    for name, arr in one_d_keys:
        lines.append(_emit_1d(name, arr))

    if strat == 4:  # ssm - needs norm_w
        lines.append(_emit_1d("NORM_W", P["norm_w"]))

    lines.append("")
    lines.append(f"#endif  // {guard}")
    lines.append("")

    body = "\n".join(lines)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(body)

    file_size = os.path.getsize(out_path)
    print(f"  Wrote: {out_path} ({file_size:,} bytes)")
    # Estimate runtime weight memory (FP32):
    n_floats = (H * D_in + H * H + N_class * H_prime
                + 3 * H + H + H + 9 * H + N_class)
    if strat == 4:
        n_floats += H_prime
    print(f"  Estimated weight memory: {n_floats * 4 / 1024:.1f} KB FP32")


def cli():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pth", help="Path to .pth checkpoint")
    ap.add_argument("-o", "--out", required=True,
                    help="Output .h path")
    ap.add_argument("--identifier", default=None,
                    help="Override identifier (default: derive from filename)")
    args = ap.parse_args()
    export(args.pth, args.out, identifier=args.identifier)


if __name__ == "__main__":
    cli()

"""Load a dataset configuration and reject keys the code never reads.

An unrecognised key is a silent reproducibility failure: the model and the
trainer read their settings with ``config.get(name, default)``, so a misspelled
or dropped key falls back to a default instead of raising. KNOWN_KEYS is the
complete set of names read by model.py and trainer.py.
"""
import ast
import json

KNOWN_KEYS = frozenset([
    # read by trainer.py
    "_config_filename", "batch_size", "data_root", "dataset_name",
    "early_stopping_patience", "epochs", "execution_mode", "grad_accum_steps",
    "grad_clip", "loss_threshold_for_patience", "lr_max",
    "min_acc_for_patience", "patience_trigger", "quiet", "rate_loss_weight",
    "save_dir", "save_threshold", "spike_target", "tbptt_loss_strategy",
    "tbptt_window_size", "voltage_loss_weight", "weight_decay",
    # read by model.py
    "beta_adapt", "chunk_size", "convergence_loss_weight", "drive_alpha",
    "dropout", "facil_amp", "hebbian_mode", "hebbian_rule",
    "hierarchy_matrix", "input_target_types", "lr_hebb", "num_regions",
    "num_transmissions", "output_dim", "overlap", "partition_region_ratio",
    "readout_source", "readout_strategy", "readout_target_types",
    "readout_weight_decay", "readout_weight_type", "region_ei_ratios",
    "stdp_A_minus", "stdp_A_plus", "stdp_tau_minus", "stdp_tau_plus",
    "synaptic_delay", "tau_adapt", "tau_facil", "v_noise_std",
    # read by both
    "hebbian_rule", "input_dim", "neuron_dim",
])

REQUIRED_KEYS = frozenset([
    "dataset_name", "input_dim", "output_dim", "neuron_dim", "lr_max",
    "batch_size", "epochs", "early_stopping_patience", "num_transmissions",
    "readout_source", "hierarchy_matrix", "num_regions",
    "partition_region_ratio", "region_ei_ratios", "input_target_types",
    "readout_target_types", "grad_clip", "weight_decay",
])


def parse_overrides(pairs):
    """Turn ["neuron_dim=32", "lr_max=1e-4"] into a dict of Python values."""
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--set expects key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        try:
            out[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            out[key] = value
    return out


def validate(config):
    unknown = sorted(set(config) - KNOWN_KEYS)
    if unknown:
        raise ValueError(
            "configuration sets keys that neither model.py nor trainer.py "
            f"reads: {', '.join(unknown)}")
    missing = sorted(REQUIRED_KEYS - set(config))
    if missing:
        raise ValueError(f"configuration is missing: {', '.join(missing)}")
    return config


def load(path, data_root, overrides=()):
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    config.update(parse_overrides(overrides))
    config["data_root"] = data_root
    return validate(config)

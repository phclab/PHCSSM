"""Training and evaluation loop.

Parameters are selected on validation accuracy and the test split is scored
once, with the parameters of the best-validation epoch. Pass
evaluate_test=False to skip that scoring, which is what configuration
selection does.
"""

import os
os.environ.setdefault("XLA_FLAGS",
    "--xla_gpu_deterministic_ops=true --xla_gpu_enable_triton_gemm=false")

import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import numpy as np
import os, sys, logging, csv, time, datetime, random, pickle
from functools import partial
from tqdm import tqdm

from . import model as phc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_backends = jax.devices()
logger.info(f"JAX devices: {_backends}")


def cross_entropy_smooth(logits, labels, smoothing=0.1):
    n = logits.shape[-1]
    one_hot = jax.nn.one_hot(labels, n)
    soft_labels = one_hot * (1.0 - smoothing) + smoothing / n
    log_probs = jax.nn.log_softmax(logits)
    return -(soft_labels * log_probs).sum(axis=-1).mean()


def _safe_pickle_load(f):
    """Load pickle that may have been saved with older JAX."""
    import inspect
    import jax._src.core as _jax_core
    _orig_update = _jax_core.ShapedArray.update
    _valid = set(inspect.signature(_jax_core.ShapedArray.__init__).parameters) - {"self"}
    def _patched_update(self, **kwargs):
        return _orig_update(self, **{k: v for k, v in kwargs.items() if k in _valid})
    _jax_core.ShapedArray.update = _patched_update
    try:
        return pickle.load(f)
    finally:
        _jax_core.ShapedArray.update = _orig_update


def load_uea_pretreated_jax(config, dataset_name, seed):
    """Load data directly as JAX arrays on default device."""
    base_path = os.path.join(config["data_root"], dataset_name, str(seed))

    def load_pkl_pair(subset):
        X_file = os.path.join(base_path, f"X_{subset}.pkl")
        y_file = os.path.join(base_path, f"y_{subset}.pkl")
        with open(X_file, "rb") as f:
            x_data = _safe_pickle_load(f)
            x_data = np.array(x_data.tolist())
            if x_data.shape[-1] == config["input_dim"] + 1:
                x_data = x_data[:, :, 1:]
        with open(y_file, "rb") as f:
            y_data = _safe_pickle_load(f)
            y_data = np.array(y_data.tolist())
            y_data = np.argmax(y_data, axis=1)
        return jnp.array(x_data, dtype=jnp.float32), jnp.array(y_data, dtype=jnp.int32)

    train_data = load_pkl_pair("train")
    val_data = load_pkl_pair("val")
    test_data = load_pkl_pair("test")
    return train_data, val_data, test_data


def load_uea_pretreated(config, dataset_name, seed):
    """Backward-compatible alias: returns 6 separate arrays for run_parallel scripts."""
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = load_uea_pretreated_jax(config, dataset_name, seed)
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def make_batches(X, Y, batch_size, key=None, shuffle=False):
    """Yield (X_batch, Y_batch) slices. Data stays on device."""
    n = X.shape[0]
    if shuffle and key is not None:
        perm = jax.random.permutation(key, n)
        X = X[perm]
        Y = Y[perm]
    num_batches = (n + batch_size - 1) // batch_size
    for i in range(num_batches):
        s = i * batch_size
        e = min(s + batch_size, n)
        yield X[s:e], Y[s:e]


def count_batches(n, batch_size):
    return (n + batch_size - 1) // batch_size


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def make_compute_grads_fn(mode, spike_target, rate_lw, volt_lw):
    """Create a JIT'd function for forward + loss + grad computation."""

    @eqx.filter_jit
    def compute_grads(params, static, bx, by, key):
        def loss_fn(params):
            model = eqx.combine(params, static)
            drop_key, aug_key1, aug_key2, noise_key = jax.random.split(key, 4)

            bx_aug = bx + jax.random.normal(aug_key1, bx.shape) * 0.05
            bx_aug = bx_aug + jax.random.normal(aug_key2, bx_aug.shape) * 0.05

            if mode in ("osp", "osp+tbptt"):
                logits, all_spikes, all_voltages, _ = model.forward(
                    bx_aug, key=drop_key, inference=False, mode=mode)
            else:
                logits, all_spikes, all_voltages = model(
                    bx_aug, key=drop_key, inference=False)

            ce_loss = cross_entropy_smooth(logits, by, smoothing=0.1)

            # all_spikes: [num_trans, B, T, H]
            layer_rates = all_spikes.mean(axis=(1, 2, 3))  # [num_trans]
            rate_loss = jnp.mean((layer_rates - spike_target) ** 2)

            layer_volts = (all_voltages ** 2).mean(axis=(1, 2, 3))  # [num_trans]
            voltage_loss = jnp.mean(layer_volts)

            # v6: transmission convergence loss
            conv_loss = model.compute_convergence_loss(all_spikes)
            conv_lw = model.convergence_loss_weight
            eff_steps = model.compute_effective_steps(all_spikes)

            total_loss = (ce_loss + rate_lw * rate_loss + volt_lw * voltage_loss
                          + conv_lw * conv_loss)
            return total_loss, (logits, all_spikes, ce_loss,
                                layer_rates.mean(), conv_loss, eff_steps)

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        return loss, aux, grads

    return compute_grads


@eqx.filter_jit
def apply_update(params, grads, opt_state, static, optimizer, region_slices, filter_spec):
    """Optimizer step + Dale's law."""
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    # Apply Dale's law
    model = eqx.combine(new_params, static)
    model = phc.apply_dale(model, region_slices)
    new_params, _ = eqx.partition(model, filter_spec)
    return new_params, new_opt_state


def make_tbptt_grads_fn(tbptt_mode, spike_target, rate_lw, volt_lw):
    """Create JIT'd function for one TBPTT window (forward + loss + grad).
    tbptt_mode: 'tbptt' uses forward_stateful, 'osp+tbptt' uses parallel_forward_stateful.
    """

    @eqx.filter_jit
    def compute_window_grads(params, static, bx_slice, by, key,
                             prev_n_state, prev_s_state, seg_weight):
        def loss_fn(params):
            model = eqx.combine(params, static)
            drop_key, aug_key1, aug_key2 = jax.random.split(key, 3)

            bx_aug = bx_slice + jax.random.normal(aug_key1, bx_slice.shape) * 0.05
            bx_aug = bx_aug + jax.random.normal(aug_key2, bx_aug.shape) * 0.05

            if tbptt_mode in ("osp+tbptt", "tbptt+osp"):
                logits, all_spikes, all_voltages, _, final_n, final_s = \
                    model.parallel_forward_stateful(
                        bx_aug, prev_n_state, prev_s_state,
                        key=drop_key, inference=False)
            else:  # tbptt
                logits, all_spikes, all_voltages, final_n, final_s = \
                    model.forward_stateful(
                        bx_aug, prev_n_state, prev_s_state,
                        key=drop_key, inference=False)

            ce_loss = cross_entropy_smooth(logits, by, smoothing=0.1)
            layer_rates = all_spikes.mean(axis=(1, 2, 3))
            rate_loss = jnp.mean((layer_rates - spike_target) ** 2)
            layer_volts = (all_voltages ** 2).mean(axis=(1, 2, 3))
            voltage_loss = jnp.mean(layer_volts)

            # v6: transmission convergence loss
            conv_loss = model.compute_convergence_loss(all_spikes)
            conv_lw = model.convergence_loss_weight
            eff_steps = model.compute_effective_steps(all_spikes)

            # seg_weight applies to the cross-entropy term only
            total_loss = (seg_weight * ce_loss + rate_lw * rate_loss
                          + volt_lw * voltage_loss + conv_lw * conv_loss)
            return total_loss, (logits, all_spikes, ce_loss, layer_rates.mean(),
                                conv_loss, eff_steps, final_n, final_s)

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        logits, all_spikes, ce_loss, avg_rate, conv_loss, eff_steps, final_n, final_s = aux
        return (loss, (logits, all_spikes, ce_loss, avg_rate, conv_loss, eff_steps),
                grads, final_n, final_s)

    return compute_window_grads


def make_infer_fn(mode):
    """Create a JIT'd inference function."""

    @eqx.filter_jit
    def infer(model, bx):
        if mode in ("osp", "osp+tbptt"):
            logits, _, _, _ = model.forward(bx, inference=True, mode=mode)
        else:
            logits, _, _ = model(bx, inference=True)
        return logits

    return infer


def make_infer_with_spikes_fn(mode):
    """Like make_infer_fn but also returns all_spikes (for test mstep)."""

    @eqx.filter_jit
    def infer(model, bx):
        if mode in ("osp", "osp+tbptt"):
            logits, all_spikes, _, _ = model.forward(bx, inference=True, mode=mode)
        else:
            logits, all_spikes, _ = model(bx, inference=True)
        return logits, all_spikes

    return infer


def train(config, seed, train_data, val_data, test_data, evaluate_test=True):
    set_seed(seed)
    X_train, Y_train = train_data
    X_val, Y_val = val_data
    X_test, Y_test = test_data if test_data is not None else (None, None)

    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    ds_name = config.get("dataset_name", "?")
    tag = f"[GPU{gpu_id}|{ds_name}|S{seed}]"
    logger.info(f"{tag} PHCSSM {phc.__version__} | JAX devices: {jax.devices()}")

    # Model
    h = config["neuron_dim"]
    topo_mask_np = phc.build_topology_mask(config)
    actual_conn = int(topo_mask_np.sum())
    total_conn = h * h
    print(f"\n========== Synapse Layer Connectivity Stats ==========")
    print(f"Total Neurons: {h}")
    print(f"Possible Connections: {total_conn:,}")
    print(f"Actual Connections: {actual_conn:,}")
    print(f"Connection Density: {actual_conn/total_conn:.2%}")
    print(f"======================================================\n")

    key = jax.random.PRNGKey(seed)
    key, model_key = jax.random.split(key)
    model = phc.create_phcssm(config, model_key)

    # Partition: params (trainable) vs static (non-trainable + structure)
    filter_spec = phc.make_param_filter(model)
    params, static = eqx.partition(model, filter_spec)

    # Optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(config["grad_clip"]),
        optax.adamw(learning_rate=config["lr_max"],
                    weight_decay=config["weight_decay"]),
    )
    opt_state = optimizer.init(params)

    # Pre-compute static helpers
    region_slices = phc.calculate_region_slices(config)
    topo_mask = jnp.array(topo_mask_np)
    hebb_mask = phc.build_hebb_mask(config, region_slices, topo_mask) if config.get("hebbian_rule") else None

    # Config values
    grad_accum_steps = config.get("grad_accum_steps", 1)
    patience = config.get("early_stopping_patience", 100)
    mode = config.get("execution_mode", "nothing")
    spike_target = config.get("spike_target", 0.2)
    rate_lw = config.get("rate_loss_weight", 1.0)
    volt_lw = config.get("voltage_loss_weight", 0.1)
    batch_size = config["batch_size"]

    # Patience trigger
    patience_trigger = config.get("patience_trigger", "acc")
    min_acc_threshold = config.get("min_acc_for_patience", 0.0)
    loss_threshold = config.get("loss_threshold_for_patience", 0.0)
    patience_activated = False

    # TBPTT config
    is_tbptt = mode in ("tbptt", "osp+tbptt", "tbptt+osp")
    tbptt_window_size = config.get("tbptt_window_size", 4096)
    tbptt_loss_strategy = config.get("tbptt_loss_strategy", "increasing")

    # JIT'd functions (created once, reused)
    compute_grads = make_compute_grads_fn(mode, spike_target, rate_lw, volt_lw)
    if is_tbptt:
        compute_window_grads = make_tbptt_grads_fn(mode, spike_target, rate_lw, volt_lw)
    infer_fn = make_infer_fn(mode)
    infer_with_spikes_fn = make_infer_with_spikes_fn(mode)

    # Tracking
    best_acc = 0.0
    best_vacc_es = 0.0
    best_eff_steps = float("nan")  # avg_eff_steps at the best-acc epoch
    best_epoch = 0
    best_params_state = None
    best_static_state = None
    no_improve = 0
    epoch_log = []
    global_start = time.time()

    n_train = X_train.shape[0]
    num_batches = count_batches(n_train, batch_size)

    # Training loop
    quiet = config.get("quiet", False)  # parallel mode: disable progress bar
    log = print if quiet else tqdm.write  # quiet: print lines; normal: write above bar
    epoch_pbar = tqdm(range(config["epochs"]), desc=tag, unit="ep",
                      leave=True, dynamic_ncols=True, disable=quiet)
    for epoch in epoch_pbar:
        epoch_start = time.time()

        # Accumulate on-device to avoid per-batch CPU-GPU sync
        total_loss_ep = jnp.float32(0.0)
        spike_stats = jnp.float32(0.0)
        conv_loss_acc = jnp.float32(0.0)
        eff_steps_acc = jnp.float32(0.0)
        num_chunks = 0
        ltp_acc_dev = jnp.int32(0)
        ltd_acc_dev = jnp.int32(0)
        hebb_mag_dev = jnp.float32(0.0)
        hebb_ratio_dev = jnp.float32(0.0)
        hebb_count = 0

        acc_grads = None
        accum_count = 0
        last_all_spikes = None

        key, shuffle_key = jax.random.split(key)

        for batch_idx, (bx, by) in enumerate(
                make_batches(X_train, Y_train, batch_size, shuffle_key, shuffle=True)):

            key, step_key = jax.random.split(key)

            if epoch == 0 and batch_idx == 0:
                log(f"{tag} XLA compiling... (first batch, mode={mode})")
                _compile_t0 = time.time()

            # Mode dispatch
            if is_tbptt:
                # TBPTT: split sequence into windows, grad per window, detach states
                B_cur, L_cur, D_cur = bx.shape
                window_size = tbptt_window_size
                num_segments = (L_cur + window_size - 1) // window_size
                current_seg = 0
                prev_n_state, prev_s_state = None, None

                for start in range(0, L_cur, window_size):
                    end = min(start + window_size, L_cur)
                    bx_slice = bx[:, start:end, :]
                    current_seg += 1

                    if tbptt_loss_strategy == "increasing":
                        seg_weight = jnp.float32(current_seg / num_segments)
                    else:
                        seg_weight = jnp.float32(1.0)

                    key, win_key = jax.random.split(key)
                    loss, (logits, all_spikes, ce_loss, avg_rate,
                           conv_loss, eff_steps), grads, \
                        final_n, final_s = compute_window_grads(
                            params, static, bx_slice, by, win_key,
                            prev_n_state, prev_s_state, seg_weight)

                    # Detach states for next window (stop_gradient)
                    prev_n_state = jax.tree_util.tree_map(jax.lax.stop_gradient, final_n)
                    prev_s_state = jax.tree_util.tree_map(jax.lax.stop_gradient, final_s)

                    # Accumulate grads (scaled by grad_accum_steps)
                    scaled_grads = jax.tree_util.tree_map(
                        lambda g: g / grad_accum_steps, grads)
                    if acc_grads is None:
                        acc_grads = scaled_grads
                    else:
                        acc_grads = jax.tree_util.tree_map(
                            lambda a, g: a + g, acc_grads, scaled_grads)

                    total_loss_ep = total_loss_ep + loss
                    spike_stats = spike_stats + avg_rate
                    conv_loss_acc = conv_loss_acc + conv_loss
                    eff_steps_acc = eff_steps_acc + eff_steps
                    num_chunks += 1
                    last_all_spikes = all_spikes

                    # Hebbian update per window
                    if config.get("hebbian_rule") and last_all_spikes is not None:
                        sg_spikes = jax.lax.stop_gradient(last_all_spikes)
                        model_tmp = eqx.combine(params, static)
                        w_grad = grads.shared_sl.weight if hasattr(grads, 'shared_sl') else None
                        model_tmp, hebb_info = phc.hebbian_update(
                            model_tmp, sg_spikes, hebb_mask, config, region_slices,
                            weight_grad=w_grad)
                        if hebb_info is not None:
                            ltp_acc_dev = ltp_acc_dev + hebb_info["ltp_count"]
                            ltd_acc_dev = ltd_acc_dev + hebb_info["ltd_count"]
                            hebb_mag_dev = hebb_mag_dev + hebb_info["hebb_mag"]
                            hebb_ratio_dev = hebb_ratio_dev + hebb_info["ratio"]
                            hebb_count += 1
                            params, _ = eqx.partition(model_tmp, filter_spec)

                accum_count += 1

            else:
                # Standard mode: nothing / osp
                loss, (logits, all_spikes, ce_loss, avg_rate,
                       conv_loss, eff_steps), grads = \
                    compute_grads(params, static, bx, by, step_key)

                scaled_grads = jax.tree_util.tree_map(
                    lambda g: g / grad_accum_steps, grads)
                if acc_grads is None:
                    acc_grads = scaled_grads
                else:
                    acc_grads = jax.tree_util.tree_map(
                        lambda a, g: a + g, acc_grads, scaled_grads)

                accum_count += 1
                total_loss_ep = total_loss_ep + loss
                spike_stats = spike_stats + avg_rate
                conv_loss_acc = conv_loss_acc + conv_loss
                eff_steps_acc = eff_steps_acc + eff_steps
                num_chunks += 1
                last_all_spikes = all_spikes

                # Hebbian update
                if config.get("hebbian_rule") and last_all_spikes is not None:
                    sg_spikes = jax.lax.stop_gradient(last_all_spikes)
                    model_tmp = eqx.combine(params, static)
                    w_grad = grads.shared_sl.weight if hasattr(grads, 'shared_sl') else None
                    model_tmp, hebb_info = phc.hebbian_update(
                        model_tmp, sg_spikes, hebb_mask, config, region_slices,
                        weight_grad=w_grad)
                    if hebb_info is not None:
                        ltp_acc_dev = ltp_acc_dev + hebb_info["ltp_count"]
                        ltd_acc_dev = ltd_acc_dev + hebb_info["ltd_count"]
                        hebb_mag_dev = hebb_mag_dev + hebb_info["hebb_mag"]
                        hebb_ratio_dev = hebb_ratio_dev + hebb_info["ratio"]
                        hebb_count += 1
                        params, _ = eqx.partition(model_tmp, filter_spec)

            if epoch == 0 and batch_idx == 0:
                jax.effects_barrier()
                log(f"{tag} XLA compile done in {time.time() - _compile_t0:.1f}s")

            # Apply gradient update
            is_last_batch = (batch_idx + 1 == num_batches)
            if accum_count % grad_accum_steps == 0 or is_last_batch:
                params, opt_state = apply_update(
                    params, acc_grads, opt_state, static, optimizer,
                    region_slices, filter_spec)
                acc_grads = None
                accum_count = 0

        epoch_dur = time.time() - epoch_start
        cps = num_chunks / (epoch_dur + 1e-8)
        # Single sync point per epoch: pull accumulated losses from GPU
        avg_tr_loss = float(total_loss_ep) / max(num_chunks, 1)
        avg_spike_rate = float(spike_stats) / max(num_chunks, 1)
        avg_conv_loss = float(conv_loss_acc) / max(num_chunks, 1)
        avg_eff_steps = float(eff_steps_acc) / max(num_chunks, 1)

        val_correct_dev = jnp.int32(0)
        val_total_dev = jnp.int32(0)
        val_loss_dev = jnp.float32(0.0)
        val_batches = 0
        model_eval = eqx.combine(params, static)
        for bx, by in make_batches(X_val, Y_val, batch_size):
            logits = infer_fn(model_eval, bx)
            val_loss_dev = val_loss_dev + cross_entropy_smooth(logits, by, smoothing=0.0)
            val_batches += 1
            preds = jnp.argmax(logits, axis=-1)
            val_correct_dev = val_correct_dev + (preds == by).sum()
            val_total_dev = val_total_dev + by.shape[0]

        # Single sync point for validation
        val_correct = int(val_correct_dev)
        val_total = int(val_total_dev)
        val_loss_sum = float(val_loss_dev)
        vacc = val_correct / max(val_total, 1)
        avg_v_loss = val_loss_sum / max(val_batches, 1)


        if not patience_activated:
            if patience_trigger == "loss" and avg_tr_loss <= loss_threshold:
                patience_activated = True
                log(f"\n{tag} Epoch {epoch+1}: Loss {avg_tr_loss:.4f} <= {loss_threshold} -> ES activated\n")
            elif patience_trigger == "acc" and best_acc >= min_acc_threshold:
                patience_activated = True
                log(f"\n{tag} Epoch {epoch+1}: Acc {vacc*100:.2f}% >= {min_acc_threshold*100:.0f}% -> ES activated\n")

        if vacc > best_vacc_es:
            best_vacc_es = vacc
            no_improve = 0
        elif patience_activated:
            no_improve += 1
        else:
            no_improve = 0

        ltp_acc = int(ltp_acc_dev) if config.get("hebbian_rule") else 0
        ltd_acc = int(ltd_acc_dev) if config.get("hebbian_rule") else 0
        avg_hebb_mag = float(hebb_mag_dev) / max(hebb_count, 1) if config.get("hebbian_rule") else 0.0
        avg_hebb_ratio = float(hebb_ratio_dev) / max(hebb_count, 1) if config.get("hebbian_rule") else 0.0

        # Track best params at each new best validation score, but defer the
        # test evaluation and the .pth write to the end of training, which
        # avoids re-running the test evaluation on every improvement.
        if vacc > best_acc and vacc > config.get("save_threshold", 0.0):
            best_acc = vacc
            best_eff_steps = float(avg_eff_steps)
            best_epoch = epoch + 1
            best_params_state = jax.tree_util.tree_map(np.array, params)
            best_static_state = jax.tree_util.tree_map(
                lambda x: np.array(x) if isinstance(x, jnp.ndarray) else x, static)
            time_to_best = time.time() - global_start
            time_str_to_best = str(datetime.timedelta(seconds=int(time_to_best)))

            hebb_str = ""
            if config.get("hebbian_rule"):
                hebb_str = f" | H/G:{avg_hebb_ratio:.3f} LTP:{ltp_acc} LTD:{ltd_acc}"
            log(
                f"{tag} NEW BEST E{epoch+1} | "
                f"Val:{best_acc*100:.2f}% | "
                f"TrL:{avg_tr_loss:.4f} VL:{avg_v_loss:.4f} | "
                f"Rate:{avg_spike_rate:.3f} "
                f"Conv:{avg_conv_loss:.4f} "
                f"Mstep:{avg_eff_steps:.1f}/{config['num_transmissions']} | "
                f"{epoch_dur:.1f}s | TTB:{time_str_to_best}{hebb_str}")

        # progress bar; everything goes in the postfix
        hebb_pf = ""
        if config.get("hebbian_rule"):
            hebb_pf = f" H/G:{avg_hebb_ratio:.3f}"
        epoch_pbar.set_postfix_str(
            f"L:{avg_tr_loss:.3f}/{avg_v_loss:.3f} "
            f"A:{vacc*100:.1f}/{best_acc*100:.1f} "
            f"R:{avg_spike_rate:.2f} "
            f"M:{avg_eff_steps:.1f} "
            f"NI:{no_improve}/{patience}{hebb_pf}")

        epoch_log.append({
            "epoch": epoch + 1,
            "train_loss": round(avg_tr_loss, 6),
            "val_loss": round(avg_v_loss, 6),
            "val_acc": round(vacc * 100, 4),
            "best_acc": round(best_acc * 100, 4),
            "spike_rate": round(avg_spike_rate, 6),
            "conv_loss": round(avg_conv_loss, 6),
            "effective_steps": round(avg_eff_steps, 2),
            "epoch_time_sec": round(epoch_dur, 3),
            "speed_chunks_per_sec": round(cps, 2),
            "no_improve": no_improve,
            "ltp_count": ltp_acc,
            "ltd_count": ltd_acc,
            "hebb_grad_ratio": round(avg_hebb_ratio, 6) if config.get("hebbian_rule") else None,
            "cumulative_time_sec": round(time.time() - global_start, 3),
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

        if no_improve >= patience:
            log(f"{tag} Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            epoch_pbar.close()
            break

        if vacc >= 1.0:
            log(f"{tag} Val acc 100% at epoch {epoch+1}, stopping (perfect validation)")
            epoch_pbar.close()
            break

    # Final evaluation on test set. This runs after the training loop has
    # finished, so skipping it cannot change training.
    if best_params_state is not None:
        params = jax.tree_util.tree_map(jnp.array, best_params_state)
    model_final = eqx.combine(params, static)

    final_acc = None
    test_eff_steps = None
    if evaluate_test:
        # capture mstep on test data using best-checkpoint params
        t_correct_dev = jnp.int32(0)
        t_total_dev = jnp.int32(0)
        test_eff_steps_acc = jnp.float32(0.0)
        test_batches = 0
        for bx, by in make_batches(X_test, Y_test, batch_size):
            lgt, all_spikes = infer_with_spikes_fn(model_final, bx)
            preds = jnp.argmax(lgt, axis=-1)
            t_correct_dev = t_correct_dev + (preds == by).sum()
            t_total_dev = t_total_dev + by.shape[0]
            test_eff_steps_acc = test_eff_steps_acc + model_final.compute_effective_steps(all_spikes)
            test_batches += 1
        final_acc = int(t_correct_dev) / max(int(t_total_dev), 1)
        test_eff_steps = float(test_eff_steps_acc) / max(test_batches, 1)

    # Save the best checkpoint
    if best_params_state is not None:
        config_tag = config.get("_config_filename",
                                f"PHCSSM_{phc.__version__}_{config.get('dataset_name','?')}")
        score = (f"{best_acc*100:.2f}" if final_acc is None
                 else f"{best_acc*100:.2f}_{final_acc*100:.2f}")
        pth_filename = f"{config_tag}_Seed{seed}_{score}_e{best_epoch}.pth"
        save_dir = config.get("save_dir", "")
        save_path = os.path.join(save_dir, pth_filename) if save_dir else pth_filename
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump({
                'params': best_params_state,
                'static': best_static_state,
                'config': config,
                'acc': best_acc,
                'test_acc': final_acc,
                'epoch': best_epoch,
                'mstep_train': best_eff_steps,
                'mstep_test': test_eff_steps,
                '__version__': phc.__version__,
            }, f)
        print(f"{tag} Final .pth saved: {save_path}")

    # Save epoch log CSV
    if epoch_log:
        config_tag = config.get("_config_filename",
                                f"{config.get('dataset_name','?')}_{config.get('num_regions','?')}R{config.get('neuron_dim','?')}")
        csv_filename = f"{config_tag}_Seed{seed}_epoch_log.csv"
        save_dir = config.get("save_dir", "")
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            csv_filename = os.path.join(save_dir, csv_filename)
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(epoch_log[0].keys()))
            writer.writeheader()
            writer.writerows(epoch_log)
        print(f"{tag} Epoch log saved: {csv_filename}")

    return final_acc, best_acc, best_eff_steps, test_eff_steps

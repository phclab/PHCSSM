"""PHCSSM: the parallelised hierarchical connectome and its two layers.

The K-transmission loop is a fixed-point solver over the within-timestep
iteration index, not an evolution in time. Every iteration recomputes the whole
timeline from the same initial state and only the loop input accumulates
synaptic feedback, so the final iteration settles on the trajectory a
sequentially executed recurrent spiking network produces in one pass.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from typing import NamedTuple, Optional, Tuple, List
from functools import partial

__version__ = "1.0.0"
__author__ = "Po-Han Chiang"


@jax.custom_vjp
def spike_fn(x):
    return (x > 0).astype(x.dtype)

def _spike_fwd(x):
    return spike_fn(x), x

def _spike_bwd(x_saved, g):
    alpha = 4.0
    return (g / (alpha * jnp.abs(x_saved) + 1.0) ** 2,)

spike_fn.defvjp(_spike_fwd, _spike_bwd)


def logcumsumexp(x, axis: int = 1):
    def log_add(a, b):
        return jnp.logaddexp(a, b)
    return jax.lax.associative_scan(log_add, x, axis=axis)


def scan_decay(x, alpha, h_init=None):
    log_alpha = jnp.log(jnp.clip(alpha, 1e-8, None))
    S_decay = jnp.cumsum(jnp.broadcast_to(log_alpha, x.shape), axis=1)
    log_sp_x = jnp.log(jax.nn.softplus(x) + 1e-8)
    z = log_sp_x - S_decay
    lcs = logcumsumexp(z, axis=1)
    out = jnp.exp(lcs + S_decay)
    if h_init is not None:
        out = out + h_init * jnp.exp(S_decay)
    return out


def calculate_region_slices(config: dict) -> list:
    dim = config["neuron_dim"]
    slices = []
    start = 0
    for i, r in enumerate(config["partition_region_ratio"]):
        end = dim if i == len(config["partition_region_ratio"]) - 1 else start + int(dim * r)
        split = start + int((end - start) * config["region_ei_ratios"][i])
        slices.append((start, split, end))
        start = end
    return slices


def build_topology_mask(config: dict) -> np.ndarray:
    h = config["neuron_dim"]
    mask = np.zeros((h, h), dtype=np.float32)
    slices = calculate_region_slices(config)
    for i in range(config["num_regions"]):
        for j in range(config["num_regions"]):
            ee, ei, ie, ii = config["hierarchy_matrix"][i][j]
            d_s, d_split, d_e = slices[i]
            s_s, s_split, s_e = slices[j]
            if ee: mask[d_s:d_split, s_s:s_split] = 1.0
            if ei: mask[d_s:d_split, s_split:s_e] = 1.0
            if ie: mask[d_split:d_e, s_s:s_split] = 1.0
            if ii: mask[d_split:d_e, s_split:s_e] = 1.0
    return mask


class NeuronState(NamedTuple):
    V_f: jax.Array      # [B, 1, H]
    adapt: jax.Array    # [B, 1, H]
    V_res: jax.Array    # [B, 1, H]


class SynapseState(NamedTuple):
    delay_buffer: jax.Array  # [B, delay, H]  (zeros if delay==0)
    R: jax.Array             # [B, 1, H]
    u_trace: jax.Array       # [B, 1, H]


class NeuronLayer(eqx.Module):
    # Learnable parameters
    tau_exc: jax.Array          # (1, 1, H)
    tau_inh: jax.Array          # (1, 1, H)
    v_th: jax.Array             # (1, 1, H)
    reset_weight: jax.Array     # (1, 1, H)
    tau_adapt_param: jax.Array  # (1, 1, H)

    # Static config
    h_dim: int = eqx.field(static=True)
    beta_adapt: float = eqx.field(static=True)
    v_noise_std: float = eqx.field(static=True)

    def __init__(self, h_dim: int, key: jax.Array,
                 tau_adapt_init: float = 50.0,
                 beta_adapt: float = 0.05,
                 v_noise_std: float = 0.0):
        self.h_dim = h_dim
        self.beta_adapt = beta_adapt
        self.v_noise_std = v_noise_std

        self.tau_exc = jnp.full((1, 1, h_dim), 2.5)
        self.tau_inh = jnp.full((1, 1, h_dim), 1.5)
        self.v_th = jnp.ones((1, 1, h_dim))
        self.reset_weight = jnp.ones((1, 1, h_dim))
        self.tau_adapt_param = jnp.full((1, 1, h_dim), tau_adapt_init)

    def __call__(self, I: jax.Array, state: NeuronState,
                 *, key: Optional[jax.Array] = None) -> Tuple[jax.Array, jax.Array, NeuronState]:
        """
        I:     [B, T, H]
        state: NeuronState with each field [B, 1, H]
        Returns: (out_spikes [B,T,H], V_final [B,T,H], new_state)
        """
        V_f_init = state.V_f
        adapt_init = state.adapt
        V_res_init = state.V_res

        alpha_exc = jax.nn.sigmoid(self.tau_exc) * 0.99
        V_f = scan_decay(I, alpha_exc, h_init=V_f_init)

        adapt_alpha = jnp.exp(-1.0 / self.tau_adapt_param)
        adapt_trace = scan_decay(jax.nn.sigmoid(V_f - self.v_th), adapt_alpha, h_init=adapt_init)

        v_th_eff = self.v_th + adapt_trace * self.beta_adapt
        spikes_pre = spike_fn(V_f - v_th_eff)
        V_res = scan_decay(spikes_pre * self.reset_weight,
                           jax.nn.sigmoid(self.tau_inh) * 0.99, h_init=V_res_init)

        V_final = V_f - V_res - adapt_trace * self.beta_adapt
        if self.v_noise_std > 0 and key is not None:
            V_final = V_final + jax.random.normal(key, V_final.shape) * self.v_noise_std

        out_spikes = spike_fn(V_final - self.v_th)
        new_state = NeuronState(
            V_f=V_f[:, -1:, :],
            adapt=adapt_trace[:, -1:, :],
            V_res=V_res[:, -1:, :],
        )
        return out_spikes, V_final, new_state


class SynapseLayer(eqx.Module):
    # Learnable parameters
    weight: jax.Array           # [H, H]
    U_logit: jax.Array          # [1, 1, H]
    tau_rec_logit: jax.Array    # [1, 1, H]
    tau_facil_param: jax.Array  # [1, 1, H]

    # Non-trainable (excluded via make_param_filter)
    mask: jax.Array             # [H, H]

    # Static config
    delay: int = eqx.field(static=True)
    facil_amp: float = eqx.field(static=True)

    def __init__(self, config: dict, key: jax.Array):
        h = config["neuron_dim"]
        self.delay = config.get("synaptic_delay", 0)
        self.facil_amp = config.get("facil_amp", 1.5)
        self.mask = jnp.array(build_topology_mask(config))

        self.weight = jax.random.normal(key, (h, h)) * (1.0 / h) ** 0.5
        self.U_logit = jnp.full((1, 1, h), -1.5)
        self.tau_rec_logit = jnp.full((1, 1, h), 2.0)
        self.tau_facil_param = jnp.full((1, 1, h), config.get("tau_facil", 25.0))

    def __call__(self, spikes: jax.Array, state: SynapseState
                 ) -> Tuple[jax.Array, SynapseState]:
        """
        spikes: [B, T, H]
        state:  SynapseState
        Returns: (syn_current [B,T,H], new_state)
        """
        B, L, H = spikes.shape

        # Synaptic Delay
        if self.delay > 0:
            padded = jnp.concatenate([state.delay_buffer, spikes], axis=1)
            delayed = padded[:, :L, :]
            new_delay_buffer = spikes[:, -self.delay:, :]
        else:
            delayed = spikes
            new_delay_buffer = state.delay_buffer  # keep the zeros placeholder

        U_full = jax.nn.sigmoid(self.U_logit)
        tau_rec = jax.nn.softplus(self.tau_rec_logit) + 1.0
        alpha_fac = jnp.exp(-1.0 / self.tau_facil_param)

        f_gain = self.facil_amp
        log_a_f = jnp.log(jnp.clip(alpha_fac - f_gain * delayed, 1e-6, 1.0))
        S_f = jnp.cumsum(log_a_f, axis=1)
        b_f = (1.0 - alpha_fac) * U_full + f_gain * delayed
        log_acc_f = logcumsumexp(jnp.log(b_f + 1e-8) - S_f, axis=1)
        u0 = state.u_trace
        u_trace = jnp.exp(S_f) * u0 + jnp.exp(log_acc_f + S_f)
        u_trace = jnp.clip(u_trace, 0.0, 1.0)

        alpha_r = 1.0 - (1.0 / tau_rec)
        log_a_r = jnp.log(jnp.clip(alpha_r - u_trace * delayed, 1e-6, 1.0))
        S_r = jnp.cumsum(log_a_r, axis=1)
        b_r = 1.0 - alpha_r
        log_acc_r = logcumsumexp(jnp.log(b_r + 1e-8) - S_r, axis=1)
        R0 = state.R
        R = jnp.exp(S_r) * R0 + jnp.exp(log_acc_r + S_r)
        R = jnp.clip(R, 0.0, 1.0)

        stp_factor = R * u_trace
        masked_w = self.weight * self.mask
        syn_current = jnp.einsum('bth,dh->btd', delayed * stp_factor, masked_w)

        new_state = SynapseState(
            delay_buffer=new_delay_buffer,
            R=R[:, -1:, :],
            u_trace=u_trace[:, -1:, :],
        )
        return syn_current, new_state


class RMSNorm(eqx.Module):
    scale: jax.Array
    epsilon: float = eqx.field(static=True, default=1e-6)

    def __init__(self, dim: int):
        self.scale = jnp.ones(dim)

    def __call__(self, x):
        rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.epsilon)
        return x / rms * self.scale


class LinearEncoder(eqx.Module):
    linear: eqx.nn.Linear
    norm: eqx.nn.LayerNorm

    def __init__(self, in_dim: int, out_dim: int, key: jax.Array):
        self.linear = eqx.nn.Linear(in_dim, out_dim, key=key)
        self.norm = eqx.nn.LayerNorm(out_dim)

    def __call__(self, x):
        # eqx.nn.Linear broadcasts over leading dims
        x = jax.vmap(jax.vmap(self.linear))(x)  # [B, T, in] -> [B, T, out]
        x = jax.vmap(jax.vmap(self.norm))(x)
        return x


def dropout(x, rate: float, key: jax.Array):
    keep = jax.random.bernoulli(key, 1.0 - rate, x.shape)
    return jnp.where(keep, x / (1.0 - rate), 0.0)


class PHCSSM(eqx.Module):
    encoder: LinearEncoder
    shared_nl: NeuronLayer
    shared_sl: SynapseLayer
    # Read by the ssm readout branch. No reported configuration selects
    # that branch, but its scale vector is inside the reported parameter
    # counts, so it stays either way.
    norm: RMSNorm
    dec: eqx.nn.Linear

    # Non-trainable arrays (excluded via make_param_filter)
    input_mask: jax.Array       # [H]
    readout_slices: tuple       # tuple of (start, end), static

    # Static config
    h_dim: int = eqx.field(static=True)
    num_transmissions: int = eqx.field(static=True)
    convergence_loss_weight: float = eqx.field(static=True)
    drive_alpha: float = eqx.field(static=True)
    dropout_rate: float = eqx.field(static=True)
    readout_source: str = eqx.field(static=True)
    readout_strategy: str = eqx.field(static=True)
    readout_weight_type: str = eqx.field(static=True)
    readout_weight_decay: float = eqx.field(static=True)
    overlap: int = eqx.field(static=True)
    chunk_size: int = eqx.field(static=True)
    synaptic_delay: int = eqx.field(static=True)

    def __init__(self, config: dict, key: jax.Array):
        h = config["neuron_dim"]
        self.h_dim = h
        self.num_transmissions = config.get("num_transmissions", 12)
        self.convergence_loss_weight = config.get("convergence_loss_weight", 0.0)
        self.drive_alpha = config.get("drive_alpha", 0.3)
        self.dropout_rate = config.get("dropout", 0.3)
        self.readout_source = config.get("readout_source", "voltage")
        self.readout_strategy = config.get("readout_strategy", "mean")
        self.readout_weight_type = config.get("readout_weight_type", "linear")
        self.readout_weight_decay = config.get("readout_weight_decay", 0.99)
        self.overlap = config.get("overlap", 140)
        self.chunk_size = config.get("chunk_size", 1024)
        self.synaptic_delay = config.get("synaptic_delay", 0)

        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.encoder = LinearEncoder(config["input_dim"], h, key=k1)
        self.shared_nl = NeuronLayer(
            h_dim=h, key=k2,
            tau_adapt_init=config.get("tau_adapt", 50.0),
            beta_adapt=config.get("beta_adapt", 0.05),
            v_noise_std=config.get("v_noise_std", 0.0),
        )
        self.shared_sl = SynapseLayer(config=config, key=k3)
        # norm is initialized later after we know readout_dim

        input_mask_np = np.zeros(h, dtype=np.float32)
        readout_slices_list = []
        start = 0
        for i, r in enumerate(config["partition_region_ratio"]):
            end = h if i == len(config["partition_region_ratio"]) - 1 else start + int(h * r)
            mid = start + int((end - start) * config["region_ei_ratios"][i])
            in_t = config["input_target_types"][i]
            if in_t == "E":    input_mask_np[start:mid] = 1.0
            elif in_t == "I":  input_mask_np[mid:end] = 1.0
            elif in_t == "EI": input_mask_np[start:end] = 1.0
            out_t = config["readout_target_types"][i]
            if out_t in ("E", "I", "EI"):
                s2 = start if out_t != "I" else mid
                e2 = mid if out_t == "E" else end
                if e2 > s2:
                    readout_slices_list.append((s2, e2))
            start = end

        self.input_mask = jnp.array(input_mask_np)
        self.readout_slices = tuple(readout_slices_list)

        # Compute readout dimension for decoder and norm
        if readout_slices_list:
            readout_dim = sum(e - s for s, e in readout_slices_list)
        else:
            readout_dim = h
        self.norm = RMSNorm(readout_dim)
        self.dec = eqx.nn.Linear(readout_dim, config.get("output_dim", 5), key=k4)


    def _make_init_states(self, B: int):
        h = self.h_dim
        zeros_b1h = jnp.zeros((B, 1, h))
        n_state = NeuronState(V_f=zeros_b1h, adapt=zeros_b1h, V_res=zeros_b1h)
        s_state = SynapseState(
            delay_buffer=jnp.zeros((B, max(self.synaptic_delay, 1), h)),
            R=jnp.ones((B, 1, h)),
            u_trace=jnp.broadcast_to(
                jax.nn.sigmoid(self.shared_sl.U_logit), (B, 1, h)),
        )
        return n_state, s_state


    def apply_readout(self, target_memory):
        if self.readout_slices:
            ri = jnp.concatenate([target_memory[:, :, s:e]
                                  for s, e in self.readout_slices], axis=2)
        else:
            ri = target_memory

        rs = self.readout_strategy
        if rs in ("last_max", "max_last"):
            readout = (ri[:, -1, :] + ri.max(axis=1)) / 2.0
        elif rs in ("max_mean", "mean_max"):
            readout = (ri.max(axis=1) + ri.mean(axis=1)) / 2.0
        elif rs == "mean":
            readout = ri.mean(axis=1)
        elif rs == "last":
            readout = ri[:, -1, :]
        elif rs == "sum":
            readout = ri.sum(axis=1)
        elif rs == "max":
            readout = ri.max(axis=1)
        elif rs == "ssm":
            readout = jax.vmap(jax.vmap(self.norm))(ri)[:, -1, :]
        elif rs == "com":
            B, T, H = ri.shape
            t_idx = jnp.arange(T).astype(jnp.float32) / jnp.maximum(T - 1, 1)
            w = ri + 1e-8
            readout = (w * t_idx[None, :, None]).sum(axis=1) / w.sum(axis=1)
        elif rs == "weighted":
            B, L, H = ri.shape
            if self.readout_weight_type == "linear":
                weights = jnp.linspace(0.0, 1.0, L)
            else:
                decay = self.readout_weight_decay
                idx = jnp.arange(L)
                weights = jnp.power(decay, (L - 1 - idx).astype(jnp.float32))
            weights = weights[None, :, None]
            readout = (ri * weights).sum(axis=1) / (weights.sum() + 1e-8)
        else:
            readout = ri[:, -1, :]

        # Decoder expects [B, readout_H], but dec is Linear(h_dim, output_dim).
        # readout_slices may select a subset of h_dim, so we need to pad or
        # ensure dec input matches. The original uses the concatenated slices
        # directly - readout_H may differ from h_dim.
        # We handle this by making dec match readout_H at construction.
        # Actually: in original code, dec = nn.Dense(output_dim) applied to readout.
        # eqx.nn.Linear needs matching in_features. We will handle this by using
        # a flexible linear below in __call__.
        return readout


    def compute_convergence_loss(self, all_spikes):
        """
        all_spikes: [num_trans, B, T, H]
        weight[i] = (i + 1) / (num_transmissions - 1)
        Returns: scalar loss.
        """
        diffs = all_spikes[1:] - all_spikes[:-1]
        step_change = (diffs ** 2).mean(axis=(1, 2, 3))  # [num_trans-1]
        n = self.num_transmissions - 1
        weights = (jnp.arange(n) + 1).astype(jnp.float32) / n
        return (step_change * weights).sum() / weights.sum()

    def compute_effective_steps(self, all_spikes, threshold=1e-3):
        """
        Estimate at which transmission step spikes have converged.
        all_spikes: [num_trans, B, T, H]
        Returns: the effective number of transmission steps.
        """
        diffs = all_spikes[1:] - all_spikes[:-1]
        step_change = (diffs ** 2).mean(axis=(1, 2, 3))  # [num_trans-1]
        still_changing = (step_change >= threshold).astype(jnp.float32)
        idx = jnp.arange(self.num_transmissions - 1).astype(jnp.float32)
        weighted = still_changing * (idx + 1)
        last_change_idx = jnp.argmax(weighted)
        effective = jnp.where(still_changing.sum() > 0,
                              last_change_idx + 2.0,
                              1.0)
        return effective


    def __call__(self, x, *, key=None, inference=False):
        """
        x: [B, T, D_in]
        Returns: (logits [B, out_dim], all_spikes [N, B, T, H], all_voltages [N, B, T, H])
        """
        x = self.encoder(x)  # [B, T, H]

        if not inference and key is not None:
            drop_key, noise_key_base = jax.random.split(key)
            x = dropout(x, self.dropout_rate, drop_key)
        else:
            noise_key_base = None

        sensory = x * self.input_mask[None, None, :]  # [B, T, H]
        B = x.shape[0]
        init_n, init_s = self._make_init_states(B)

        # Capture sensory and model refs in closure (JAX handles this correctly)
        shared_nl = self.shared_nl
        shared_sl = self.shared_sl
        v_noise_std = self.shared_nl.v_noise_std
        drive_alpha = self.drive_alpha

        # Every transmission iteration starts from the SAME
        # initial neuron + synapse state (reset per iter). State does NOT carry
        # across iterations; only x_curr accumulates synaptic feedback from
        # the previous iter. This makes iter K-1 see zero-state dynamics like a
        # chip-deployed RSNN single pass.
        def transmission_step(x_curr, step_idx):
            nk = None
            if noise_key_base is not None and v_noise_std > 0:
                nk = jax.random.fold_in(noise_key_base, step_idx)
            spikes, voltage, _ = shared_nl(x_curr, init_n, key=nk)
            syn_out, _ = shared_sl(spikes, init_s)
            x_next = syn_out + drive_alpha * sensory
            return x_next, (spikes, voltage)

        _, (all_spikes, all_voltages) = jax.lax.scan(
            transmission_step,
            drive_alpha * sensory,
            xs=jnp.arange(self.num_transmissions),
            length=self.num_transmissions,
        )
        # all_spikes: [num_trans, B, T, H]
        # all_voltages: [num_trans, B, T, H]

        target = all_voltages[-1] if self.readout_source == "voltage" else all_spikes[-1]
        readout = self.apply_readout(target)  # [B, readout_H]
        logits = jax.vmap(self.dec)(readout)  # [B, out_dim]
        return logits, all_spikes, all_voltages


    def parallel_forward(self, x, *, key=None, inference=False):
        """
        x: [B, T, D_in]
        Returns: (logits, all_spikes, all_voltages, reconstructed, final_n_state, final_s_state)
        """
        B, L, D = x.shape
        overlap = self.overlap
        chunk_size = self.chunk_size
        stride = chunk_size - overlap

        # Pad beginning with overlap zeros
        x_padded = jnp.concatenate([jnp.zeros((B, overlap, D)), x], axis=1)
        curr_len = x_padded.shape[1]

        # Pad end to fit chunk/stride alignment
        if curr_len < chunk_size:
            x_padded = jnp.concatenate(
                [x_padded, jnp.zeros((B, chunk_size - curr_len, D))], axis=1)
        else:
            rem = (curr_len - chunk_size) % stride
            if rem > 0:
                x_padded = jnp.concatenate(
                    [x_padded, jnp.zeros((B, stride - rem, D))], axis=1)

        padded_len = x_padded.shape[1]
        num_chunks = (padded_len - chunk_size) // stride + 1

        # Sliding windows: [B, num_chunks, chunk_size, D]
        chunk_idx = (jnp.arange(num_chunks)[:, None] * stride +
                     jnp.arange(chunk_size)[None, :])
        windows = x_padded[:, chunk_idx, :]  # [B, num_chunks, chunk_size, D]
        x_parallel = windows.reshape(B * num_chunks, chunk_size, D)

        # Run forward on all chunks
        out_logits, all_spikes, all_voltages = self(
            x_parallel, key=key, inference=inference)

        # Reconstruct target from chunks
        target_chunked = (all_spikes[-1] if self.readout_source == "spike"
                          else all_voltages[-1])
        H = target_chunked.shape[-1]
        target_chunked = target_chunked.reshape(B, num_chunks, chunk_size, H)
        valid_segs = target_chunked[:, :, overlap:, :]
        reconstructed = valid_segs.reshape(B, -1, H)[:, :L, :]
        readout = self.apply_readout(reconstructed)
        logits = jax.vmap(self.dec)(readout)

        return logits, all_spikes, all_voltages, reconstructed


    def forward_stateful(self, x, prev_n_state=None, prev_s_state=None,
                         *, key=None, inference=False):
        """
        Like __call__ but accepts/returns (NeuronState, SynapseState) for TBPTT.
        x: [B, T, D_in]
        Returns: (logits, all_spikes, all_voltages, final_n_state, final_s_state)
        """
        x = self.encoder(x)

        if not inference and key is not None:
            drop_key, noise_key_base = jax.random.split(key)
            x = dropout(x, self.dropout_rate, drop_key)
        else:
            noise_key_base = None

        sensory = x * self.input_mask[None, None, :]
        B = x.shape[0]

        if prev_n_state is None or prev_s_state is None:
            init_n, init_s = self._make_init_states(B)
        else:
            init_n, init_s = prev_n_state, prev_s_state

        shared_nl = self.shared_nl
        shared_sl = self.shared_sl
        v_noise_std = self.shared_nl.v_noise_std
        drive_alpha = self.drive_alpha

        # State resets to (init_n, init_s) for every
        # transmission iteration. For TBPTT, init_n / init_s already contain
        # the previous window's iter-K-1 final state, so cross-window state
        # carry is preserved; only WITHIN a window does the K-iter loop reset.
        # final_n / final_s for next window come from the LAST iter's final
        # neuron / synapse state at t=T-1.
        def transmission_step(x_curr, step_idx):
            nk = None
            if noise_key_base is not None and v_noise_std > 0:
                nk = jax.random.fold_in(noise_key_base, step_idx)
            spikes, voltage, new_n_st = shared_nl(x_curr, init_n, key=nk)
            syn_out, new_s_st = shared_sl(spikes, init_s)
            x_next = syn_out + drive_alpha * sensory
            return x_next, (spikes, voltage, new_n_st, new_s_st)

        _, (all_spikes, all_voltages, all_n_states, all_s_states) = jax.lax.scan(
            transmission_step,
            drive_alpha * sensory,
            xs=jnp.arange(self.num_transmissions),
            length=self.num_transmissions,
        )
        # Last iter's final state for TBPTT next-window carry
        final_n = jax.tree_util.tree_map(lambda t: t[-1], all_n_states)
        final_s = jax.tree_util.tree_map(lambda t: t[-1], all_s_states)

        target = all_voltages[-1] if self.readout_source == "voltage" else all_spikes[-1]
        readout = self.apply_readout(target)
        logits = jax.vmap(self.dec)(readout)
        return logits, all_spikes, all_voltages, final_n, final_s


    def parallel_forward_stateful(self, x, prev_n_state=None, prev_s_state=None,
                                  *, key=None, inference=False):
        """
        OSP forward with state carry-over for TBPTT windows.
        Returns: (logits, all_spikes, all_voltages, reconstructed, final_n, final_s)
        """
        B, L, D = x.shape
        overlap = self.overlap
        chunk_size = self.chunk_size
        stride = chunk_size - overlap

        x_padded = jnp.concatenate([jnp.zeros((B, overlap, D)), x], axis=1)
        curr_len = x_padded.shape[1]

        if curr_len < chunk_size:
            x_padded = jnp.concatenate(
                [x_padded, jnp.zeros((B, chunk_size - curr_len, D))], axis=1)
        else:
            rem = (curr_len - chunk_size) % stride
            if rem > 0:
                x_padded = jnp.concatenate(
                    [x_padded, jnp.zeros((B, stride - rem, D))], axis=1)

        padded_len = x_padded.shape[1]
        num_chunks = (padded_len - chunk_size) // stride + 1

        chunk_idx = (jnp.arange(num_chunks)[:, None] * stride +
                     jnp.arange(chunk_size)[None, :])
        windows = x_padded[:, chunk_idx, :]
        x_parallel = windows.reshape(B * num_chunks, chunk_size, D)

        # Expand prev_states across chunks if provided
        if prev_n_state is not None and prev_s_state is not None:
            expanded_n = jax.tree_util.tree_map(
                lambda t: jnp.repeat(t, num_chunks, axis=0), prev_n_state)
            expanded_s = jax.tree_util.tree_map(
                lambda t: jnp.repeat(t, num_chunks, axis=0), prev_s_state)
        else:
            expanded_n, expanded_s = None, None

        _, all_spikes, all_voltages, flat_n, flat_s = self.forward_stateful(
            x_parallel, expanded_n, expanded_s, key=key, inference=inference)

        # Extract final states from last chunk
        final_n = jax.tree_util.tree_map(
            lambda t: t.reshape(B, num_chunks, *t.shape[1:])[:, -1], flat_n)
        final_s = jax.tree_util.tree_map(
            lambda t: t.reshape(B, num_chunks, *t.shape[1:])[:, -1], flat_s)

        target_chunked = (all_spikes[-1] if self.readout_source == "spike"
                          else all_voltages[-1])
        H = target_chunked.shape[-1]
        target_chunked = target_chunked.reshape(B, num_chunks, chunk_size, H)
        valid_segs = target_chunked[:, :, overlap:, :]
        reconstructed = valid_segs.reshape(B, -1, H)[:, :L, :]
        readout = self.apply_readout(reconstructed)
        logits = jax.vmap(self.dec)(readout)

        return logits, all_spikes, all_voltages, reconstructed, final_n, final_s


    def forward(self, x, *, key=None, inference=False, mode="nothing"):
        if mode in ("osp", "osp+tbptt"):
            return self.parallel_forward(x, key=key, inference=inference)
        else:
            logits, all_spikes, all_voltages = self(
                x, key=key, inference=inference)
            return logits, all_spikes, all_voltages, None


def apply_dale(model: PHCSSM, region_slices: list) -> PHCSSM:
    weight = model.shared_sl.weight
    for s, split, e in region_slices:
        weight = weight.at[:, s:split].set(jnp.clip(weight[:, s:split], 0.0, None))
        weight = weight.at[:, split:e].set(jnp.clip(weight[:, split:e], None, 0.0))
    new_sl = eqx.tree_at(lambda sl: sl.weight, model.shared_sl, weight)
    return eqx.tree_at(lambda m: m.shared_sl, model, new_sl)


def apply_dale_params(params, static, region_slices: list):
    """Apply Dale's law on the params partition (for use inside JIT)."""
    model = eqx.combine(params, static)
    model = apply_dale(model, region_slices)
    return eqx.partition(model, eqx.is_inexact_array)


def build_hebb_mask(config: dict, region_slices: list, topology_mask: jax.Array) -> jax.Array:
    """Precompute the combined topology * type mask for Hebbian updates."""
    hebb_mode = config.get("hebbian_mode", [1, 0, 0, 0])
    ee, ei, ie, ii = hebb_mode
    h = config["neuron_dim"]
    type_mask = np.zeros((h, h), dtype=np.float32)
    for s, split, e in region_slices:
        for s2, split2, e2 in region_slices:
            if ee: type_mask[s2:split2, s:split] = 1.0
            if ei: type_mask[split2:e2, s:split] = 1.0
            if ie: type_mask[s2:split2, split:e] = 1.0
            if ii: type_mask[split2:e2, split:e] = 1.0
    return jnp.array(type_mask) * topology_mask


@partial(jax.jit, static_argnums=(4, 5, 6, 7, 8))
def _hebbian_core(weight, all_spikes, hebb_mask, reward,
                  lr_hebb, tau_plus, tau_minus, A_plus, A_minus):
    """JIT-compiled STDP core. All computation stays on device."""
    spikes = all_spikes.mean(axis=0)
    B, T, H = spikes.shape

    alpha_plus = jnp.exp(-1.0 / tau_plus)
    alpha_minus = jnp.exp(-1.0 / tau_minus)

    def get_traces(s, alpha):
        T_idx = jnp.arange(T)[None, :, None].astype(jnp.float32)
        S_dec = T_idx * jnp.log(alpha + 1e-8)
        return jnp.exp(logcumsumexp(jnp.log(s + 1e-8) - S_dec, axis=1) + S_dec)

    trace_plus = get_traces(spikes, alpha_plus)
    trace_minus = get_traces(spikes, alpha_minus)
    zeros_b1h = jnp.zeros((B, 1, H))
    trace_pre = jnp.concatenate([zeros_b1h, trace_plus[:, :-1, :]], axis=1)
    trace_post = jnp.concatenate([zeros_b1h, trace_minus[:, :-1, :]], axis=1)

    dW_ltp = jnp.einsum('bth,bti->hi', spikes, trace_pre)
    dW_ltd = jnp.einsum('bth,bti->hi', trace_post, spikes)
    dW = (A_plus * dW_ltp - A_minus * dW_ltd) / (B * T)
    dW = jnp.where(reward is not None, dW * reward, dW)
    dW = dW * hebb_mask

    new_weight = weight + lr_hebb * dW
    # info stays on device as JAX arrays
    ltp_count = (dW > 0).sum()
    ltd_count = (dW < 0).sum()
    hebb_mag = jnp.abs(lr_hebb * dW).mean()
    return new_weight, hebb_mag, ltp_count, ltd_count


def hebbian_update(model: PHCSSM, all_spikes: jax.Array,
                   hebb_mask: jax.Array, config: dict,
                   region_slices: list, reward=None,
                   weight_grad: Optional[jax.Array] = None):
    """
    all_spikes: [num_transmissions, B, T, H] (from scan output, already stacked)
    weight_grad: optional gradient of weight from backprop (for Hebb/Grad ratio)
    Returns: (new_model, info_dict) or (model, None) if disabled.
    """
    rule = config.get("hebbian_rule", None)
    if rule is None:
        return model, None

    lr_hebb = config.get("lr_hebb", 1e-3)
    tau_plus = config.get("stdp_tau_plus", 20.0)
    tau_minus = config.get("stdp_tau_minus", 20.0)
    A_plus = config.get("stdp_A_plus", 1.0)
    A_minus = config.get("stdp_A_minus", 1.05)

    reward_val = reward if reward is not None else jnp.float32(1.0)

    new_weight, hebb_mag, ltp_count, ltd_count = _hebbian_core(
        model.shared_sl.weight, all_spikes, hebb_mask, reward_val,
        lr_hebb, tau_plus, tau_minus, A_plus, A_minus)

    new_sl = eqx.tree_at(lambda sl: sl.weight, model.shared_sl, new_weight)
    new_model = eqx.tree_at(lambda m: m.shared_sl, model, new_sl)
    new_model = apply_dale(new_model, region_slices)

    # Hebb/Grad ratio
    if weight_grad is not None:
        grad_mag = jnp.abs(weight_grad).mean() + 1e-12
        ratio = hebb_mag / grad_mag
    else:
        ratio = jnp.float32(0.0)

    info = {
        "hebb_mag": hebb_mag,       # on device
        "ltp_count": ltp_count,     # on device
        "ltd_count": ltd_count,     # on device
        "ratio": ratio,             # Hebb/Grad ratio
    }
    return new_model, info


def make_param_filter(model: PHCSSM):
    """
    Returns a filter_spec pytree: True for trainable arrays, False for
    non-trainable ones (mask, input_mask).
    """
    filter_spec = jax.tree_util.tree_map(eqx.is_inexact_array, model)
    # Mark non-trainable
    filter_spec = eqx.tree_at(
        lambda m: m.shared_sl.mask, filter_spec, False)
    filter_spec = eqx.tree_at(
        lambda m: m.input_mask, filter_spec, False)
    return filter_spec


def create_phcssm(config: dict, key: jax.Array) -> PHCSSM:
    """Create PHCSSM with default config values filled in."""
    defaults = {
        "num_regions": 3,
        "partition_region_ratio": [0.3, 0.4, 0.3],
        "region_ei_ratios": [0.8, 0.8, 0.8],
        "hierarchy_matrix": [
            [[1, 1, 1, 1], [1, 0, 1, 0], [1, 0, 1, 0]],
            [[1, 0, 1, 0], [1, 1, 1, 1], [1, 0, 1, 0]],
            [[1, 0, 1, 0], [1, 0, 1, 0], [1, 1, 1, 1]],
        ],
        "input_target_types": ["EI", "EI", "EI"],
        "readout_target_types": ["EI", "EI", "EI"],
        "drive_alpha": 0.3,
        "num_transmissions": 12,
    }
    full_config = {**defaults, **config}
    return PHCSSM(full_config, key)


if __name__ == "__main__":
    print(f"PHCSSM {__version__} self-test")
    print("=" * 60)

    config = {
        "input_dim": 3, "output_dim": 4, "neuron_dim": 64,
        "num_regions": 2, "partition_region_ratio": [0.5, 0.5],
        "region_ei_ratios": [0.8, 0.8],
        "hierarchy_matrix": [
            [(1, 1, 1, 1), (0, 0, 0, 0)],
            [(1, 1, 0, 0), (1, 1, 1, 1)],
        ],
        "input_target_types": ["E", 0],
        "readout_target_types": [0, "E"],
        "num_transmissions": 4,
    }
    key = jax.random.PRNGKey(0)
    model = create_phcssm(config, key)
    x = jnp.ones((2, 100, 3))
    logits, spikes, voltages = model(x, inference=True)
    print(f"Output shape: {logits.shape}")
    print(f"Spikes shape: {spikes.shape}")
    print(f"Voltages shape: {voltages.shape}")

    # Test parameter filter
    fs = make_param_filter(model)
    params, static = eqx.partition(model, fs)
    print(f"Partition OK")

    # Test Dale's law
    region_slices = calculate_region_slices(config)
    model2 = apply_dale(model, region_slices)
    print(f"Dale's law OK")

    print("All tests passed!")

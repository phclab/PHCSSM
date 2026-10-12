// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Copyright National Yang Ming Chiao Tung University, Taiwan.
// Creator Po-Han Chiang. Patent pending.
// Noncommercial use only. For commercial licensing see LICENSE.
// rsnn_forward.cpp
// ================
// PHCSSM 1.0.0 RSNN sequential-deployment forward pass, FP32.
// See header for API. Source-of-truth Python implementation:
//   rsnn/rsnn_jax_cpu.py (functions _encode,
//   _make_scan_body, _readout).
//
// Design constraints:
//   * Streaming readout never store the full [T, H] activity tensor
//     (Worms T=17984, H=64 would need 4.6 MB which exceeds micro:bit RAM).
//   * Static buffers no malloc.
//   * Cross-platform compiles cleanly with g++ (PC test) and Arduino IDE
//     (micro:bit). No platform-specific includes here.

#include "rsnn_forward.h"
#include <math.h>
#include <string.h>


// ===================================================================
// State buffers (all sized by compile-time macros from rsnn_weights_*.h)
// ===================================================================

// Step-state vectors (per-neuron, length H)
static float V_f[H];
static float adapt[H];
static float V_res[H];
static float u_trace[H];
static float R_state[H];
static float prev_sp[H];

// Delay buffer (length max(DELAY,1) of H-vectors)
#if DELAY > 0
static float delay_buf[DELAY][H];
#else
static float delay_buf[1][H];   // size-1 placeholder (unused)
#endif

// Per-step scratch (avoid stack churn on Cortex-M)
static float sensory_enc[H];       // post-encoder, post-input_mask
static float delayed[H];
static float syn[H];

// Readout accumulator: holds the running aggregate over t = 0..T-1 of
// the (sliced) activity vector. Size HPRIME.
static float readout_acc[HPRIME];
// For "max" strategy, holds the running max; also reused for "last" / "ssm".
static float readout_last[HPRIME];

static int   readout_t_count;      // number of timesteps accumulated so far


// ===================================================================
// Numerical helpers
// ===================================================================

static inline float clipf(float x, float lo, float hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static inline float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Numerically stable softplus(x) = log(1 + exp(x))
//   = max(x, 0) + log(1 + exp(-|x|))
static inline float softplusf(float x) {
    float ax = x < 0 ? -x : x;
    float base = x > 0 ? x : 0.0f;
    return base + log1pf(expf(-ax));
}


// ===================================================================
// Encoder: x[D_IN] -> sensory_enc[H]
// ===================================================================
extern "C"
void rsnn_apply_encoder(const float* x_Din, float* sensory_out_H)
{
    // h = x @ enc_w.T + enc_b
    float h[H];
    for (int i = 0; i < H; i++) {
        float acc = ENC_B[i];
        // ENC_W is [H, D_IN] row-major; sum over D_IN
        for (int j = 0; j < D_IN; j++) {
            acc += ENC_W[i][j] * x_Din[j];
        }
        h[i] = acc;
    }
    // LayerNorm over H
    float mean = 0.0f;
    for (int i = 0; i < H; i++) mean += h[i];
    mean /= (float)H;
    float var = 0.0f;
    for (int i = 0; i < H; i++) {
        float d = h[i] - mean;
        var += d * d;
    }
    var /= (float)H;
    float inv_std = 1.0f / sqrtf(var + 1e-5f);
    // (h - mean) * inv_std * ln_w + ln_b, then * input_mask
    for (int i = 0; i < H; i++) {
        float v = (h[i] - mean) * inv_std * LN_W[i] + LN_B[i];
        sensory_out_H[i] = v * INPUT_MASK[i];
    }
}


// ===================================================================
// Streaming-readout accumulator
// ===================================================================

static void readout_init(void)
{
    readout_t_count = 0;
    switch (READOUT_STRATEGY) {
        case 0:  // mean: zero accumulator
        case 2:  // sum: zero accumulator
            for (int i = 0; i < HPRIME; i++) readout_acc[i] = 0.0f;
            break;
        case 3:  // max: start very negative
            for (int i = 0; i < HPRIME; i++) readout_acc[i] = -INFINITY;
            break;
        case 1:  // last: don't care, will be overwritten
        case 4:  // ssm: only needs last, will be overwritten
            for (int i = 0; i < HPRIME; i++) readout_acc[i] = 0.0f;
            break;
    }
    for (int i = 0; i < HPRIME; i++) readout_last[i] = 0.0f;
}

// Extract the slice channels from a full H-vector into a HPRIME-vector.
static void slice_into(const float* full_H, float* sliced_Hprime)
{
#if N_SLICES > 0
    int offset = 0;
    for (int s = 0; s < N_SLICES; s++) {
        int start = SLICE_START[s];
        int end   = SLICE_END[s];
        for (int i = start; i < end; i++) {
            sliced_Hprime[offset++] = full_H[i];
        }
    }
    // offset should now equal HPRIME
#else
    // No slices: identity copy (HPRIME == H in this branch)
    for (int i = 0; i < HPRIME; i++) sliced_Hprime[i] = full_H[i];
#endif
}

// Accumulate one timestep of activity into the readout buffer.
static void readout_accumulate(const float* activity_H)
{
    float sliced[HPRIME];
    slice_into(activity_H, sliced);
    switch (READOUT_STRATEGY) {
        case 0:  // mean: sum + divide at the end
        case 2:  // sum: just sum
            for (int i = 0; i < HPRIME; i++) readout_acc[i] += sliced[i];
            break;
        case 1:  // last: overwrite every step (last write wins)
            for (int i = 0; i < HPRIME; i++) readout_last[i] = sliced[i];
            break;
        case 3:  // max: running max
            for (int i = 0; i < HPRIME; i++) {
                if (sliced[i] > readout_acc[i]) readout_acc[i] = sliced[i];
            }
            break;
        case 4:  // ssm: keep the latest activity slice; compute RMS at finalize
            for (int i = 0; i < HPRIME; i++) readout_last[i] = sliced[i];
            break;
    }
    readout_t_count++;
}

// Compute final readout vector from accumulator + readout_last, place in r.
static void readout_finalize_to(float* r_Hprime)
{
    switch (READOUT_STRATEGY) {
        case 0:  // mean
            if (readout_t_count > 0) {
                float inv = 1.0f / (float)readout_t_count;
                for (int i = 0; i < HPRIME; i++) r_Hprime[i] = readout_acc[i] * inv;
            } else {
                for (int i = 0; i < HPRIME; i++) r_Hprime[i] = 0.0f;
            }
            break;
        case 1:  // last
            for (int i = 0; i < HPRIME; i++) r_Hprime[i] = readout_last[i];
            break;
        case 2:  // sum
            for (int i = 0; i < HPRIME; i++) r_Hprime[i] = readout_acc[i];
            break;
        case 3:  // max
            for (int i = 0; i < HPRIME; i++) r_Hprime[i] = readout_acc[i];
            break;
        case 4: { // ssm: (activity_last / rms_last * norm_w)
#ifdef NORM_W
            float ms = 0.0f;
            for (int i = 0; i < HPRIME; i++) ms += readout_last[i] * readout_last[i];
            ms /= (float)HPRIME;
            float inv_rms = 1.0f / sqrtf(ms + 1e-6f);
            for (int i = 0; i < HPRIME; i++) {
                r_Hprime[i] = readout_last[i] * inv_rms * NORM_W[i];
            }
#else
            // NORM_W not present in weight header for non-ssm checkpoints
            for (int i = 0; i < HPRIME; i++) r_Hprime[i] = readout_last[i];
#endif
            break;
        }
    }
}


// ===================================================================
// Decoder: r[HPRIME] -> argmax over N_CLASS logits
// ===================================================================
static int decoder_argmax(const float* r_Hprime)
{
    float logits[N_CLASS];
    for (int c = 0; c < N_CLASS; c++) {
        float acc = DEC_B[c];
        // DEC_W is [N_CLASS, HPRIME] row-major
        for (int j = 0; j < HPRIME; j++) acc += DEC_W[c][j] * r_Hprime[j];
        logits[c] = acc;
    }
    int best = 0;
    float best_v = logits[0];
    for (int c = 1; c < N_CLASS; c++) {
        if (logits[c] > best_v) { best_v = logits[c]; best = c; }
    }
    return best;
}


// ===================================================================
// State reset
// ===================================================================
extern "C"
void rsnn_reset_state(void)
{
    for (int i = 0; i < H; i++) V_f[i]     = 0.0f;
    for (int i = 0; i < H; i++) adapt[i]   = 0.0f;
    for (int i = 0; i < H; i++) V_res[i]   = 0.0f;
    for (int i = 0; i < H; i++) u_trace[i] = U_FULL[i];
    for (int i = 0; i < H; i++) R_state[i] = 1.0f;
    for (int i = 0; i < H; i++) prev_sp[i] = 0.0f;
#if DELAY > 0
    for (int d = 0; d < DELAY; d++)
        for (int i = 0; i < H; i++)
            delay_buf[d][i] = 0.0f;
#endif
    readout_init();
}


// ===================================================================
// Streaming step: ingest one timestep of sensory and update state
// ===================================================================
extern "C"
void rsnn_streaming_step(const float* sensory_t_Din)
{
    // 1. Encode sensory_t_Din [D_IN] -> sensory_enc [H]
    rsnn_apply_encoder(sensory_t_Din, sensory_enc);

    // 2. Determine 'delayed' (the spike vector this step sees as recurrent input)
#if DELAY > 0
    for (int i = 0; i < H; i++) delayed[i] = delay_buf[0][i];
    // roll(delay_buf, -1, axis=0): shift up by 1, last slot to be filled at end
    for (int d = 0; d < DELAY - 1; d++)
        for (int i = 0; i < H; i++) delay_buf[d][i] = delay_buf[d + 1][i];
    // delay_buf[DELAY - 1] will be set at end of step from spike
#else
    for (int i = 0; i < H; i++) delayed[i] = prev_sp[i];
#endif

    // 3. STP facilitation: u_trace update
    for (int i = 0; i < H; i++) {
        float u_decay = clipf(ALPHA_F[i] - FACIL_AMP * delayed[i], 1e-6f, 1.0f);
        u_trace[i] = u_decay * u_trace[i]
                     + (1.0f - ALPHA_F[i]) * U_FULL[i]
                     + FACIL_AMP * delayed[i];
        u_trace[i] = clipf(u_trace[i], 0.0f, 1.0f);
    }

    // 4. STP recovery: R update
    for (int i = 0; i < H; i++) {
        float r_decay = clipf(ALPHA_R[i] - u_trace[i] * delayed[i], 1e-6f, 1.0f);
        R_state[i] = r_decay * R_state[i] + (1.0f - ALPHA_R[i]);
        R_state[i] = clipf(R_state[i], 0.0f, 1.0f);
    }

    // 5. Synaptic input syn = (delayed * R * u_trace) @ W_eff.T
    //    Here W_eff has shape [H, H] (row index = postsynaptic, col index = presynaptic)
    //    so syn[i] = sum_j W_eff[i][j] * (delayed[j] * R[j] * u_trace[j])
    {
        float pre[H];
        for (int j = 0; j < H; j++) pre[j] = delayed[j] * R_state[j] * u_trace[j];
        for (int i = 0; i < H; i++) {
            float acc = 0.0f;
            for (int j = 0; j < H; j++) acc += W_EFF[i][j] * pre[j];
            syn[i] = acc;
        }
    }

    // 6. Drive + recurrent input, voltage updates
    for (int i = 0; i < H; i++) {
        float I_t = DRIVE_ALPHA * sensory_enc[i] + syn[i];
        V_f[i]    = ALPHA_EXC[i] * V_f[i] + softplusf(I_t);
        adapt[i]  = ADAPT_ALPHA[i] * adapt[i]
                    + softplusf(sigmoidf(V_f[i] - V_TH[i]));
    }

    // 7. Pre-reset spike + V_res update + final voltage + spike
    float V_final_buf[H];
    float spike_buf[H];
    for (int i = 0; i < H; i++) {
        float v_eff  = V_TH[i] + adapt[i] * BETA_ADAPT;
        float sp_pre = (V_f[i] - v_eff > 0.0f) ? 1.0f : 0.0f;
        V_res[i] = ALPHA_INH[i] * V_res[i]
                   + softplusf(sp_pre * RESET_WEIGHT[i]);
        float V_final = V_f[i] - V_res[i] - adapt[i] * BETA_ADAPT;
        V_final_buf[i] = V_final;
        spike_buf[i]   = (V_final - V_TH[i] > 0.0f) ? 1.0f : 0.0f;
    }

    // 8. Update delay buffer + prev_sp
#if DELAY > 0
    for (int i = 0; i < H; i++) delay_buf[DELAY - 1][i] = spike_buf[i];
#endif
    for (int i = 0; i < H; i++) prev_sp[i] = spike_buf[i];

    // 9. Streaming readout: accumulate activity (voltage or spike)
    const float* activity = (READOUT_SOURCE == 0) ? V_final_buf : spike_buf;
    readout_accumulate(activity);
}


// ===================================================================
// Streaming finalize: compute readout aggregate and argmax
// ===================================================================
extern "C"
int rsnn_streaming_finalize(void)
{
    float r[HPRIME];
    readout_finalize_to(r);
    return decoder_argmax(r);
}


// ===================================================================
// Whole-sequence inference (convenience wrapper)
// ===================================================================
extern "C"
int rsnn_predict(const float* sensory_TxDin, int T)
{
    rsnn_reset_state();
    for (int t = 0; t < T; t++) {
        rsnn_streaming_step(sensory_TxDin + t * D_IN);
    }
    return rsnn_streaming_finalize();
}


extern "C"
int rsnn_apply_readout_and_argmax(void)
{
    return rsnn_streaming_finalize();
}

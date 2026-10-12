// rsnn_forward.h
// ==============
// PHCSSM 1.0.0 RSNN sequential-deployment forward pass, FP32.
// Cross-platform: PC test harness AND micro:bit Arduino sketch share this.
//
// Hidden dimension H, input dim D_IN, output classes N_CLASS, sliced readout
// width HPRIME, delay buffer length DELAY are taken from the included
// weights header (rsnn_weights_<variant>.h via rsnn_config.h).
//
// Public API:
//   void rsnn_reset_state(void);
//   int  rsnn_predict(const float* sensory_TxDin, int T);
//   void rsnn_streaming_step(const float* sensory_t_Din);
//   int  rsnn_streaming_finalize(void);
//
// Numerical contract: produces argmax predictions matching the Python
// reference rsnn_jax_cpu.make_rsnn_forward_cpu(P) within FP32 round-off
// across UEA-MTSCA test samples. See PC test harness for cross-check.

#ifndef RSNN_FORWARD_H
#define RSNN_FORWARD_H

#include "rsnn_config.h"   // selects which rsnn_weights_<variant>.h to include
                            // and exposes the per-variant macros / arrays.

#ifdef __cplusplus
extern "C" {
#endif

// Reset all state buffers to t=0 initial conditions.
void rsnn_reset_state(void);

// Whole-sequence forward: returns argmax class index.
// sensory_TxDin layout: row-major [T][D_IN], FP32, length T*D_IN.
int rsnn_predict(const float* sensory_TxDin, int T);

// Streaming API: feed one timestep at a time. Caller must invoke
// rsnn_reset_state() before the first call of a new sample.
void rsnn_streaming_step(const float* sensory_t_Din);

// After streaming all T timesteps, call finalize to obtain the argmax.
int rsnn_streaming_finalize(void);

// Internal helpers exposed for unit tests:
void rsnn_apply_encoder(const float* sensory_t_Din, float* sensory_out_H);
int  rsnn_apply_readout_and_argmax(void);

#ifdef __cplusplus
}
#endif

#endif  // RSNN_FORWARD_H

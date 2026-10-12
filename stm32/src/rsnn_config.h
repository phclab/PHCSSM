// rsnn_config.h
// =============
// Build-time selector for which (dataset, seed) weight blob to embed.
// Edit RSNN_VARIANT_INC before each Arduino IDE compile.
//
// For PC test harness, pass -DRSNN_VARIANT_INC=\"weights/rsnn_weights_heart_seed2345.h\"
// on the g++ command line to override.

#ifndef RSNN_CONFIG_H
#define RSNN_CONFIG_H

// Default build variant (edit per reflash):
#ifndef RSNN_VARIANT_INC
#define RSNN_VARIANT_INC "weights/rsnn_weights_heart_seed2345.h"
#endif

#include RSNN_VARIANT_INC

#endif  // RSNN_CONFIG_H

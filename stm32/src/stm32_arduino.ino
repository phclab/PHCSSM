// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Copyright National Yang Ming Chiao Tung University, Taiwan.
// Creator Po-Han Chiang. Patent pending.
// Noncommercial use only. For commercial licensing see LICENSE.
// microbit_arduino.ino
// =====================
// PHCSSM RSNN sequential-deployment firmware for Microbit v2.
// USB-serial slave: receives test samples from a host PC, runs RSNN
// inference at FP32, returns argmax prediction and per-sample latency.
//
// Build (Arduino IDE 2.x):
//   Board:  Tools > Board > Nordic Semiconductor nRF5 Boards > BBC micro:bit V2
//   Add to the sketch folder:
//     - microbit_arduino.ino (this file)
//     - rsnn_forward.h
//     - rsnn_forward.cpp
//     - rsnn_config.h
//     - weights/rsnn_weights_<variant>.h (only the one named by rsnn_config.h)
//   Sketch > Verify, then Sketch > Upload to micro:bit.
//
// Protocol (binary, little-endian; 115200 baud USB CDC ACM):
//
//   Host -> micro:bit:
//     'I' <T:u16> <D:u16> <T*D*4 bytes FP32>      -- whole-sequence inference
//     'S' <T:u16> <D:u16>                          -- streaming start
//     <D*4 bytes FP32>                             -- one timestep (S only)
//     'P' 0x00 0x00 0x00                           -- ping (variant identifier)
//
//   micro:bit -> host:
//     'i' <pred:u16> <latency_us:u32>              -- after 'I' completion
//     's'                                          -- ack on 'S' start
//     <per_step_us:u32>                            -- after each S timestep
//     'd' <pred:u16> <total_us:u32>                -- after T S-timesteps
//     'p' <variant_name_string\n>                  -- after 'P'
//
// Notes:
//   - 'I' is implemented as INTERNAL streaming: micro:bit reads D*4 bytes,
//     runs one rsnn_streaming_step, then reads the next; never keeps a full
//     [T,D] tensor in RAM. This is required for Worms (T=17984, D=6 -> 432 KB).
//   - Latency window covers ONLY the rsnn forward; serial read time is
//     excluded (start timer AFTER serial bytes are received).
//   - First inference per power-cycle is slow due to first-touch effects;
//     host script should discard the timing of sample 0.

#include "rsnn_forward.h"
#include <Arduino.h>

// =====================================================================
// OPTIONAL: force 80 MHz HSI + PLL (overrides STM32duino default).
// Activate via build_flag -DFORCE_80MHZ_HSI_PLL.
// Independent of USB / external HSE works when board is powered only
// through the +3V3 pin from an external source (e.g. AD3 V+).
// =====================================================================
#ifdef FORCE_4MHZ_MSI
// Drop SYSCLK to 4 MHz MSI (low-power mode). USB clock from HSI48 so CDC survives.
static void pm_drop_to_4MHz(void) {
    RCC_OscInitTypeDef RCC_OscInitStruct = {0};
    RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
    RCC_PeriphCLKInitTypeDef PeriphClkInit = {0};

    __HAL_RCC_PWR_CLK_ENABLE();

    RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_MSI
                                     | RCC_OSCILLATORTYPE_HSI48;
    RCC_OscInitStruct.MSIState              = RCC_MSI_ON;
    RCC_OscInitStruct.MSICalibrationValue   = RCC_MSICALIBRATION_DEFAULT;
    RCC_OscInitStruct.MSIClockRange         = RCC_MSIRANGE_6;  // 4 MHz
    RCC_OscInitStruct.HSI48State            = RCC_HSI48_ON;
    RCC_OscInitStruct.PLL.PLLState          = RCC_PLL_OFF;
    if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK) { while (1); }

    RCC_ClkInitStruct.ClockType    = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK
                                   | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
    RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_MSI;
    RCC_ClkInitStruct.AHBCLKDivider  = RCC_SYSCLK_DIV1;
    RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
    RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;
    if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_0) != HAL_OK) {
        while (1);
    }

    PeriphClkInit.PeriphClockSelection = RCC_PERIPHCLK_USB;
    PeriphClkInit.UsbClockSelection    = RCC_USBCLKSOURCE_HSI48;
    if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInit) != HAL_OK) { while (1); }

    // Low-power voltage scaling (safe for SYSCLK <= 26 MHz)
    HAL_PWREx_ControlVoltageScaling(PWR_REGULATOR_VOLTAGE_SCALE2);

    SystemCoreClockUpdate();
}
#endif

#ifdef FORCE_80MHZ_HSI_PLL
// Bump SYSCLK to 80 MHz using HSI 16 MHz x PLL.
// Critical: USB FS needs 48 MHz clock. We can't get clean 48 MHz from
// PLL_Q at VCO=160 MHz (160/48 not integer), so we route USB clock to
// HSI48 (dedicated always-48-MHz oscillator) to keep USB CDC alive.
static void pm_bump_to_80MHz(void) {
    RCC_OscInitTypeDef RCC_OscInitStruct = {0};
    RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
    RCC_PeriphCLKInitTypeDef PeriphClkInit = {0};

    __HAL_RCC_PWR_CLK_ENABLE();
    if (HAL_PWREx_ControlVoltageScaling(PWR_REGULATOR_VOLTAGE_SCALE1)
        != HAL_OK) { while (1); }

    // Enable HSI48 AND HSI (for PLL source). 80 MHz via HSI/1*10/PLLR=2.
    RCC_OscInitStruct.OscillatorType        = RCC_OSCILLATORTYPE_HSI
                                            | RCC_OSCILLATORTYPE_HSI48;
    RCC_OscInitStruct.HSIState              = RCC_HSI_ON;
    RCC_OscInitStruct.HSICalibrationValue   = RCC_HSICALIBRATION_DEFAULT;
    RCC_OscInitStruct.HSI48State            = RCC_HSI48_ON;
    RCC_OscInitStruct.PLL.PLLState          = RCC_PLL_ON;
    RCC_OscInitStruct.PLL.PLLSource         = RCC_PLLSOURCE_HSI;
    RCC_OscInitStruct.PLL.PLLM              = 1;
    RCC_OscInitStruct.PLL.PLLN              = 10;
    RCC_OscInitStruct.PLL.PLLQ              = RCC_PLLQ_DIV2;
    RCC_OscInitStruct.PLL.PLLR              = RCC_PLLR_DIV2;
    if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK) { while (1); }

    RCC_ClkInitStruct.ClockType   = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK
                                  | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
    RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
    RCC_ClkInitStruct.AHBCLKDivider  = RCC_SYSCLK_DIV1;
    RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
    RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;
    if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_4) != HAL_OK) {
        while (1);
    }

    // Route USB FS clock to HSI48 so CDC stays alive
    PeriphClkInit.PeriphClockSelection = RCC_PERIPHCLK_USB;
    PeriphClkInit.UsbClockSelection    = RCC_USBCLKSOURCE_HSI48;
    if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInit) != HAL_OK) { while (1); }

    SystemCoreClockUpdate();
}
#endif



// ===================================================================
// USB serial helpers
// ===================================================================

static void wait_for(int n_bytes) {
    while (Serial.available() < n_bytes) { /* spin */ }
}

static uint16_t read_u16_le() {
    wait_for(2);
    uint8_t b0 = (uint8_t)Serial.read();
    uint8_t b1 = (uint8_t)Serial.read();
    return (uint16_t)b0 | ((uint16_t)b1 << 8);
}

static void write_u16_le(uint16_t v) {
    Serial.write((uint8_t)(v & 0xff));
    Serial.write((uint8_t)((v >> 8) & 0xff));
}

static void write_u32_le(uint32_t v) {
    Serial.write((uint8_t)(v & 0xff));
    Serial.write((uint8_t)((v >> 8) & 0xff));
    Serial.write((uint8_t)((v >> 16) & 0xff));
    Serial.write((uint8_t)((v >> 24) & 0xff));
}

// Read D_IN consecutive FP32 little-endian bytes into a float[D_IN] buffer.
static void read_sample_row(float* row_Din) {
    int total = D_IN * 4;
    int got = 0;
    union { uint8_t b[4]; float f; } u;
    while (got < total) {
        while (Serial.available() < 1) { /* spin */ }
        u.b[got & 3] = (uint8_t)Serial.read();
        got++;
        if ((got & 3) == 0) {
            row_Din[(got >> 2) - 1] = u.f;
        }
    }
}


// ===================================================================
// Command handlers
// ===================================================================

// 'I': whole-sequence inference. Internally streaming.
static void handle_inference() {
    uint16_t T = read_u16_le();
    uint16_t D = read_u16_le();
    // Sanity: D should match compiled-in D_IN; if not, drain and error.
    if ((int)D != D_IN) {
        // Drain T*D*4 bytes and reply with error indicator.
        long to_drain = (long)T * (long)D * 4;
        while (to_drain > 0) {
            if (Serial.available() > 0) { Serial.read(); to_drain--; }
        }
        Serial.write('e');           // error marker
        write_u16_le(0xFFFF);
        write_u32_le(0);
        return;
    }

    rsnn_reset_state();

    float row[D_IN];
    uint32_t t_start = micros();
    for (int t = 0; t < T; t++) {
        read_sample_row(row);        // out of timing window? See note below.
        rsnn_streaming_step(row);    // ~50 us at H=16, ~150 us at H=64
    }
    int pred = rsnn_streaming_finalize();
    uint32_t t_end = micros();

    Serial.write('i');
    write_u16_le((uint16_t)pred);
    write_u32_le(t_end - t_start);
}
// Note: the timing window above INCLUDES serial read time. To measure
// pure compute, host should compute (t_end - t_start) - (T * estimated_serial_time).
// Alternatively, host can time on its OWN side from "after sending last byte" to
// "after receiving 'i' response" and subtract one-way serial transit.
// The 'S' command below isolates compute by timing only the rsnn_streaming_step.


// 'S': per-timestep streaming with isolated per-step latency.
static void handle_streaming() {
    uint16_t T = read_u16_le();
    uint16_t D = read_u16_le();
    if ((int)D != D_IN) {
        Serial.write('e');
        return;
    }
    Serial.write('s');       // ready ack
    Serial.flush();

    rsnn_reset_state();
    float row[D_IN];
    uint32_t total_us = 0;

    for (int t = 0; t < T; t++) {
        read_sample_row(row);       // wait for host to send one timestep
        uint32_t t0 = micros();
        rsnn_streaming_step(row);
        uint32_t t1 = micros();
        uint32_t dt = t1 - t0;
        total_us += dt;
        write_u32_le(dt);
        Serial.flush();
    }
    int pred = rsnn_streaming_finalize();
    Serial.write('d');
    write_u16_le((uint16_t)pred);
    write_u32_le(total_us);
    Serial.flush();
}


// 'P': ping - return variant identifier so host can verify.
static void handle_ping() {
    // Drain 3 padding bytes from the host
    wait_for(3);
    Serial.read(); Serial.read(); Serial.read();
    Serial.write('p');
    Serial.print(RSNN_VARIANT_NAME);
    Serial.print('\n');
    Serial.flush();
}


// ===================================================================
// Arduino entry points
// ===================================================================

// =====================================================================
// POWER MEASUREMENT MODE
// =====================================================================
// Build env that defines POWER_MEASURE_LOOP or POWER_MEASURE_IDLE skips
// the serial protocol entirely. Used to measure active / idle current
// with Analog Discovery 3 (or any source-meter).
//
// POWER_MEASURE_LOOP: continuously run rsnn inference with synthetic
//   input, toggle PA12 (CN3 pin 5 / Arduino D2) at inference boundary
//   so scope can sync.
// POWER_MEASURE_IDLE: setup() inits everything then loop() __WFI()s
//   forever -> measures MCU idle (clocks still on, no inference).

#if defined(POWER_MEASURE_LOOP) || defined(POWER_MEASURE_IDLE)
#define POWER_MEASURE_TRIG_PIN PA12   // CN3 pin 5 / Arduino D2 on NUCLEO-L412KB
#endif

#ifdef POWER_MEASURE_LOOP
// Synthetic input drives the inference: deterministic, non-trivial so
// neurons actually spike (zeros would still work for compute timing,
// but spike trace counts would be degenerate).
static float pm_synth_input[D_IN];

#ifndef POWER_MEASURE_T
#define POWER_MEASURE_T 1000  // timesteps per inference call
#endif

static void pm_make_input(int t) {
    for (int d = 0; d < D_IN; d++) {
        // small-amplitude sinusoid; varies across t and d
        float x = 0.5f * sinf(0.0173f * (float)(t * 7 + d * 13));
        pm_synth_input[d] = x;
    }
}

static void pm_one_inference(void) {
    rsnn_reset_state();
    for (int t = 0; t < POWER_MEASURE_T; t++) {
        pm_make_input(t);
        rsnn_streaming_step(pm_synth_input);
    }
    (void)rsnn_streaming_finalize();
}
#endif


void setup() {
#ifdef FORCE_4MHZ_MSI
    pm_drop_to_4MHz();    // do BEFORE Serial
#endif
#ifdef FORCE_80MHZ_HSI_PLL
    pm_bump_to_80MHz();   // do BEFORE Serial so UART baud divisors compute on new clock
#endif

#if !defined(POWER_MEASURE_LOOP) && !defined(POWER_MEASURE_IDLE)
    Serial.begin(115200);
    while (!Serial && millis() < 5000) { /* wait up to 5s for USB */ }
#elif defined(POWER_MEASURE_LOOP)
    // Init Serial non-blocking so T_inf can be printed when USB is later plugged.
    // If USB never plugged in: print() goes to a small buffer that wraps; no hang.
    Serial.begin(115200);
    delay(500);
    Serial.print("BOOT_SYSCLK_Hz=");
    Serial.println(SystemCoreClock);
#endif
    rsnn_reset_state();
#if !defined(POWER_MEASURE_LOOP) && !defined(POWER_MEASURE_IDLE)
    Serial.println("READY");
    Serial.print("variant=");
    Serial.println(RSNN_VARIANT_NAME);
    Serial.print("H=");      Serial.print(H);
    Serial.print(" D_IN=");  Serial.print(D_IN);
    Serial.print(" N_CLASS="); Serial.print(N_CLASS);
    Serial.print(" HPRIME="); Serial.print(HPRIME);
    Serial.print(" DELAY=");  Serial.println(DELAY);
#endif

#if defined(POWER_MEASURE_LOOP) || defined(POWER_MEASURE_IDLE)
    pinMode(POWER_MEASURE_TRIG_PIN, OUTPUT);
    digitalWrite(POWER_MEASURE_TRIG_PIN, LOW);
    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, HIGH);  // visible "I'm alive" indicator
#endif
}

void loop() {
#if defined(POWER_MEASURE_LOOP)
    digitalWrite(POWER_MEASURE_TRIG_PIN, HIGH);
    uint32_t t0 = micros();
    pm_one_inference();
    uint32_t t1 = micros();
    digitalWrite(POWER_MEASURE_TRIG_PIN, LOW);
    Serial.print("T_inf_us=");
    Serial.println(t1 - t0);
#elif defined(POWER_MEASURE_IDLE)
    __WFI();   // sleep until next interrupt (SysTick keeps clock alive)
#else
    if (Serial.available() < 1) return;
    char cmd = (char)Serial.read();
    switch (cmd) {
        case 'I': handle_inference(); break;
        case 'S': handle_streaming(); break;
        case 'P': handle_ping();      break;
        default:
            // Unknown command; reply with '?' and drain any padding bytes
            Serial.write('?');
            break;
    }
#endif
}

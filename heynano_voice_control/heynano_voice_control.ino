/* heynano_voice_control.ino
 *
 * Arduino Nano 33 BLE Sense — Two-stage keyword spotting
 *
 * Stage 1 (IDLE):      Listen continuously for "heynano"
 * Stage 2 (ACTIVATED): Within 3 s, detect "on"  → LED_BUILTIN ON
 *                                        "off" → LED_BUILTIN OFF
 *                       Times out silently and returns to IDLE
 *
 * Pipeline (mirrors TensorFlow Lite Micro micro_speech example):
 *   PDM capture  →  ring-buffer  →  microfrontend (25 ms / 20 ms stride, 32 ch)
 *   →  49 × 32 int8 spectrogram  →  TFLite Micro inference
 *   →  rolling-window score averaging  →  state machine  →  LEDs / Serial
 *
 * Model:
 *   Input  : int8[1568]  (49 slices × 32 filterbank channels)
 *   Output : int8[5]     softmax scores for
 *              0 noise | 1 unknown | 2 heynano | 3 on | 4 off
 *   Quant  : full int8 (symmetric or asymmetric — zero-point handled via
 *              output->params.zero_point during dequant logging)
 *
 * Required libraries:
 *   Arduino_TensorFlowLite  (installs TFLite Micro + microfrontend)
 *   model_data.h / model_data.cc  (your compiled flatbuffer)
 *
 * Target: Arduino Nano 33 BLE Sense (ARDUINO_ARDUINO_NANO33BLE)
 * Baud  : 115200
 */

// ─────────────────────────────────────────────────────────────────────────────
//  Guard: Nano 33 BLE Sense only (PDM + built-in RGB LEDs required)
// ─────────────────────────────────────────────────────────────────────────────
#if !defined(ARDUINO_ARDUINO_NANO33BLE)
  #error "This sketch targets the Arduino Nano 33 BLE Sense only."
#endif

// ─────────────────────────────────────────────────────────────────────────────
//  Includes
// ─────────────────────────────────────────────────────────────────────────────
#include <climits>         // INT32_MIN, INT32_MAX

#include <PDM.h>
#include <TensorFlowLite.h>

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/experimental/microfrontend/lib/frontend.h"
#include "tensorflow/lite/experimental/microfrontend/lib/frontend_util.h"

#include "model_data.h"   // g_model_data[]

// ─────────────────────────────────────────────────────────────────────────────
//  Section 1 — Feature / audio constants
//  (mirrors micro_features_micro_model_settings.h)
// ─────────────────────────────────────────────────────────────────────────────

// PDM runs at 16 kHz mono (16-bit samples).
// FFT window: 25 ms × 16 samples/ms = 400 samples → next power-of-two = 512.
constexpr int kMaxAudioSampleSize    = 512;
constexpr int kAudioSampleFrequency  = 16000;  // Hz

// Spectrogram layout — must match the training pipeline exactly.
constexpr int kFeatureSliceSize       = 32;    // mel-filterbank channels per slice
constexpr int kFeatureSliceCount      = 49;    // slices in one 1-second window
constexpr int kFeatureElementCount    = kFeatureSliceSize * kFeatureSliceCount; // 1568
constexpr int kFeatureSliceStrideMs   = 20;    // ms between consecutive slice starts
constexpr int kFeatureSliceDurationMs = 25;    // ms of audio analysed per slice

// ─────────────────────────────────────────────────────────────────────────────
//  Section 2 — Label definitions
// ─────────────────────────────────────────────────────────────────────────────

constexpr int kLabelCount   = 5;
constexpr int kNoiseIndex   = 0;
constexpr int kUnknownIndex = 1;
constexpr int kHeyNanoIndex = 2;
constexpr int kOnIndex      = 3;
constexpr int kOffIndex     = 4;

// All label pointers live here; pointer-equality comparisons in the score
// averaging code rely on every label always referencing this array.
const char* const kLabels[kLabelCount] = {
  "noise",    // 0
  "unknown",  // 1
  "heynano",  // 2
  "on",       // 3
  "off"       // 4
};

// ─────────────────────────────────────────────────────────────────────────────
//  Section 2b — Shared types
//
//  Declared here — before any function definitions — so that the Arduino IDE's
//  auto-generated function prototypes (inserted near the top of the translation
//  unit) can see these types when QueueAt() and ProcessScores() are prototyped.
// ─────────────────────────────────────────────────────────────────────────────

// One inference result stored in the averaging queue.
struct InferenceResult {
  int32_t time_ms;
  int8_t  scores[kLabelCount];
};

// Per-stage suppression bookkeeping.  The caller manages one instance and
// resets it whenever the application transitions between IDLE and ACTIVATED.
struct RecognitionState {
  const char* prev_label;      // pointer into kLabels[]
  int32_t     prev_label_time; // ms timestamp of the last fired detection

  RecognitionState()
      : prev_label(kLabels[kNoiseIndex]),
        prev_label_time(INT32_MIN) {}

  void Reset() {
    prev_label      = kLabels[kNoiseIndex];
    prev_label_time = INT32_MIN;
  }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Section 3 — Detection / averaging parameters
// ─────────────────────────────────────────────────────────────────────────────

// IDLE-state averaging window (ms).  Longer = smoother, higher latency.
constexpr int32_t kIdleWindowMs        = 1000;

// ACTIVATED-state averaging window (ms).  Shorter = faster response to on/off.
constexpr int32_t kActivatedWindowMs   = 600;

// Minimum inference count in the window before we trust the average.
constexpr int32_t kMinResultCount      = 3;

// Detection threshold (0–255, uint8).  The averaged score of the winning class
// must exceed this to fire.  200 matches the TFLite Micro default.
constexpr uint8_t kDetectionThreshold  = 200;

// After any keyword fires, ignore further detections for this many ms.
// Prevents the same utterance being counted twice.
constexpr int32_t kSuppressionMs       = 1500;

// Time (ms) to wait for "on"/"off" after "heynano" before giving up.
constexpr int32_t kActivationTimeoutMs = 3000;

// Duration (ms) the green LED stays on to confirm "heynano" was heard.
constexpr int32_t kActivationLEDMs     = 500;

// ─────────────────────────────────────────────────────────────────────────────
//  Section 4 — Audio capture
//  (mirrors arduino_audio_provider.cpp)
// ─────────────────────────────────────────────────────────────────────────────

namespace {

// The PDM library DMAs kPdmBufferSize bytes per callback (~16 ms @ 16 kHz).
// The ring buffer is 16× larger so the main loop can safely lag behind.
constexpr int kPdmBufferSize          = 512;                       // bytes / DMA transfer
constexpr int kAudioCaptureBufferSize = kPdmBufferSize * 16 / 2;  // in int16 samples

int16_t g_audio_capture_buffer[kAudioCaptureBufferSize];
int16_t g_audio_output_buffer[kMaxAudioSampleSize];

// Updated from the PDM ISR; read (as volatile) from the main loop.
volatile int32_t g_latest_audio_timestamp = 0;

bool g_is_audio_initialized = false;

} // namespace

// PDM DMA callback — runs in interrupt context, must be fast and lock-free.
void CaptureSamples() {
  const int kSamplesPerCallback = kPdmBufferSize / 2;  // bytes → int16 samples

  // Absolute timestamp (ms) of the last sample in this transfer.
  const int32_t time_in_ms =
      g_latest_audio_timestamp +
      (kSamplesPerCallback / (kAudioSampleFrequency / 1000));

  // Map the absolute sample index to a slot in the ring buffer.
  const int32_t start_sample  =
      g_latest_audio_timestamp * (kAudioSampleFrequency / 1000);
  const int capture_index = start_sample % kAudioCaptureBufferSize;

  // DMA read directly into the ring buffer.
  const int bytes_read = PDM.read(
      reinterpret_cast<uint8_t*>(g_audio_capture_buffer + capture_index),
      kPdmBufferSize);

  if (bytes_read != kPdmBufferSize) {
    // Short read — halt so the developer can investigate.
    while (true) {}
  }

  g_latest_audio_timestamp = time_in_ms;
}

// Start the PDM microphone.  Safe to call multiple times.
bool InitAudioRecording() {
  if (g_is_audio_initialized) return true;

  PDM.onReceive(CaptureSamples);

  // 1 channel (mono), 16 kHz
  if (!PDM.begin(1, kAudioSampleFrequency)) {
    Serial.println("ERROR: PDM.begin() failed.");
    return false;
  }

  // Gain: hardware min (-20 dB) + 13 steps ≈ -13.5 dB total.
  // Increase toward 43 if the mic is too quiet in your environment.
  PDM.setGain(40);

  // Block until the first DMA transfer completes.
  while (g_latest_audio_timestamp == 0) {}

  g_is_audio_initialized = true;
  return true;
}

// Copy [start_ms, start_ms + duration_ms) of ring-buffer audio into
// g_audio_output_buffer and return a pointer + sample count to the caller.
bool GetAudioSamples(int32_t start_ms, int32_t duration_ms,
                     int* out_size, int16_t** out_samples) {
  const int start_offset          = start_ms    * (kAudioSampleFrequency / 1000);
  const int duration_sample_count = duration_ms * (kAudioSampleFrequency / 1000);

  for (int i = 0; i < duration_sample_count; ++i) {
    const int capture_index = (start_offset + i) % kAudioCaptureBufferSize;
    g_audio_output_buffer[i] = g_audio_capture_buffer[capture_index];
  }

  *out_size    = duration_sample_count;
  *out_samples = g_audio_output_buffer;
  return true;
}

// Return the timestamp (ms) of the most recently captured audio sample.
int32_t LatestAudioTimestamp() {
  return g_latest_audio_timestamp;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section 5 — Microfrontend feature generator
//  (mirrors micro_features_micro_features_generator.cpp)
// ─────────────────────────────────────────────────────────────────────────────

namespace {
  FrontendState g_frontend_state;
  bool          g_frontend_initialized = false;
} // namespace

// Configure and initialise the microfrontend pipeline.
// Must be called once before the first GenerateMicroFeatures().
bool InitializeMicroFeatures() {
  FrontendConfig config;

  // Analysis window and stride — must match training.
  config.window.size_ms      = kFeatureSliceDurationMs;  // 25 ms
  config.window.step_size_ms = kFeatureSliceStrideMs;    // 20 ms

  // Mel filterbank: 32 channels, 125–7500 Hz.
  config.filterbank.num_channels     = kFeatureSliceSize;  // 32
  config.filterbank.lower_band_limit = 125.0f;
  config.filterbank.upper_band_limit = 7500.0f;

  // Noise reduction (SRNN-style).
  config.noise_reduction.smoothing_bits       = 10;
  config.noise_reduction.even_smoothing       = 0.025f;
  config.noise_reduction.odd_smoothing        = 0.06f;
  config.noise_reduction.min_signal_remaining = 0.05f;

  // PCAN automatic gain control.
  config.pcan_gain_control.enable_pcan = 1;
  config.pcan_gain_control.strength    = 0.95f;
  config.pcan_gain_control.offset      = 80.0f;
  config.pcan_gain_control.gain_bits   = 21;

  // Log-scale compression.
  config.log_scale.enable_log  = 1;
  config.log_scale.scale_shift = 6;

  if (!FrontendPopulateState(&config, &g_frontend_state, kAudioSampleFrequency)) {
    Serial.println("ERROR: FrontendPopulateState() failed.");
    return false;
  }

  g_frontend_initialized = true;
  return true;
}

// Convert one slice of raw 16-bit PCM into quantised int8 filterbank features.
//
// Scaling matches the training pipeline:
//   raw frontend  →  uint16 [0, ~670]
//   / 25.6        →  float  [0, ~26.0]   (historical normalisation)
//   / 26.0 × 256 − 128  →  int8 [-128, 127]
//
// Integer equivalent: out = (raw × 256) / round(25.6 × 26.0) − 128
bool GenerateMicroFeatures(const int16_t* input, int input_size,
                           int output_size, int8_t* output,
                           size_t* num_samples_read) {
  size_t samples_read = 0;
  FrontendOutput frontend_output = FrontendProcessSamples(
      &g_frontend_state, input, input_size, &samples_read);

  if (num_samples_read) *num_samples_read = samples_read;

  constexpr int32_t kValueScale = 256;
  // round(25.6 × 26.0) = 666
  constexpr int32_t kValueDiv   = static_cast<int32_t>(25.6f * 26.0f + 0.5f);

  for (size_t i = 0; i < frontend_output.size; ++i) {
    int32_t value =
        ((frontend_output.values[i] * kValueScale) + (kValueDiv / 2)) / kValueDiv;
    value -= 128;
    if (value < -128) value = -128;
    if (value >  127) value =  127;
    output[i] = static_cast<int8_t>(value);
  }
  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section 6 — Feature provider (sliding spectrogram window)
//  (mirrors feature_provider.cpp)
// ─────────────────────────────────────────────────────────────────────────────

// Flat row-major array: kFeatureSliceCount rows × kFeatureSliceSize columns.
// Row 0 = oldest slice; row kFeatureSliceCount-1 = newest slice.
// Total: 49 × 32 = 1568 bytes — fed directly into the model input tensor.
int8_t g_feature_data[kFeatureElementCount];

namespace {
  int32_t s_feature_last_time_ms = 0;
  bool    s_feature_first_run    = true;
} // namespace

// Advance the spectrogram by as many new 20 ms slices as the elapsed time
// allows, reusing older slices with a memmove-style shift.
//
// Returns:  number of newly computed slices (≥ 0)
//          -1 on any error
int PopulateFeatureData(int32_t current_time_ms) {
  // First call: initialise the frontend and record the base timestamp.
  if (s_feature_first_run) {
    if (!InitializeMicroFeatures()) return -1;
    s_feature_first_run    = false;
    s_feature_last_time_ms = current_time_ms;
    return 0;
  }

  // How many 20 ms strides fit into the elapsed time?
  // Formula mirrors feature_provider.cpp exactly.
  const int32_t elapsed = current_time_ms - s_feature_last_time_ms;
  int slices_needed =
      ((((elapsed - kFeatureSliceDurationMs) * kFeatureSliceStrideMs) /
          kFeatureSliceStrideMs) +
        kFeatureSliceStrideMs) /
      kFeatureSliceStrideMs;

  if (slices_needed <= 0)                 return 0;
  if (slices_needed > kFeatureSliceCount) slices_needed = kFeatureSliceCount;

  const int last_step      = s_feature_last_time_ms / kFeatureSliceStrideMs;
  const int slices_to_keep = kFeatureSliceCount - slices_needed;
  const int slices_to_drop = kFeatureSliceCount - slices_to_keep;

  // Shift surviving slices toward the beginning of the buffer.
  //
  // Before (last_time=80ms, current_time=120ms):
  //   [data@20ms][data@40ms][data@60ms][data@80ms]
  // After:
  //   [data@60ms][data@80ms][  empty ][  empty  ]
  if (slices_to_keep > 0) {
    for (int dest = 0; dest < slices_to_keep; ++dest) {
      const int src = dest + slices_to_drop;
      memcpy(g_feature_data + dest * kFeatureSliceSize,
             g_feature_data + src  * kFeatureSliceSize,
             kFeatureSliceSize);
    }
  }

  // Fill the now-empty tail slots with freshly computed features.
  for (int new_slice = slices_to_keep; new_slice < kFeatureSliceCount; ++new_slice) {
    const int     new_step       = last_step + (new_slice - slices_to_keep);
    const int32_t slice_start_ms = new_step * kFeatureSliceStrideMs;

    int16_t* audio_samples      = nullptr;
    int      audio_samples_size = 0;

    if (!GetAudioSamples(slice_start_ms, kFeatureSliceDurationMs,
                         &audio_samples_size, &audio_samples)) {
      Serial.println("ERROR: GetAudioSamples() failed.");
      return -1;
    }

    // 25 ms × 16 samples/ms = 400 samples
    constexpr int kWantedSamples =
        kFeatureSliceDurationMs * (kAudioSampleFrequency / 1000);
    if (audio_samples_size != kWantedSamples) {
      Serial.print("ERROR: audio size mismatch (got ");
      Serial.print(audio_samples_size);
      Serial.print(", want ");
      Serial.print(kWantedSamples);
      Serial.println(").");
      return -1;
    }

    int8_t* dest_slice   = g_feature_data + new_slice * kFeatureSliceSize;
    size_t  samples_read = 0;

    if (!GenerateMicroFeatures(audio_samples, audio_samples_size,
                               kFeatureSliceSize, dest_slice, &samples_read)) {
      Serial.println("ERROR: GenerateMicroFeatures() failed.");
      return -1;
    }
  }

  s_feature_last_time_ms = current_time_ms;
  return slices_needed;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section 7 — Score averaging queue
//  (mirrors PreviousResultsQueue + RecognizeCommands in recognize_commands.h/cpp)
//
//  The queue stores raw int8 output tensors tagged with timestamps.
//  ProcessScores():
//    1. Pushes the latest inference result.
//    2. Prunes entries older than the averaging window.
//    3. Converts int8 → uint8 (+ 128), averages across all entries.
//    4. Applies detection threshold + suppression to decide if a new
//       command has been recognised.
// ─────────────────────────────────────────────────────────────────────────────

namespace {
  // 50 slots covers 1000 ms @ 20 ms stride with a small margin.
  constexpr int kMaxQueueSize = 50;
  InferenceResult g_result_queue[kMaxQueueSize];
  int g_queue_front = 0;
  int g_queue_size  = 0;
} // namespace

// ── Queue helpers ─────────────────────────────────────────────────────────────

void QueueClear() {
  g_queue_front = 0;
  g_queue_size  = 0;
}

void QueuePushBack(int32_t time_ms, const int8_t* scores) {
  if (g_queue_size >= kMaxQueueSize) {
    // Overflow — silently drop the oldest entry to make room.
    g_queue_front = (g_queue_front + 1) % kMaxQueueSize;
    --g_queue_size;
  }
  const int back_index = (g_queue_front + g_queue_size) % kMaxQueueSize;
  g_result_queue[back_index].time_ms = time_ms;
  for (int i = 0; i < kLabelCount; ++i)
    g_result_queue[back_index].scores[i] = scores[i];
  ++g_queue_size;
}

void QueuePopFront() {
  if (g_queue_size <= 0) return;
  g_queue_front = (g_queue_front + 1) % kMaxQueueSize;
  --g_queue_size;
}

const InferenceResult& QueueAt(int offset) {
  return g_result_queue[(g_queue_front + offset) % kMaxQueueSize];
}

// ── Main averaging + recognition function ────────────────────────────────────

// Push new_scores into the queue, prune old entries, average, and determine
// whether a new command has been confidently recognised.
//
// Parameters:
//   current_time_ms   — timestamp of this inference result
//   new_scores        — raw int8 output tensor (kLabelCount elements)
//   average_window_ms — how far back (ms) to average; tune per application
//   suppression_ms    — silence period after any detection
//   min_count         — minimum queue depth before trusting the average
//   detection_thresh  — uint8 averaged-score threshold (0–255)
//   rstate            — suppression state (caller manages; reset between stages)
//
// Outputs (set on every call):
//   *found_label  — current top label (pointer into kLabels[])
//   *top_score    — averaged uint8 score of the winning class
//   *is_new_cmd   — true iff a new command just crossed the threshold
//
// Returns the index into kLabels[] of the winning class.
int ProcessScores(int32_t       current_time_ms,
                  const int8_t* new_scores,
                  int32_t       average_window_ms,
                  int32_t       suppression_ms,
                  int32_t       min_count,
                  uint8_t       detection_thresh,
                  RecognitionState* rstate,
                  const char**  found_label,
                  uint8_t*      top_score,
                  bool*         is_new_cmd) {
  // 1. Enqueue the latest result.
  QueuePushBack(current_time_ms, new_scores);

  // 2. Prune results that have fallen outside the averaging window.
  const int64_t time_limit = current_time_ms - average_window_ms;
  while (g_queue_size > 0 && QueueAt(0).time_ms < time_limit) {
    QueuePopFront();
  }

  // 3. Require a minimum number of results and a minimum window coverage
  //    before trusting the average (prevents false fires at startup).
  const int64_t earliest_time     = (g_queue_size > 0) ? QueueAt(0).time_ms : current_time_ms;
  const int64_t samples_duration  = current_time_ms - earliest_time;
  if ((g_queue_size < min_count) ||
      (samples_duration < (average_window_ms / 4))) {
    *found_label = rstate->prev_label;
    *top_score   = 0;
    *is_new_cmd  = false;
    return kNoiseIndex;
  }

  // 4. Compute per-class averages.
  //    int8 scores are mapped to [0, 255] by adding 128 before averaging,
  //    matching the TFLite Micro RecognizeCommands implementation exactly.
  int32_t avg_scores[kLabelCount] = {};
  for (int offset = 0; offset < g_queue_size; ++offset) {
    const InferenceResult& r = QueueAt(offset);
    for (int i = 0; i < kLabelCount; ++i) {
      avg_scores[i] += static_cast<int32_t>(r.scores[i]) + 128;
    }
  }
  for (int i = 0; i < kLabelCount; ++i) {
    avg_scores[i] /= g_queue_size;
  }

  // 5. Find the winning class.
  int     top_index = 0;
  int32_t top_val   = 0;
  for (int i = 0; i < kLabelCount; ++i) {
    if (avg_scores[i] > top_val) {
      top_val   = avg_scores[i];
      top_index = i;
    }
  }
  const char* top_label = kLabels[top_index];

  // 6. Suppression logic — mirrors RecognizeCommands::ProcessLatestResults().
  //    If we have never fired (prev_label == noise OR prev_label_time == INT32_MIN)
  //    then time_since_last is treated as infinite, allowing the first detection.
  int64_t time_since_last;
  if ((rstate->prev_label      == kLabels[kNoiseIndex]) ||
      (rstate->prev_label_time == INT32_MIN)) {
    time_since_last = INT32_MAX;
  } else {
    time_since_last = current_time_ms - rstate->prev_label_time;
  }

  if ((top_val > detection_thresh) &&
      ((top_label != rstate->prev_label) ||
       (time_since_last > suppression_ms))) {
    rstate->prev_label      = top_label;
    rstate->prev_label_time = current_time_ms;
    *is_new_cmd             = true;
  } else {
    *is_new_cmd = false;
  }

  *found_label = top_label;
  *top_score   = static_cast<uint8_t>(top_val);
  return top_index;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section 8 — TFLite Micro interpreter
// ─────────────────────────────────────────────────────────────────────────────

namespace {
  const tflite::Model*      g_model       = nullptr;
  tflite::MicroInterpreter* g_interpreter = nullptr;
  TfLiteTensor*             g_input       = nullptr;
  TfLiteTensor*             g_output      = nullptr;

  // 40 kB matches the allocation in kws_inference_test.ino.
  // Increase if AllocateTensors() fails; decrease only if RAM is critical.
  constexpr int kTensorArenaSize = 64 * 1024;
  alignas(16) uint8_t tensor_arena[kTensorArenaSize];
} // namespace

// ─────────────────────────────────────────────────────────────────────────────
//  Section 9 — Application state machine
// ─────────────────────────────────────────────────────────────────────────────

enum AppState {
  // Continuously running inference, looking for "heynano".
  IDLE,
  // "heynano" was detected; now listening for "on" or "off" within the timeout.
  ACTIVATED
};

namespace {
  AppState         g_app_state          = IDLE;
  int32_t          g_activation_time_ms = 0;
  RecognitionState g_rstate;            // suppression bookkeeping
} // namespace

// ─────────────────────────────────────────────────────────────────────────────
//  Section 10 — LED helpers
//
//  LED_BUILTIN  — white, active HIGH  — controlled by "on"/"off" commands
//  LEDG         — green, active LOW   — brief pulse to confirm "heynano"
//  LEDR         — red,   active LOW   — unused (available for extension)
//  LEDB         — blue,  active LOW   — unused (available for extension)
// ─────────────────────────────────────────────────────────────────────────────

namespace {
  bool    g_builtin_led_on      = false;  // current commanded state
  int32_t g_ledg_off_time_ms    = 0;      // 0 = LEDG already off
} // namespace

void InitLEDs() {
  pinMode(LED_BUILTIN, OUTPUT);
  pinMode(LEDR,        OUTPUT);
  pinMode(LEDG,        OUTPUT);
  pinMode(LEDB,        OUTPUT);

  // Active-LOW RGB LEDs: HIGH = off.
  digitalWrite(LEDR, HIGH);
  digitalWrite(LEDG, HIGH);
  digitalWrite(LEDB, HIGH);

  // Built-in LED: LOW = off.
  digitalWrite(LED_BUILTIN, LOW);
}

// Call from setup() / state machine when "heynano" is detected.
// Lights LEDG for kActivationLEDMs milliseconds (non-blocking).
void SignalActivation(int32_t now_ms) {
  digitalWrite(LEDG, LOW);                          // green on (active LOW)
  g_ledg_off_time_ms = now_ms + kActivationLEDMs;  // schedule auto-off
}

// Call once per loop() iteration.  Turns LEDG off when its deadline passes.
void UpdateLEDs(int32_t now_ms) {
  if (g_ledg_off_time_ms != 0 && now_ms >= g_ledg_off_time_ms) {
    digitalWrite(LEDG, HIGH);   // green off
    g_ledg_off_time_ms = 0;
  }
}

// Apply the current commanded state of LED_BUILTIN.
void ApplyBuiltinLED() {
  digitalWrite(LED_BUILTIN, g_builtin_led_on ? HIGH : LOW);
}

// ─────────────────────────────────────────────────────────────────────────────
//  setup()
// ─────────────────────────────────────────────────────────────────────────────

void setup() {
  Serial.begin(115200);
  // Wait up to 5 s for a Serial monitor (mirrors micro_speech DSP sketch).
  for (int i = 0; i < 50 && !Serial; ++i) delay(100);

  Serial.println("=== HeyNano Voice Control ===");
  Serial.print("Feature buffer: ");
  Serial.print(kFeatureSliceCount);
  Serial.print(" × ");
  Serial.print(kFeatureSliceSize);
  Serial.print(" = ");
  Serial.print(kFeatureElementCount);
  Serial.println(" int8 elements");

  tflite::InitializeTarget();
  InitLEDs();

  // ── TFLite Micro setup ────────────────────────────────────────────────────

  g_model = tflite::GetModel(g_model_data);
  if (g_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("ERROR: Model schema version ");
    Serial.print(g_model->version());
    Serial.print(" != supported version ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    while (true) {}
  }

  // Register only the ops the model actually uses.
  // <10> must equal the number of AddXxx() calls below.
  static tflite::MicroMutableOpResolver<10> resolver;
  if (resolver.AddShape()          != kTfLiteOk) { Serial.println("ERROR: AddShape");          while(true){} }
  if (resolver.AddStridedSlice()   != kTfLiteOk) { Serial.println("ERROR: AddStridedSlice");   while(true){} }
  if (resolver.AddPack()           != kTfLiteOk) { Serial.println("ERROR: AddPack");           while(true){} }
  if (resolver.AddReshape()        != kTfLiteOk) { Serial.println("ERROR: AddReshape");        while(true){} }
  if (resolver.AddExpandDims()     != kTfLiteOk) { Serial.println("ERROR: AddExpandDims");     while(true){} }
  if (resolver.AddConv2D()         != kTfLiteOk) { Serial.println("ERROR: AddConv2D");         while(true){} }
  if (resolver.AddFullyConnected() != kTfLiteOk) { Serial.println("ERROR: AddFullyConnected"); while(true){} }
  if (resolver.AddMean()           != kTfLiteOk) { Serial.println("ERROR: AddMean");           while(true){} }
  if (resolver.AddSoftmax()        != kTfLiteOk) { Serial.println("ERROR: AddSoftmax");        while(true){} }
  if (resolver.AddMaxPool2D()      != kTfLiteOk) { Serial.println("ERROR: AddMaxPool2D");      while(true){} }

  static tflite::MicroInterpreter static_interpreter(
      g_model, resolver, tensor_arena, kTensorArenaSize);
  g_interpreter = &static_interpreter;

  if (g_interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors() failed!  Try increasing kTensorArenaSize.");
    while (true) {}
  }

  g_input  = g_interpreter->input(0);
  g_output = g_interpreter->output(0);

  // Sanity-check the input tensor shape.
  if ((g_input->dims->size != 2) ||
      (g_input->dims->data[0] != 1) ||
      (g_input->dims->data[1] != kFeatureElementCount) ||
      (g_input->type != kTfLiteInt8)) {
    Serial.println("ERROR: Unexpected input tensor shape or type.");
    Serial.print("  dims->size = "); Serial.println(g_input->dims->size);
    Serial.print("  dims[0]   = "); Serial.println(g_input->dims->data[0]);
    Serial.print("  dims[1]   = "); Serial.println(g_input->dims->data[1]);
    Serial.print("  type      = "); Serial.println(g_input->type);
    while (true) {}
  }

  Serial.print("Tensor arena used: ");
  Serial.print(g_interpreter->arena_used_bytes());
  Serial.println(" bytes");

  // ── Audio setup ───────────────────────────────────────────────────────────
  if (!InitAudioRecording()) {
    Serial.println("ERROR: Audio init failed — halting.");
    while (true) {}
  }

  Serial.println("PDM microphone ready.");
  Serial.println("Listening for 'heynano'...");
}

// ─────────────────────────────────────────────────────────────────────────────
//  loop()
// ─────────────────────────────────────────────────────────────────────────────

void loop() {
  static int32_t previous_audio_timestamp = 0;

  // ── 1. Gate on new audio ──────────────────────────────────────────────────
  const int32_t now_ms = LatestAudioTimestamp();
  if (now_ms == previous_audio_timestamp) return;   // nothing new yet
  previous_audio_timestamp = now_ms;

  // ── 2. Advance the spectrogram ────────────────────────────────────────────
  const int new_slices = PopulateFeatureData(now_ms);
  if (new_slices < 0) {
    Serial.println("ERROR: Feature extraction failed — skipping frame.");
    return;
  }
  if (new_slices == 0) return;  // not enough elapsed time for a new slice yet

  // ── 3. Run inference ──────────────────────────────────────────────────────
  // g_feature_data is row-major [49][32] int8 — copy straight into the tensor.
  memcpy(g_input->data.int8, g_feature_data, kFeatureElementCount);

  if (g_interpreter->Invoke() != kTfLiteOk) {
    Serial.println("ERROR: Inference failed!");
    return;
  }

  // ── 4. Score averaging ────────────────────────────────────────────────────
  // Use a longer window in IDLE (smoother "heynano" detection) and a shorter
  // window in ACTIVATED (faster response to "on"/"off").
  const int32_t window_ms =
      (g_app_state == IDLE) ? kIdleWindowMs : kActivatedWindowMs;

  const char* found_label = nullptr;
  uint8_t     top_score   = 0;
  bool        is_new_cmd  = false;

  const int top_index = ProcessScores(
      now_ms,
      g_output->data.int8,
      window_ms,
      kSuppressionMs,
      kMinResultCount,
      kDetectionThreshold,
      &g_rstate,
      &found_label,
      &top_score,
      &is_new_cmd);

  // ── 5. Update LEDs (non-blocking) ─────────────────────────────────────────
  UpdateLEDs(now_ms);

  // ── 6. State machine ──────────────────────────────────────────────────────
  switch (g_app_state) {

    // ──────────────────────────────────────────────────────────────────────
    case IDLE:
    // ──────────────────────────────────────────────────────────────────────
      if (is_new_cmd && top_index == kHeyNanoIndex) {
        Serial.print("Heard 'heynano' (");
        Serial.print(top_score);
        Serial.print("/255) @");
        Serial.print(now_ms);
        Serial.println(" ms — say 'on' or 'off'...");

        g_app_state          = ACTIVATED;
        g_activation_time_ms = now_ms;

        // Flash the green LED as activation feedback.
        SignalActivation(now_ms);

        // Clear the queue so stale "heynano" scores from IDLE do not
        // bleed into the ACTIVATED detection window.
        QueueClear();
        g_rstate.Reset();
      }
      break;

    // ──────────────────────────────────────────────────────────────────────
    case ACTIVATED:
    // ──────────────────────────────────────────────────────────────────────

      // Check for timeout first.
      if ((now_ms - g_activation_time_ms) > kActivationTimeoutMs) {
        Serial.println("Activation timed out — back to IDLE.");
        g_app_state = IDLE;
        QueueClear();
        g_rstate.Reset();
        break;
      }

      // Process a recognised command.
      if (is_new_cmd) {
        if (top_index == kOnIndex) {
          Serial.print("Heard 'on' (");
          Serial.print(top_score);
          Serial.print("/255) @");
          Serial.print(now_ms);
          Serial.println(" ms — LED ON.");
          g_builtin_led_on = true;
          ApplyBuiltinLED();
          g_app_state = IDLE;
          QueueClear();
          g_rstate.Reset();

        } else if (top_index == kOffIndex) {
          Serial.print("Heard 'off' (");
          Serial.print(top_score);
          Serial.print("/255) @");
          Serial.print(now_ms);
          Serial.println(" ms — LED OFF.");
          g_builtin_led_on = false;
          ApplyBuiltinLED();
          g_app_state = IDLE;
          QueueClear();
          g_rstate.Reset();

        }
        // "noise" and "unknown" are simply ignored while ACTIVATED;
        // the timeout handles the case where neither word is ever heard.
      }
      break;

  } // switch (g_app_state)
}

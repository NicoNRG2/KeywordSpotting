"""
feature_extraction.py
=====================
Replica Python della pipeline DSP di micro_speech_dsp.ino
(TF Lite Micro Speech – Arduino Nano 33 BLE Sense).

Pipeline per ogni clip WAV:
  PCM 16 kHz → frame 25 ms / stride 20 ms → FFT 512 pt
  → Mel filterbank 32 ch (125–7500 Hz)
  → Noise reduction SRNN
  → PCAN gain control
  → Log scale
  → Quantizzazione int8  →  output (49, 32) int8

Output finale pronto per il training TensorFlow:
  X_train / X_test : np.ndarray  (N, 49, 32)  dtype=int8
  y_train / y_test : tf.Tensor   (N, num_classes)  one-hot int64
"""

# ─────────────────────────────────────────────────────────────────────────────
# Dipendenze
# ─────────────────────────────────────────────────────────────────────────────
import os
import glob
import numpy as np
import librosa
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, GaussianNoise, Reshape
from tensorflow.keras.layers import GlobalAveragePooling1D


# ─────────────────────────────────────────────────────────────────────────────
# Costanti del DSP  (identiche al .ino)
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_RATE         = 16_000     # Hz
FRAME_LEN_MS        = 25         # ms per slice  → 400 campioni
FRAME_STEP_MS       = 20         # ms di stride  → 320 campioni
N_MEL               = 32         # canali filterbank
FMIN                = 125.0      # Hz
FMAX                = 7_500.0    # Hz
N_SLICES            = 49         # slice nella finestra da 1 s
FFT_SIZE            = 512        # prossima pot. di 2 ≥ 400

FRAME_LEN           = FRAME_LEN_MS  * SAMPLE_RATE // 1000  # 400
FRAME_STEP          = FRAME_STEP_MS * SAMPLE_RATE // 1000  # 320

# Verifica: (16000 − 400) // 320 + 1 = 49  ✓
assert (SAMPLE_RATE - FRAME_LEN) // FRAME_STEP + 1 == N_SLICES, \
    "Mismatch numero di slices: controlla SAMPLE_RATE / FRAME_LEN / FRAME_STEP"

# Noise reduction (noise_reduction.c)
EVEN_SMOOTH         = 0.025
ODD_SMOOTH          = 0.06
MIN_SIG_REMAIN      = 0.05

# PCAN gain control (pcan_gain_control.c)
PCAN_STRENGTH       = 0.95
PCAN_OFFSET         = 80.0
PCAN_GAIN_BITS      = 21

# Log scale (log_scale.c)
LOG_SCALE_SHIFT     = 6          # shift prima del log  → divide per 64
LOG_SCALE_OUT_BITS  = 12         # bit dell'output      → moltiplica per 4096

# Quantizzazione int8  (GenerateMicroFeatures nel .ino)
#   raw uint16 ∈ [0, ~670]  →  int8 ∈ [-128, 127]
#   formula: value = (raw × 256 + 333) / 666 − 128
K_VALUE_SCALE       = 256
K_VALUE_DIV         = int(25.6 * 26.0 + 0.5)   # 666


# ─────────────────────────────────────────────────────────────────────────────
# Mel filterbank  (calcolato una sola volta all'avvio)
# ─────────────────────────────────────────────────────────────────────────────
# tf.signal.linear_to_mel_weight_matrix usa la stessa scala HTK del
# microfrontend TF Lite; restituisce shape (257, 32) già trasposta per @.
_MEL_FILTERBANK: np.ndarray = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins          = N_MEL,
    num_spectrogram_bins  = FFT_SIZE // 2 + 1,   # 257
    sample_rate           = SAMPLE_RATE,
    lower_edge_hertz      = FMIN,
    upper_edge_hertz      = FMAX,
    dtype                 = tf.float32,
).numpy()   # (257, 32)

# Finestra di Hann per i frame da 400 campioni (identica all'implementazione C)
_HANN_WINDOW: np.ndarray = np.hanning(FRAME_LEN).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline DSP
# ─────────────────────────────────────────────────────────────────────────────

def _power_spectrum(frames: np.ndarray) -> np.ndarray:
    """
    Applica finestra di Hann, zero-padding a 512 pt, FFT e spettro di potenza.

    Parametri
    ---------
    frames : (N_SLICES, FRAME_LEN) float32
        Campioni PCM scalati in range int16 (×32768).

    Ritorna
    -------
    (N_SLICES, 257) float32 – modulo quadro dei bin FFT mono-laterali.
    """
    windowed = frames * _HANN_WINDOW                          # (49, 400)
    padded   = np.zeros((N_SLICES, FFT_SIZE), dtype=np.float32)
    padded[:, :FRAME_LEN] = windowed
    spectra  = np.abs(np.fft.rfft(padded, axis=1)) ** 2      # (49, 257)
    return spectra


def _mel_energy(spectra: np.ndarray) -> np.ndarray:
    """
    Proietta lo spettro di potenza nei canali mel.

    Parametri
    ---------
    spectra : (N_SLICES, 257) float32

    Ritorna
    -------
    (N_SLICES, N_MEL) float32 – energia per canale mel (valori assoluti grandi).
    """
    return spectra @ _MEL_FILTERBANK   # (49, 32)


def _noise_reduction_and_pcan(mel: np.ndarray) -> np.ndarray:
    """
    Noise reduction SRNN + PCAN gain control frame per frame.

    Equivale all'esecuzione in sequenza di:
      NoiseReductionApply()  →  aggiorna noise_estimate, produce segnale ripulito
      PcanGainControlApply() →  normalizza usando lo stesso noise_estimate

    Parametri
    ---------
    mel : (N_SLICES, N_MEL) float32

    Ritorna
    -------
    (N_SLICES, N_MEL) float32 – output PCAN con valori tipicamente ∈ [0, ~20].
    """
    noise_est  = np.zeros(N_MEL, dtype=np.float64)
    pcan_out   = np.zeros_like(mel)

    for i in range(N_SLICES):
        # ── Noise reduction ──────────────────────────────────────
        # Il microfrontend C usa smoothing diverso per frame pari/dispari.
        alpha = EVEN_SMOOTH if (i % 2 == 0) else ODD_SMOOTH
        noise_est = (1.0 - alpha) * noise_est + alpha * mel[i].astype(np.float64)

        sig      = mel[i].astype(np.float64)
        denoised = np.maximum(
            sig - (1.0 - MIN_SIG_REMAIN) * noise_est,
            MIN_SIG_REMAIN * sig
        )

        # ── PCAN gain control ────────────────────────────────────
        # output = denoised / (noise_est^strength + offset)
        # Lo stesso noise_estimate usato sopra, come in pcan_gain_control.c
        denom         = np.power(np.maximum(noise_est, 1e-12), PCAN_STRENGTH) + PCAN_OFFSET
        pcan_out[i]   = (denoised / denom).astype(np.float32)

    return pcan_out


def _log_scale(pcan: np.ndarray) -> np.ndarray:
    """
    Log-scale compression (log_scale.c, scale_shift=6, out_bits=12).

      output = log2(1 + x / 2^scale_shift) × 2^out_bits
             = log2(1 + x / 64) × 4096

    Produce valori uint16 tipicamente ∈ [0, ~670].

    Parametri
    ---------
    pcan : (N_SLICES, N_MEL) float32

    Ritorna
    -------
    (N_SLICES, N_MEL) float32
    """
    scale_in  = float(1 << LOG_SCALE_SHIFT)    # 64.0
    scale_out = float(1 << LOG_SCALE_OUT_BITS)  # 4096.0
    return np.log2(1.0 + np.maximum(pcan, 0.0) / scale_in) * scale_out


def _quantize_int8(log_out: np.ndarray) -> np.ndarray:
    """
    Quantizzazione uint16 → int8 identica a GenerateMicroFeatures() nel .ino.

      value = (raw × 256 + 333) / 666 − 128
      clamp a [-128, 127]

    Parametri
    ---------
    log_out : (N_SLICES, N_MEL) float32  – valori raw ∈ [0, ~670]

    Ritorna
    -------
    (N_SLICES, N_MEL) int8
    """
    raw   = log_out.astype(np.int64)
    value = (raw * K_VALUE_SCALE + K_VALUE_DIV // 2) // K_VALUE_DIV - 128
    return np.clip(value, -128, 127).astype(np.int8)


def extract_features(wav_path: str) -> np.ndarray:
    """
    Carica un file WAV e restituisce il tensore di feature int8.

    L'output è identico al buffer g_feature_data[kFeatureElementCount]
    dell'Arduino (49 × 32 = 1568 byte).

    Parametri
    ---------
    wav_path : str – percorso al file .wav (qualsiasi sample rate / canali).

    Ritorna
    -------
    (49, 32) int8 – spectrogram quantizzato.
    """
    # ── Carica e normalizza l'audio ───────────────────────────────────────────
    # librosa resampla a 16 kHz e converte in mono automaticamente.
    audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True, duration=1.0)

    # Padding / trimming per garantire esattamente 1 s = 16000 campioni
    target_len = SAMPLE_RATE
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]

    # Scala a range int16 come i campioni PDM dell'Arduino
    audio = (audio * 32768.0).astype(np.float32)

    # ── Framing ───────────────────────────────────────────────────────────────
    # sliding_window_view + stride manuale = equivalente di librosa.util.frame
    frames = np.lib.stride_tricks.sliding_window_view(audio, FRAME_LEN)[::FRAME_STEP]
    frames = frames[:N_SLICES].copy()   # (49, 400)

    # ── DSP pipeline ─────────────────────────────────────────────────────────
    spectra  = _power_spectrum(frames)           # (49, 257) float32
    mel      = _mel_energy(spectra)              # (49, 32)  float32
    pcan     = _noise_reduction_and_pcan(mel)    # (49, 32)  float32
    log_out  = _log_scale(pcan)                  # (49, 32)  float32
    features = _quantize_int8(log_out)           # (49, 32)  int8

    return features


# ─────────────────────────────────────────────────────────────────────────────
# Caricamento dataset (struttura identica al codice nel tuo notebook)
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(folder: str):
    """
    Scansiona *folder* per file .wav.
    Il label è la parte del filename prima del primo punto.

    Ritorna
    -------
    file_paths : list[str]
    labels     : list[str]
    """
    file_paths, labels = [], []
    for file in glob.glob(os.path.join(folder, "*.wav")):
        label = os.path.basename(file).split(".")[0]
        file_paths.append(file)
        labels.append(label)
    return file_paths, labels


def process_split(file_paths: list, split_name: str) -> np.ndarray:
    """
    Estrae le feature da tutti i file di un split con barra di avanzamento.

    Parametri
    ---------
    file_paths : list[str]
    split_name : str – usato solo per il titolo della progress bar

    Ritorna
    -------
    (N, 49, 32) int8
    """
    features = []
    errors   = []

    for path in tqdm(file_paths, desc=f"Extracting {split_name}", unit="file"):
        try:
            features.append(extract_features(path))
        except Exception as exc:
            errors.append((path, str(exc)))
            # Inserisce un tensore di zeri per non rompere l'allineamento con y
            features.append(np.zeros((N_SLICES, N_MEL), dtype=np.int8))

    if errors:
        print(f"\n[WARN] {len(errors)} file non processati in {split_name}:")
        for p, e in errors[:5]:
            print(f"  {p}: {e}")
        if len(errors) > 5:
            print(f"  … e altri {len(errors) - 5}")

    return np.stack(features, axis=0)   # (N, 49, 32)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    dataset_path = "dataset"
    train_dir    = os.path.join(dataset_path, "training")
    test_dir     = os.path.join(dataset_path, "testing")

    # ── Carica percorsi e label testuali ─────────────────────────────────────
    X_train_paths, y_train_str = load_dataset(train_dir)
    X_test_paths,  y_test_str  = load_dataset(test_dir)

    print(f"Training samples : {len(X_train_paths)}")
    print(f"Testing samples  : {len(X_test_paths)}")

    # ── StringLookup → one-hot (identico al notebook originale) ──────────────
    lookup = tf.keras.layers.StringLookup(
        output_mode    = "one_hot",
        num_oov_indices = 0,
    )
    lookup.adapt(y_train_str)

    y_train = lookup(y_train_str)   # (N_train, num_classes) int64
    y_test  = lookup(y_test_str)    # (N_test,  num_classes) int64

    labels      = lookup.get_vocabulary()
    num_classes = len(labels)
    print(f"Classi ({num_classes}): {labels}")

    # ── Estrazione feature ────────────────────────────────────────────────────
    X_train = process_split(X_train_paths, "train")   # (N_train, 49, 32) int8
    X_test  = process_split(X_test_paths,  "test")    # (N_test,  49, 32) int8

    print(f"\nX_train shape : {X_train.shape}  dtype={X_train.dtype}")
    print(f"X_test  shape : {X_test.shape}   dtype={X_test.dtype}")
    print(f"y_train shape : {y_train.shape}  dtype={y_train.dtype}")
    print(f"y_test  shape : {y_test.shape}   dtype={y_test.dtype}")

    # ── Salva su disco ────────────────────────────────────────────────────────
    os.makedirs("features", exist_ok=True)
    np.save("features/X_train.npy", X_train)
    np.save("features/X_test.npy",  X_test)
    np.save("features/y_train.npy", y_train.numpy())
    np.save("features/y_test.npy",  y_test.numpy())
    np.save("features/labels.npy",  np.array(labels))

    print("\nFeature salvate in features/")
    print("  X_train.npy  X_test.npy  y_train.npy  y_test.npy  labels.npy")

    # ─────────────────────────────────────────────────────────────────────────
    # Costruzione tf.data.Dataset pronti per il training
    # ─────────────────────────────────────────────────────────────────────────

    BATCH_SIZE  = 32
    AUTOTUNE    = tf.data.AUTOTUNE

    # Converti X in float32 (range approssimativo [-1, 1] dopo rescaling)
    # Il modello riceve (49, 32, 1) – aggiunta dim canale per Conv2D.
    def make_dataset(X: np.ndarray, y: tf.Tensor, shuffle: bool) -> tf.data.Dataset:
        # float32 normalizzato in [-1, 1] per facilitare il training
        X_f = (X.astype(np.float32) / 128.0).reshape(len(X), -1)  # (N, 1568)
        ds  = tf.data.Dataset.from_tensor_slices((X_f, y))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(X), reshuffle_each_iteration=True)
        return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    train_ds = make_dataset(X_train, y_train, shuffle=True)
    test_ds  = make_dataset(X_test,  y_test,  shuffle=False)

    print("\ntrain_ds:", train_ds)
    print("test_ds :", test_ds)

    # ─────────────────────────────────────────────────────────────────────────
    # Esempio di modello CNN compatibile con il tensore (49, 32, 1)
    # (opzionale – decommentare per un training di verifica)
    # ─────────────────────────────────────────────────────────────────────────

    model = Sequential()
    model.add(tf.keras.Input(shape=(1568,)))
    model.add(GaussianNoise(0.1))
    model.add(Reshape((49, 32)))
    model.add(Conv1D(8, kernel_size=5, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv1D(16, kernel_size=5, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
    model.add(Dropout(0.25))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(num_classes, activation='softmax')) 
    
    model.compile(
        optimizer = "adam",
        loss      = "categorical_crossentropy",
        metrics   = ["accuracy"],
    )
    model.summary()
    
    history = model.fit(
        train_ds,
        validation_data = test_ds,
        epochs          = 100,
    )
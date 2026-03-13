# GOOGLE COLAB ADATTATO A LOCALE

# Load dataset
import os
import glob
import tensorflow as tf

dataset_path = "./dataset"

train_dir = os.path.join(dataset_path, "training")
test_dir = os.path.join(dataset_path, "testing")

def load_dataset(folder):
    file_paths = []
    labels = []

    wav_files = glob.glob(os.path.join(folder, "*.wav"))

    for file in wav_files:
        filename = os.path.basename(file)

        # label = parte prima del primo punto
        label = filename.split(".")[0]

        file_paths.append(file)
        labels.append(label)

    return file_paths, labels


X_train_paths, y_train = load_dataset(train_dir)
X_test_paths, y_test = load_dataset(test_dir)

print("Training samples:", len(X_train_paths))
print("Testing samples:", len(X_test_paths))

print("Esempio:")
print(X_train_paths[0], "->", y_train[0])

lookup = tf.keras.layers.StringLookup(output_mode="one_hot", num_oov_indices=0)
lookup.adapt(y_train)

y_train = lookup(y_train)
y_test = lookup(y_test)

labels = lookup.get_vocabulary()
num_classes = len(labels)

print("Classi:", labels)
print(X_train_paths[0], "->", y_train[0])

# Plot segnale audio
import matplotlib.pyplot as plt
import numpy as np
import librosa

audio, sr = librosa.load(X_train_paths[0], sr=16000)

max_val = np.max(np.abs(audio))

plt.figure(figsize=(10,6))
plt.plot(audio)
plt.title("Audio Signal", fontsize=20, fontweight='bold')
plt.ylim(-max_val, max_val)

plt.xlabel("Samples")
plt.ylabel("Amplitude")

plt.show()

# MFCC
import librosa
import numpy as np
from scipy.fftpack import dct

# parametri Edge Impulse
N_MFCC = 13
FRAME_LENGTH = 0.025
FRAME_STRIDE = 0.02
N_FFT = 512
N_MELS = 32
FMIN = 80
PRE_EMPHASIS = 0.98


def extract_mfcc(file_path):

    # carica audio (1 secondo, 16kHz)
    audio, sr = librosa.load(file_path, sr=None)

    # --- PRE-EMPHASIS ---
    emphasized = np.append(audio[0], audio[1:] - PRE_EMPHASIS * audio[:-1])

    # --- FRAMING ---
    frame_length = int(FRAME_LENGTH * sr)
    frame_step = int(FRAME_STRIDE * sr)

    signal_length = len(emphasized)

    num_frames = int(
        np.ceil(float(np.abs(signal_length - frame_length)) / frame_step)
    ) + 1

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized, z)

    indices = (
        np.tile(np.arange(0, frame_length), (num_frames, 1))
        + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    )

    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # finestra Hamming
    frames *= np.hamming(frame_length)

    # --- FFT ---
    mag_frames = np.absolute(np.fft.rfft(frames, N_FFT))

    # --- POWER SPECTRUM ---
    pow_frames = (1.0 / N_FFT) * (mag_frames ** 2)

    # --- MEL FILTERBANK ---
    mel_filter = librosa.filters.mel(
        sr=sr,
        n_fft=N_FFT,
        n_mels=N_MELS,
        fmin=FMIN
    )

    mel_energy = np.dot(pow_frames, mel_filter.T)

    # evita log(0)
    mel_energy = np.where(mel_energy == 0, np.finfo(float).eps, mel_energy)

    # --- LOG ---
    log_mel = np.log(mel_energy)

    # --- DCT ---
    mfcc = dct(log_mel, type=2, axis=1, norm='ortho')[:, :N_MFCC]

    return mfcc.astype(np.float32)  # shape (50,13)


X_train = []
X_test = []

for file in X_train_paths:
    mfcc = extract_mfcc(file)
    X_train.append(mfcc)

for file in X_test_paths:
    mfcc = extract_mfcc(file)
    X_test.append(mfcc)

X_train = np.array(X_train)
X_test = np.array(X_test)

print("Shape X_train:", X_train.shape)
print("Shape X_test:", X_test.shape)

# Save feature
np.savez("X_train.npz", X_train=X_train, X_test=X_test)
# Load feature
data = np.load("X_train.npz")

X_train = data["X_train"]
X_test = data["X_test"]

print(X_train.shape)
print(X_test.shape)

# Plot MFCC
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.imshow(X_train[0].T, cmap='viridis', origin='lower')
plt.title("Feature MFCC")
plt.xlabel("Frame Index")
plt.ylabel("Cepstral Coefficient Index")
plt.show()

# NORMALIZZAZIONE
mean = np.mean(X_train, axis=(0,1))
std = np.std(X_train, axis=(0,1)) + 1e-8

X_train_norm = (X_train - mean) / std
X_test_norm = (X_test - mean) / std

# Plot MFCC
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.imshow(X_train_norm[0].T, cmap='viridis', origin='lower')
plt.title("Feature MFCC")
plt.xlabel("Frame Index")
plt.ylabel("Cepstral Coefficient Index")
plt.show()

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, GaussianNoise, Reshape, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam

EPOCHS = 100
BATCH_SIZE = 32

# flatten
X_train_norm = X_train_norm.reshape(X_train_norm.shape[0], -1)
X_test_norm = X_test_norm.reshape(X_test_norm.shape[0], -1)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train_norm, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test_norm, y_test))

# dimensione del validation set
val_size = int(0.2 * len(X_train_norm))

# shuffle prima di dividere
train_dataset = train_dataset.shuffle(buffer_size=len(X_train_norm), seed=42)

val_dataset = train_dataset.take(val_size)
train_dataset = train_dataset.skip(val_size)

# batching
train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.AUTOTUNE)



#test 9
model = Sequential()
model.add(tf.keras.Input(shape=(650,)))
model.add(GaussianNoise(0.1))
model.add(Reshape((50, 13)))
model.add(Conv1D(4, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
model.add(Dropout(0.1))
model.add(Conv1D(8, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
model.add(Dropout(0.1))
model.add(GlobalAveragePooling1D())
model.add(Dense(num_classes, activation='softmax'))

model.summary()

learning_rate = 0.005
optimizer = Adam(learning_rate=learning_rate)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    verbose=2
)

os.makedirs("models", exist_ok=True)
model.save("models/model2.keras")

# Valutazione
test_loss, test_acc = model.evaluate(test_dataset)

print("Test accuracy:", test_acc)

# Convert to TFLite
def representative_dataset():
    for i in range(200):
        data = X_train[i].astype(np.float32)
        data = np.expand_dims(data, axis=0)
        yield [data]

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.representative_dataset = representative_dataset

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_quant_model = converter.convert()

save_path = "models/model1.tflite"

with open(save_path, "wb") as f:
    f.write(tflite_quant_model)

print("Modello salvato in:", save_path)

# Controllo dimensione del modello quantizzato
import os

size_kb = os.path.getsize(save_path) / 1024
print("Dimensione modello:", round(size_kb,2), "KB")
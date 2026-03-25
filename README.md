# KeywordSpotting
This project implements a Keyword Spotting (KWS) pipeline inspired by the approach used in Edge Impulse, but implemented entirely in TensorFlow.
The final goal is to deploy the trained model on an embedded device such as the Arduino Nano 33 BLE Sense Rev2.

The pipeline includes:
1. Audio data collection
2. Feature extraction using MFCC
3. Neural network training for keyword classification
4. Deployment on an embedded microcontroller

## Requirements
This project was developed using:
- Python 3.12.10
- TensorFlow (CPU version)

## Environment Setup
On Windows, create and activate a virtual environment with Python 3.12:
```bash
py -3.12 -m venv venv
```
Activate the virtual environment:
```bash
.\venv\Scripts\activate
```
Upgrade pip:
```bash
python -m pip install --upgrade pip
```
Install TensorFlow (CPU version):
```bash
pip install tensorflow
```
## Getting Started
Clone the repository:
```bash
git clone https://github.com/NicoNRG2/KeywordSpotting.git
cd KeywordSpotting
```
After setting up the environment, you can start working directly from the provided Jupyter notebook, which contains the main experimentation pipeline.

## Project Pipeline

The keyword spotting workflow follows these main steps:

### 1. Audio Data Collection
Audio samples are recorded as **WAV files** with the following specifications:

- Duration: **1 second**
- Sampling rate: **16 kHz**

These samples represent the spoken keywords used for training.

### 2. Feature Extraction
The audio signals are processed using **Mel-Frequency Cepstral Coefficients (MFCC)** to extract meaningful features from the waveform.

### 3. Model Training
Different **neural network architectures** are trained to classify the extracted features and recognize the target keywords.

### 4. Embedded Deployment
The final trained model is optimized for deployment on an embedded device such as the Arduino Nano 33 BLE Sense Rev2.

# Results
test CNN (3 conv layer, 8-16-32) partenza
Total params: 3,413 (13.33 KB)
Test accuracy: 0.8829953074455261

test CNN (3 conv layer, 4-8-16) ridotto i filtri
Total params: 1,229 (4.80 KB)
Test accuracy: 0.8346334099769592

test CNN (2 conv layer, 8-16) tolto un layer conv
Total params: 1,765 (6.89 KB)
Test accuracy: 0.8642745614051819

test CNN (2 conv layer, 8-16) tolto un layer conv, kernel 5, con GlobalAveragePooling1D
Total params: 1,269 (4.96 KB)
Test accuracy: 0.8845553994178772

test CNN (2 conv layer, 8-16) tolto un layer conv, kernel 2, con GlobalAveragePooling1D
Total params: 573 (2.24 KB)
Test accuracy: 0.798751950263977

test CNN (2 conv layer, 8-16) tolto un layer conv dropout 0.1
Total params: 1,765 (6.89 KB)
Test accuracy: 0.8580343127250671

test CNN (2 conv layer, 8-16) tolto un layer conv dropout 0.1, con GlobalAveragePooling1D
Total params: 805 (3.14 KB)
Test accuracy dropout 0.25: 0.8268330693244934
Test accuracy dropout 0.1: 0.8471139073371887
Test accuracy no dropout: 0.8393135666847229

test CNN (2 conv layer, 4-8) tolto un layer conv e ridotto filtri
Total params: 789 (3.08 KB)
Test accuracy: 0.8205928206443787

test CNN (2 conv layer, 4-8) tolto un layer conv e ridotto filtri dropout 0.1
Total params: 789 (3.08 KB)
Test accuracy: 0.8377535343170166

test CNN (2 conv layer, 4-8) tolto un layer conv e ridotto filtri no dropout
Total params: 789 (3.08 KB)
Test accuracy: 0.815912663936615

test CNN (2 conv layer, 4-8) tolto un layer conv e ridotto filtri no dropout, con GlobalAveragePooling1D
Total params: 309 (1.21 KB)
Test accuracy: 0.7628704905509949

test on MLP:
model = keras.Sequential([
    keras.layers.Input(shape=(650,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])
Test accuracy: 0.8096724152565002
Quantization Aware test accuracy: 0.815912663936615
Post training quantization Accuracy: 0.8112324492979719
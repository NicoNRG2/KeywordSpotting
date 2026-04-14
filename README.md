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

## Report
Click the link to view the final [Report](Report_low_power.pdf)
///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////             1 prova - PYTHON              /////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////

def build_model(input_shape):
    inputs = keras.Input(shape=input_shape, name="mfcc_input")

    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv1D(256, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="output")(x)

    return keras.Model(inputs, outputs, name="KeywordSpotting_Conv1D")



[5/5] Valutazione e salvataggio in: ./saved_model
      Test Loss    : 0.2754
      Test Accuracy: 0.9073

--- Classification Report ---
              precision    recall  f1-score   support

     heynano       1.00      1.00      1.00        63
          on       0.76      0.95      0.84       100
         off       0.86      0.95      0.90       109
_background_       0.98      0.85      0.91       278

    accuracy                           0.91       550
   macro avg       0.90      0.94      0.92       550
weighted avg       0.92      0.91      0.91       550



 Total params: 161,028 (629.02 KB)
 Trainable params: 160,132 (625.52 KB)
 Non-trainable params: 896 (3.50 KB)


///////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////       2 prova - EDGE IMPULSE SU ARDUINO         //////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////


![Testo alternativo](/evaluation/imm.png "Titolo opzionale")
Predictions (DSP: 97 ms., Classification: 9 ms., Anomaly: 0 ms.): 



///////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////       3 prova - con o senza augmentation da edge impulse       /////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////

EDGE IMPULSE
tranining senza augmentation --> accuracy  85.5     loss 0.40
testing senza augmentation ----> accuracy  79.32
tranining con augmentation ----> accuracy  86.1     loss  0.45
testing con augmentation ------> accuracy  77.46


PYTHON
test con modello con augmentation 0.45 ---> 83.73
test con modello con augmentation 0.2 ---> 85.76
test con modello con augmentation 0.1 ---> 85.42
test con modello senza augmentation ---> 84.9



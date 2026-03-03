"""
Simulatore Keyword Spotting - Arduino Nano BLE Sense Rev2
=========================================================
Simula il comportamento firmware Arduino con wake word detection.

Stato macchina:
  IDLE       -> ascolta solo "heynano"
  LISTENING  -> ascolta "on" / "off" per 5 secondi, poi torna IDLE

LED simulato nel terminale con colori ANSI.

Dipendenze aggiuntive:
  pip install sounddevice

Esecuzione:
  python simulator.py
"""

import os
import time
import queue
import threading
import numpy as np
import sounddevice as sd
import librosa
from tensorflow import keras

# ============================================================
#  CONFIGURAZIONE - deve corrispondere al training
# ============================================================
MODEL_PATH    = "./saved_model/keyword_spotting_model.keras"
SAMPLE_RATE   = 16000
DURATION      = 1.0          # secondi per inferenza
N_MFCC        = 40
HOP_LENGTH    = 512
N_FFT         = 2048

CONFIDENCE_THRESHOLD = 0.70  # soglia minima per accettare una predizione
LISTENING_TIMEOUT    = 5.0   # secondi di attesa dopo heynano

CLASS_NAMES = ["heynano", "on", "off", "_background_"]

# Sliding window: campiona ogni STEP_DURATION secondi
STEP_DURATION = 0.5          # overlap 50% -> piu reattivo

# ============================================================
#  COLORI ANSI PER TERMINALE
# ============================================================
RESET  = "[0m"
BOLD   = "[1m"
RED    = "[91m"
GREEN  = "[92m"
YELLOW = "[93m"
CYAN   = "[96m"
GRAY   = "[90m"
WHITE  = "[97m"

# ============================================================
#  STATO MACCHINA
# ============================================================
class State:
    IDLE      = "IDLE"
    LISTENING = "LISTENING"

# ============================================================
#  LED SIMULATO
# ============================================================
class SimulatedLED:
    def __init__(self):
        self.on = False

    def turn_on(self):
        self.on = True
        print(f"{BOLD}{YELLOW}  ╔══════════════╗{RESET}")
        print(f"{BOLD}{YELLOW}  ║  💡 LED ON   ║{RESET}")
        print(f"{BOLD}{YELLOW}  ╚══════════════╝{RESET}")

    def turn_off(self):
        self.on = False
        print(f"{BOLD}{GRAY}  ╔══════════════╗{RESET}")
        print(f"{BOLD}{GRAY}  ║  🌑 LED OFF  ║{RESET}")
        print(f"{BOLD}{GRAY}  ╚══════════════╝{RESET}")

    def status(self):
        if self.on:
            return f"{YELLOW}💡 ON {RESET}"
        else:
            return f"{GRAY}🌑 OFF{RESET}"

# ============================================================
#  FEATURE EXTRACTION
# ============================================================
def extract_mfcc(audio: np.ndarray) -> np.ndarray:
    target_len = int(SAMPLE_RATE * DURATION)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]
    audio = audio.astype(np.float32)

    mfcc = librosa.feature.mfcc(
        y=audio, sr=SAMPLE_RATE,
        n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=N_FFT,
    )
    mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-8)
    return mfcc.T[np.newaxis, ...]  # (1, time_frames, N_MFCC)

# ============================================================
#  MAIN SIMULATOR
# ============================================================
class KeywordSpottingSimulator:
    def __init__(self):
        print(f"{CYAN}Caricamento modello...{RESET}")
        self.model      = keras.models.load_model(MODEL_PATH)
        self.led        = SimulatedLED()
        self.state      = State.IDLE
        self.audio_buf  = queue.Queue()
        self.ring_buf   = np.zeros(int(SAMPLE_RATE * DURATION), dtype=np.float32)
        self.listen_timer = None
        print(f"{GREEN}Modello caricato!{RESET}")

    def _predict(self, audio: np.ndarray):
        """Esegue inferenza e restituisce (label, confidence) o None se sotto soglia."""
        mfcc  = extract_mfcc(audio)
        probs = self.model.predict(mfcc, verbose=0)[0]
        idx   = np.argmax(probs)
        conf  = float(probs[idx])
        label = CLASS_NAMES[idx]
        if conf < CONFIDENCE_THRESHOLD:
            return "_background_", conf
        return label, conf

    def _go_idle(self):
        """Torna allo stato IDLE."""
        self.state = State.IDLE
        print(f"{GRAY}[{self._ts()}] Timeout — torno in IDLE. In ascolto per 'heynano'...{RESET}")

    def _start_listening_timer(self):
        """Avvia timer di timeout per lo stato LISTENING."""
        if self.listen_timer and self.listen_timer.is_alive():
            self.listen_timer.cancel()
        self.listen_timer = threading.Timer(LISTENING_TIMEOUT, self._go_idle)
        self.listen_timer.daemon = True
        self.listen_timer.start()

    def _ts(self):
        return time.strftime("%H:%M:%S")

    def _process(self, audio: np.ndarray):
        """Processa un frame audio e aggiorna lo stato."""
        label, conf = self._predict(audio)

        # Stampa solo predizioni non-background con confidence significativa
        if label != "_background_" and conf > 0.55:
            bar = "█" * int(conf * 20)
            print(f"{GRAY}[{self._ts()}]{RESET} {WHITE}{label:12s}{RESET} {CYAN}{bar:<20s}{RESET} {conf:.0%}  stato={self.state}  LED={self.led.status()}")

        # --- Macchina a stati ---
        if self.state == State.IDLE:
            if label == "heynano" and conf >= CONFIDENCE_THRESHOLD:
                self.state = State.LISTENING
                self._start_listening_timer()
                print(f"{GREEN}{BOLD}  >>> Hey Nano rilevato! In ascolto per 'on' / 'off'... ({LISTENING_TIMEOUT}s) <<<{RESET}")

        elif self.state == State.LISTENING:
            if label == "on" and conf >= CONFIDENCE_THRESHOLD:
                self.led.turn_on()
                self._start_listening_timer()   # reset timer dopo comando
            elif label == "off" and conf >= CONFIDENCE_THRESHOLD:
                self.led.turn_off()
                self._start_listening_timer()   # reset timer dopo comando
            elif label == "heynano" and conf >= CONFIDENCE_THRESHOLD:
                # heynano ripetuto: reset timer senza azione
                self._start_listening_timer()
                print(f"{CYAN}[{self._ts()}] heynano ripetuto — timer resettato{RESET}")

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback sounddevice: riceve chunk audio dal microfono."""
        if status:
            print(f"{RED}Audio status: {status}{RESET}")
        self.audio_buf.put(indata[:, 0].copy())

    def run(self):
        step_samples = int(SAMPLE_RATE * STEP_DURATION)
        chunk_samples = int(SAMPLE_RATE * DURATION)

        print(f"{BOLD}{'='*55}{RESET}")
        print(f"{BOLD}  Simulatore Arduino Nano BLE Sense - Keyword Spotting{RESET}")
        print(f"{BOLD}{'='*55}{RESET}")
        print(f"  Modello    : {MODEL_PATH}")
        print(f"  Soglia     : {CONFIDENCE_THRESHOLD:.0%}")
        print(f"  Timeout    : {LISTENING_TIMEOUT}s")
        print(f"  Step       : {STEP_DURATION}s (sliding window)")
        print(f"{BOLD}{'='*55}{RESET}")
        print(f"{GREEN}Stato iniziale: IDLE — di' 'Hey Nano'!{RESET}")

        accumulated = np.array([], dtype=np.float32)

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=step_samples,
            callback=self._audio_callback,
        ):
            print(f"{GRAY}Microfono attivo. Premi Ctrl+C per uscire.{RESET}")
            try:
                while True:
                    chunk = self.audio_buf.get(timeout=2.0)
                    accumulated = np.concatenate([accumulated, chunk])

                    # Quando abbiamo abbastanza audio, processa e fai slide
                    while len(accumulated) >= chunk_samples:
                        frame = accumulated[:chunk_samples]
                        self._process(frame)
                        accumulated = accumulated[step_samples:]  # slide di STEP_DURATION

            except KeyboardInterrupt:
                print(f"{CYAN}Simulatore fermato.{RESET}")
                if self.listen_timer:
                    self.listen_timer.cancel()

# ============================================================
#  ENTRY POINT
# ============================================================
if __name__ == "__main__":
    sim = KeywordSpottingSimulator()
    sim.run()
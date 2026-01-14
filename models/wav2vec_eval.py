import os
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.transforms as T
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from jiwer import wer

# -----------------------------
# Configuration
# -----------------------------
MODEL_NAME = "facebook/wav2vec2-base-960h"
WAV_FOLDER = "data/audio"
TRANSCRIPTIONS_FILE = "data/transcripts.csv"

# -----------------------------
# Load model
# -----------------------------
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -----------------------------
# Transcription function
# -----------------------------
def transcribe(wav_path):
    # Load audio using soundfile
    waveform, sample_rate = sf.read(wav_path)
    
    # Convert to float32 if necessary
    if waveform.dtype != np.float32:
        waveform = waveform.astype(np.float32)
    
    # If stereo, convert to mono
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)
    
    # Resample to 16kHz if needed
    if sample_rate != 16000:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform_tensor = torch.tensor(waveform).unsqueeze(0)
        waveform = resampler(waveform_tensor).squeeze().numpy()
    
    # Prepare input for Wav2Vec2
    input_values = processor(waveform, sampling_rate=16000, return_tensors="pt").input_values.to(device)
    
    # Run inference
    with torch.no_grad():
        logits = model(input_values).logits
    
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription.lower()

# -----------------------------
# Load ground truth from CSV
# -----------------------------
df = pd.read_csv(TRANSCRIPTIONS_FILE)
ground_truths = dict(zip(df['file'], df['reference'].str.lower()))

# -----------------------------
# Evaluate
# -----------------------------
wer_scores = {}

print("Starting transcription and evaluation...\n")

for file in os.listdir(WAV_FOLDER):
    if file.endswith(".wav"):
        path = os.path.join(WAV_FOLDER, file)
        predicted = transcribe(path)
        true_text = ground_truths.get(file, "")
        error = wer(true_text, predicted)
        wer_scores[file] = error
        print(f"{file}: WER = {error:.2%}")
        print(f"  True: {true_text}")
        print(f"  Pred: {predicted}\n")

average_wer = sum(wer_scores.values()) / len(wer_scores)
print(f"Average WER: {average_wer:.2%}")


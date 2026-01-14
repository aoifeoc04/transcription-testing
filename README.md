1. What wav2vec 2.0 Is (Quick Context)
- A pretrained self-supervised ASR model
- Trained on raw audio (no handcrafted features)
- Available via Hugging Face
- Commonly used in research benchmarks

2. Packages
torch
- Core deep learning library
- Runs the neural networks
- Required by wav2vec

tourchaudio
- Audio utilities built for PyTorch
- Helps load and process audio

transformers
- Hugging Face library
- Provides wav2vec 2.0 model + tokenizer
- Saves from writing ML code from scratch

jiwer
- Calculates Word Error Rate (WER)
- Critical for evaluation 

librosa
- Audio loading and resampling
- Ensure 16kHZ mono audio

soundfile
- Backend library for reading WAV files
- Used by librosa internally 



3. Installation 
uv add torch torchaudio transformers jiwer librosa soundfile
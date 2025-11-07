import librosa
import soundfile as sf

# Parameters — set as needed for your data!
sr = 22050  # Sampling rate used for your melspecs and expected output audio
n_fft = 1024
hop_length = 256
win_length = 1024
n_mels = 80  # Number of mel bins in your spectrogram

# If output is (T, 80), transpose to (80, T) for librosa
if out.shape[1] == n_mels:
    out_mel = out.T  # (n_mels, T)
else:
    out_mel = out

# Inverse mel to waveform
wav = librosa.feature.inverse.mel_to_audio(
    out_mel,
    sr=sr,
    n_fft=n_fft,
    hop_length=hop_length,
    win_length=win_length,
    power=1.0,  # Or 2.0 depending on your mel-spectrogram computation
    n_iter=60   # Number of Griffin-Lim iterations, optional
)

# Save waveform as .wav
os.makedirs(output_dir, exist_ok=True)
wav_path = os.path.join(output_dir, "converted.wav")
sf.write(wav_path, wav, sr)
print(f"Waveform saved to {wav_path}")

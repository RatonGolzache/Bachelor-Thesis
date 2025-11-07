import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

filename = "c:/Users/alexn/Desktop/0011_000367_gen.wav"   # Input WAV file path
window_length = 2048     # Window size in samples
hop_length = 512        # Hop size in samples

# Load audio
y, sr = librosa.load(filename, sr=None)

# Compute STFT
D = librosa.stft(y, n_fft=window_length, hop_length=hop_length, window='hann')

# Convert amplitude spectrogram to dB
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# Plot spectrogram
plt.figure(figsize=(12, 6))
librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', cmap='magma')
plt.colorbar(format='%+2.0f dB')
plt.title('STFT Magnitude Spectrogram')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.tight_layout()
plt.show()

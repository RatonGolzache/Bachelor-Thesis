import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

filename = "c:/Users/alexn/Desktop/0011_000367_gen.wav"    # Path to your audio file

# Load audio signal
y, sr = librosa.load(filename, sr=None)

# Mel parameters
n_mels = 80
hop_length = 512
n_fft = 2048

# Compute Mel spectrogram
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft, fmin=0, fmax=sr//2)
S_db = librosa.power_to_db(S, ref=np.max)

# Get mel filter centers in Hz
mel_centers = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=sr//2)

plt.figure(figsize=(10, 6))
librosa.display.specshow(S_db, x_axis='time', sr=sr, hop_length=hop_length, cmap='magma')

# Set custom y-ticks to mel filter center frequencies
num_ticks = 8  # Number of y-ticks to show for readability
idxs = np.linspace(0, n_mels-1, num=num_ticks, dtype=int)
plt.yticks(idxs, np.round(mel_centers[idxs]).astype(int))
plt.ylabel('Frequency (Hz)')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram (Mel filter centers in Hz)')
plt.tight_layout()
plt.show()

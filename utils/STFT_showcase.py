import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

filename = "c:/Users/alexn/Desktop/0011_000367_gen.wav"     # Input WAV file path
window_length = 2048       # Window size in samples
hop_length = 1024           # Hop size in samples
arbitrary_frame = 10       # Arbitrary frame index to visualize

y, sr = librosa.load(filename, sr=None)

# Full signal time axis
duration = len(y) / sr

# Extract window start samples
start_sample_1 = arbitrary_frame * hop_length
start_sample_2 = start_sample_1 + hop_length

# Extract windowed segments
window_1 = y[start_sample_1 : start_sample_1 + window_length]
window_2 = y[start_sample_2 : start_sample_2 + window_length]

# Zero-pad the windowed signals to full length to keep time alignment
def zero_pad_segment(segment, start_idx, total_len):
    padded = np.zeros(total_len)
    padded[start_idx:start_idx+len(segment)] = segment
    return padded

window_1_padded = zero_pad_segment(window_1, start_sample_1, len(y))
window_2_padded = zero_pad_segment(window_2, start_sample_2, len(y))

# Plot
fig, axs = plt.subplots(3, 1, figsize=(19, 18), sharex=True)

librosa.display.waveplot(y, sr=sr, ax=axs[0])
axs[0].set_title("Full Audio Signal", fontsize=26)

librosa.display.waveplot(window_1_padded, sr=sr, ax=axs[1])
axs[1].set_title(f"Window at frame {arbitrary_frame}", fontsize=26)

librosa.display.waveplot(window_2_padded, sr=sr, ax=axs[2])
axs[2].set_title(f"Window at frame {arbitrary_frame + 1}", fontsize=26)

# Only set x-axis label on the bottom plot
axs[2].set_xlabel("Time (s)", fontsize=22)

# y-axis label for all
for ax in axs:
    ax.set_ylabel("Amplitude", fontsize=18)
    ax.tick_params(axis='both', labelsize=16)

plt.tight_layout()
plt.subplots_adjust(top=0.94, bottom=0.10)
plt.show()
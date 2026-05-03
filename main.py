import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# -----------------------------
# 1. Load Audio
# -----------------------------
fs, signal = wavfile.read("audio.wav")

# Convert to mono if stereo
if len(signal.shape) == 2:
    signal = np.mean(signal, axis=1)

signal = signal.astype(float)

print("Sampling Frequency:", fs)
print("Full Signal Length:", len(signal))

# -----------------------------
# 2. Extract Clip
# -----------------------------
start_time = 2      # seconds (CHANGE THIS)
duration = 1        # seconds

start_sample = int(start_time * fs)
end_sample = int((start_time + duration) * fs)

clip = signal[start_sample:end_sample]

print("Clip Length:", len(clip))
print("Original Clip Position:", start_time, "seconds")

# -----------------------------
# 3. FFT Function
# -----------------------------
def compute_fft(sig):
    fft_vals = np.fft.fft(sig)
    mag = np.abs(fft_vals)
    return mag[:len(mag)//2]  # one-sided FFT

clip_fft = compute_fft(clip)

# -----------------------------
# 4. Sliding Window Matching
# -----------------------------
window_size = len(clip)
step = int(0.1 * fs)  # step size (adjust for speed/accuracy)

scores = []
positions = []

for i in range(0, len(signal) - window_size, step):
    segment = signal[i:i + window_size]
    seg_fft = compute_fft(segment)

    # Normalize and compute similarity
    numerator = np.dot(clip_fft, seg_fft)
    denominator = np.linalg.norm(clip_fft) * np.linalg.norm(seg_fft)

    similarity = numerator / denominator if denominator != 0 else 0

    scores.append(similarity)
    positions.append(i / fs)

# -----------------------------
# 5. Detect Best Match
# -----------------------------
best_index = np.argmax(scores)
detected_time = positions[best_index]

print("Detected Position:", detected_time, "seconds")
print("Best Similarity Score:", scores[best_index])

# -----------------------------
# 6. Plot Similarity
# -----------------------------
plt.figure()
plt.plot(positions, scores)
plt.axvline(start_time, color='g', linestyle='--', label="Actual")
plt.axvline(detected_time, color='r', linestyle='--', label="Detected")
plt.title("Similarity Score vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Similarity")
plt.legend()
plt.show()

# -----------------------------
# 7. Compare Signals
# -----------------------------
detected_sample = int(detected_time * fs)
detected_segment = signal[detected_sample:detected_sample + window_size]

plt.figure()
plt.subplot(2,1,1)
plt.plot(clip)
plt.title("Original Clip")

plt.subplot(2,1,2)
plt.plot(detected_segment)
plt.title("Detected Segment")

plt.tight_layout()
plt.show()
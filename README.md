# 🎵 Audio Matching System using FFT

A Python-based system that detects the position of a short audio clip within a longer signal using Fast Fourier Transform (FFT) and spectral similarity.

---

## 🚀 Features

- Convert audio from time domain to frequency domain using FFT
- Extract and analyze short audio clips
- Sliding window matching algorithm
- Compute similarity using normalized dot product
- Detect best match position
- Visualize signals and matching results

---

## ⚙️ How It Works

1. Load full audio signal
2. Extract a short query clip
3. Convert both signals to frequency domain using FFT
4. Slide a window over the full signal
5. Compute similarity between clip and each segment
6. Detect the highest similarity score

---

## 🧪 Example Output

The program prints:
- Sampling frequency
- Signal length
- Clip position
- Detected position
- Best similarity score

---

## 📊 Visualizations

- Full signal (time domain)
- Clip (time domain)
- Clip (frequency domain)
- Similarity score vs time
- Original clip vs detected segment

---

## 🛠️ Technologies

- Python
- NumPy
- SciPy
- Matplotlib

---

## ▶️ How to Run

```bash
python main.py

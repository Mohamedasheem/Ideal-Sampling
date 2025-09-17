# Ideal, Natural, & Flat-top -Sampling
# Aim
Write a simple Python program for the construction and reconstruction of ideal, natural, and flattop sampling.
# Tools required
# Program
```
# --- Impulse Sampling ---
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample

fs = 100          # Sampling frequency (Hz)
t = np.arange(0, 1, 1/fs)
f = 5             # Signal frequency (Hz)
signal = np.sin(2 * np.pi * f * t)

# Continuous signal
plt.figure(figsize=(10, 4))
plt.plot(t, signal, label='Continuous Signal')
plt.title('Continuous Signal (fs = 100 Hz)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()

# Sampled signal
signal_sampled = np.sin(2 * np.pi * f * t)
plt.figure(figsize=(10, 4))
plt.stem(t, signal_sampled, linefmt='r-', markerfmt='ro', basefmt='r-', label='Sampled Signal (fs = 100 Hz)')
plt.title('Sampling of Continuous Signal (fs = 100 Hz)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()

# Reconstructed signal
reconstructed_signal = resample(signal_sampled, len(t))
plt.figure(figsize=(10, 4))
plt.plot(t, reconstructed_signal, 'r--', label='Reconstructed Signal (fs = 100 Hz)')
plt.title('Reconstruction of Sampled Signal (fs = 100 Hz)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()


# --- Natural Sampling ---
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

fs = 1000         # Sampling frequency
T = 1             # Duration (s)
t = np.arange(0, T, 1/fs)

# Message signal
fm = 5
message_signal = np.sin(2 * np.pi * fm * t)

# Pulse train
pulse_rate = 50
pulse_train = np.zeros_like(t)
pulse_width = int(fs / pulse_rate / 2)
for i in range(0, len(t), int(fs / pulse_rate)):
    pulse_train[i:i+pulse_width] = 1

# Natural sampling
nat_signal = message_signal * pulse_train

# Reconstruct (zero-order hold + optional lowpass)
sample_times = t[pulse_train == 1]
sampled_signal = nat_signal[pulse_train == 1]
reconstructed_signal = np.zeros_like(t)
for i, time in enumerate(sample_times):
    idx = np.argmin(np.abs(t - time))
    reconstructed_signal[idx:idx+pulse_width] = sampled_signal[i]

def lowpass_filter(signal, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, signal)

reconstructed_signal = lowpass_filter(reconstructed_signal, 10, fs)

# Plots
plt.figure(figsize=(14, 10))
plt.subplot(4, 1, 1)
plt.plot(t, message_signal, label='Original Message Signal')
plt.legend(); plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(t, pulse_train, label='Pulse Train')
plt.legend(); plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(t, nat_signal, label='Natural Sampling')
plt.legend(); plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(t, reconstructed_signal, color='green', label='Reconstructed Message Signal')
plt.legend(); plt.grid(True)

plt.tight_layout()
plt.show()


# --- Flat-Top Sampling ---
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

fs = 1000
T = 1
t = np.arange(0, T, 1/fs)

# Message signal
fm = 5
message_signal = np.sin(2 * np.pi * fm * t)

# Pulse train for sampling
pulse_rate = 50
pulse_train_indices = np.arange(0, len(t), int(fs / pulse_rate))
pulse_train = np.zeros_like(t)
pulse_train[pulse_train_indices] = 1

# Flat-top sampled signal
flat_top_signal = np.zeros_like(t)
pulse_width_samples = int(fs / (2 * pulse_rate))
for idx in pulse_train_indices:
    sample_value = message_signal[idx]
    end_index = min(idx + pulse_width_samples, len(t))
    flat_top_signal[idx:end_index] = sample_value

# Low-pass filter for reconstruction
def lowpass_filter(signal, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, signal)

cutoff_freq = 2 * fm
reconstructed_signal = lowpass_filter(flat_top_signal, cutoff_freq, fs)

# Plots
plt.figure(figsize=(14, 10))

plt.subplot(4, 1, 1)
plt.plot(t, message_signal, label='Original Message Signal')
plt.title('Original Message Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend(); plt.grid(True)

plt.subplot(4, 1, 2)
plt.stem(t[pulse_train_indices], pulse_train[pulse_train_indices], basefmt=" ", label='Ideal Sampling Instances')
plt.title('Ideal Sampling Instances')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend(); plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(t, flat_top_signal, label='Flat-Top Sampled Signal')
plt.title('Flat-Top Sampled Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend(); plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(t, reconstructed_signal, color='green',
         label=f'Reconstructed Signal (Cutoff={cutoff_freq} Hz)')
plt.title('Reconstructed Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend(); plt.grid(True)

plt.tight_layout()
plt.show()
```
# Output Waveform
```
<img width="866" height="393" alt="1 1" src="https://github.com/user-attachments/assets/d6fcd305-ccb4-440b-a473-418344bd39b1" />
<img width="1390" height="989" alt="2 1" src="https://github.com/user-attachments/assets/8a6c8b51-66e0-4a81-bdb4-40748e425a75" />
<img width="1398" height="990" alt="3 1" src="https://github.com/user-attachments/assets/ec1ddf84-af27-468f-8523-45dabdde8e62" />



```
# Results
```
Attach the output waveform
```
# Hardware experiment output waveform.

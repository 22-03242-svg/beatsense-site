# beatsense_analysis.py
import wfdb
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import os

# --- Bandpass Filter ---
def bandpass_filter(signal, fs, lowcut=0.5, highcut=40):
    nyquist = 0.5 * fs
    b, a = butter(2, [lowcut / nyquist, highcut / nyquist], btype='band')
    return filtfilt(b, a, signal)

# --- R-peak Detection ---
def detect_r_peaks(signal, fs):
    filtered = bandpass_filter(signal, fs)
    peaks, _ = find_peaks(filtered, distance=fs*0.6, height=np.mean(filtered)*0.8)
    return peaks

# --- Classification ---
def classify_heart_rate(hr):
    if hr < 60:
        return "Bradycardia"
    elif hr > 100:
        return "Tachycardia"
    else:
        return "Normal"

# --- Main Analysis Function ---
def analyze_ecg(record_id, data_path):
    record_path = os.path.join(data_path, record_id)
    record = wfdb.rdrecord(record_path)
    fs = record.fs
    signal = record.p_signal[:, 0]

    peaks = detect_r_peaks(signal, fs)
    if len(peaks) < 2:
        raise ValueError("Not enough peaks detected.")

    rr_intervals = np.diff(peaks) / fs
    bpm = 60 / np.mean(rr_intervals)
    classification = classify_heart_rate(bpm)

    result = {
        "record": record_id,
        "heart_rate": round(float(bpm), 2),
        "classification": classification,
        "num_peaks": int(len(peaks))
    }
    return result

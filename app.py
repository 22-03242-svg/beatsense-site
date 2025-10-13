from flask import Flask, request, jsonify
import wfdb
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

def preprocess_signal(signal, fs):
    nyquist = 0.5 * fs
    low = 0.5 / nyquist
    high = 40.0 / nyquist
    b, a = butter(1, [low, high], btype='band')
    return filtfilt(b, a, signal)

def detect_r_peaks(signal, fs):
    threshold = 0.6 * np.max(signal)
    r_peaks = np.where(signal > threshold)[0]
    if len(r_peaks) > 0:
        r_peaks_diff = np.diff(r_peaks)
        r_peaks = r_peaks[np.concatenate(([True], r_peaks_diff > fs / 10))]
    return r_peaks

def calculate_heart_rate(r_peaks, fs):
    if len(r_peaks) < 2:
        return 0
    rr_intervals = np.diff(r_peaks) / fs
    mean_rr = np.mean(rr_intervals)
    return 60 / mean_rr

def classify_arrhythmia(heart_rate):
    if heart_rate == 0:
        return "Undetermined"
    elif heart_rate < 60:
        return "Bradycardia"
    elif heart_rate > 100:
        return "Tachycardia"
    else:
        return "Normal Sinus Rhythm"

@app.route('/analyze', methods=['POST'])
def analyze_signal():
    file = request.files['file']
    filename = file.filename
    filepath = os.path.join('uploads', filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(filepath)

    # Load ECG data (assuming PhysioNet .dat or .csv format)
    try:
        record_name = os.path.splitext(filename)[0]
        record = wfdb.rdrecord(record_name, pn_dir='uploads')
        signal = record.p_signal[:, 0]
        fs = record.fs
    except Exception as e:
        return jsonify({'error': f'Failed to load ECG file: {e}'})

    filtered_signal = preprocess_signal(signal, fs)
    r_peaks = detect_r_peaks(filtered_signal, fs)
    heart_rate = calculate_heart_rate(r_peaks, fs)
    arrhythmia_type = classify_arrhythmia(heart_rate)

    return jsonify({
        'heart_rate': round(heart_rate, 2),
        'classification': arrhythmia_type
    })

if __name__ == '__main__':
    app.run(debug=True)

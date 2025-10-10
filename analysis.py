import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# --- We keep your original helper functions ---

def preprocess_signal(signal, fs):
    """Preprocesses the ECG signal with a bandpass filter."""
    nyquist = 0.5 * fs
    low = 0.5 / nyquist
    high = 40.0 / nyquist
    b, a = butter(1, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def detect_r_peaks(signal, fs):
    """Detects R-peaks in the ECG signal using a simple threshold."""
    threshold = 0.6 * np.max(signal)
    r_peaks = np.where(signal > threshold)[0]
    if r_peaks.size > 0:
        r_peaks_diff = np.diff(r_peaks)
        r_peaks = r_peaks[np.concatenate(([True], r_peaks_diff > fs / 10))]
    return r_peaks

def calculate_heart_rate(r_peaks, fs):
    """Calculates the average heart rate from R-peaks."""
    if len(r_peaks) < 2:
        return 0
    rr_intervals = np.diff(r_peaks) / fs
    mean_rr = np.mean(rr_intervals)
    heart_rate = 60 / mean_rr
    return heart_rate

def classify_arrhythmia(heart_rate):
    """Classifies arrhythmia based on heart rate."""
    if heart_rate == 0:
        return "Undetermined"
    if heart_rate < 60:
        return "Bradycardia"
    elif heart_rate > 100:
        return "Tachycardia"
    else:
        return "Normal Sinus Rhythm"

# --- This is the new main function for the web app ---

def analyze_ecg_from_file(filepath, fs):
    """
    Main analysis pipeline for a single uploaded ECG file.

    Args:
        filepath (str): The path to the uploaded ECG data file.
        fs (int): The sampling frequency of the signal.

    Returns:
        dict: A dictionary containing the analysis results, or None if an error occurs.
    """
    try:
        # Load signal data from a text or CSV file, assuming a single column of values
        signal = np.loadtxt(filepath)
    except Exception as e:
        print(f"Error loading or processing file {filepath}: {e}")
        return None

    # 1. Preprocessing
    filtered_signal = preprocess_signal(signal, fs)

    # 2. Feature Extraction (R-peak detection)
    r_peaks = detect_r_peaks(filtered_signal, fs)

    # 3. Heart Rate Calculation
    heart_rate = calculate_heart_rate(r_peaks, fs)

    # 4. Classification
    arrhythmia_type = classify_arrhythmia(heart_rate)
    
    # 5. Visualization: Save the plot instead of showing it
    time = np.arange(len(filtered_signal)) / fs
    plt.figure(figsize=(15, 5))
    plt.plot(time, filtered_signal, label='Filtered ECG Signal')
    if len(r_peaks) > 0:
        plt.plot(r_peaks / fs, filtered_signal[r_peaks], 'ro', label='Detected R-peaks')
    
    title = f'ECG Analysis - Classification: {arrhythmia_type} ({heart_rate:.2f} BPM)'
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (mV)')
    plt.legend()
    plt.grid(True)
    
    # Define the path to save the plot image
    # The image will be saved in 'static/images/analysis_plot.png'
    os.makedirs('static/images', exist_ok=True)
    plot_path = 'static/images/analysis_plot.png'
    plt.savefig(plot_path)
    plt.close() # Important to close the plot to free up memory

    # Return all the results in a dictionary for the web page to use
    return {
        'classification': arrhythmia_type,
        'heart_rate': round(heart_rate, 2),
        'num_r_peaks': len(r_peaks),
        'plot_path': plot_path
    }

# BeatSense: ECG Signal Processing for Arrhythmia Detection
# This script provides a basic implementation of the concepts outlined in the research paper.
# It focuses on reading ECG data, preprocessing it, detecting R-peaks, and classifying arrhythmia.

# Import necessary libraries
import wfdb  # For reading ECG data from PhysioNet databases
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting the ECG signals
from scipy.signal import butter, filtfilt  # For filtering the signal

def load_ecg_data(record_name, pn_dir='mitdb'):
    """
    Load ECG data from the PhysioNet MIT-BIH Arrhythmia Database.

    Args:
        record_name (str): The name of the record to load (e.g., '100').
        pn_dir (str): The PhysioNet database directory (e.g., 'mitdb').

    Returns:
        tuple: A tuple containing the signal, fields, and annotation.
    """
    try:
        # Load a portion of the record to speed up the demo
        record = wfdb.rdrecord(record_name, pn_dir=pn_dir, sampto=15000)
        annotation = wfdb.rdann(record_name, 'atr', pn_dir=pn_dir, sampto=15000)
        # Use the first channel of the signal
        return record.p_signal[:, 0], record.fs, annotation
    except Exception as e:
        print(f"Error loading record {record_name}: {e}")
        return None, None, None

def preprocess_signal(signal, fs):
    """
    Preprocess the ECG signal by applying a bandpass filter to remove noise.

    Args:
        signal (np.array): The raw ECG signal.
        fs (int): The sampling frequency of the signal.

    Returns:
        np.array: The filtered ECG signal.
    """
    # Design a bandpass filter (0.5 - 40 Hz)
    nyquist = 0.5 * fs
    low = 0.5 / nyquist
    high = 40.0 / nyquist
    b, a = butter(1, [low, high], btype='band')

    # Apply the filter
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def detect_r_peaks(signal, fs):
    """
    Detect R-peaks in the ECG signal.
    This is a simplified R-peak detection algorithm using a threshold.

    Args:
        signal (np.array): The preprocessed ECG signal.
        fs (int): The sampling frequency.

    Returns:
        np.array: An array of indices where R-peaks are detected.
    """
    # Simple peak detection based on a dynamic threshold
    threshold = 0.6 * np.max(signal)
    r_peaks = np.where(signal > threshold)[0]

    # Remove consecutive peaks to get single R-peaks
    # (assumes a minimum distance between peaks)
    if r_peaks.size > 0:
        r_peaks_diff = np.diff(r_peaks)
        r_peaks = r_peaks[np.concatenate(([True], r_peaks_diff > fs / 10))] # Min distance ~0.1s
    return r_peaks

def calculate_heart_rate(r_peaks, fs):
    """
    Calculate the average heart rate from the detected R-peaks.

    Args:
        r_peaks (np.array): The indices of the R-peaks.
        fs (int): The sampling frequency.

    Returns:
        float: The calculated heart rate in beats per minute (BPM).
    """
    if len(r_peaks) < 2:
        return 0
    # Calculate RR intervals in seconds
    rr_intervals = np.diff(r_peaks) / fs
    mean_rr = np.mean(rr_intervals)
    heart_rate = 60 / mean_rr
    return heart_rate

def classify_arrhythmia(heart_rate):
    """
    Classify the arrhythmia based on the average heart rate.

    Args:
        heart_rate (float): The heart rate in BPM.

    Returns:
        str: The classification of the heart rhythm.
    """
    if heart_rate == 0:
        return "Undetermined"
    if heart_rate < 60:
        return "Bradycardia"
    elif heart_rate > 100:
        return "Tachycardia"
    else:
        return "Normal Sinus Rhythm"

def plot_ecg(signal, r_peaks, fs, title):
    """
    Plot the ECG signal with detected R-peaks.

    Args:
        signal (np.array): The ECG signal to plot.
        r_peaks (np.array): The indices of the detected R-peaks.
        fs (int): The sampling frequency.
        title (str): The title for the plot.
    """
    time = np.arange(len(signal)) / fs
    plt.figure(figsize=(15, 5))
    plt.plot(time, signal, label='Filtered ECG Signal')
    if len(r_peaks) > 0:
        plt.plot(r_peaks / fs, signal[r_peaks], 'ro', label='Detected R-peaks')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (mV)')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    """
    Main function to run the BeatSense demo on all records in the database.
    """
    print("Starting BeatSense Arrhythmia Detection Demo")

    # Get all record names from the MIT-BIH Arrhythmia Database
    try:
        record_names = wfdb.get_record_list('mitdb')
        print(f"Found {len(record_names)} records to analyze.")
    except Exception as e:
        print(f"Could not fetch record list from PhysioNet: {e}")
        # Fallback to a small list if the network request fails
        record_names = ['100', '101', '200', '207']

    for record_name in record_names:
        print(f"\n--- Processing Record: {record_name} ---")
        signal, fs, annotation = load_ecg_data(record_name)

        if signal is not None:
            # 1. Preprocessing
            filtered_signal = preprocess_signal(signal, fs)
            print("Signal preprocessed.")

            # 2. Feature Extraction (R-peak detection)
            r_peaks = detect_r_peaks(filtered_signal, fs)
            print(f"Detected {len(r_peaks)} R-peaks.")

            # 3. Heart Rate Calculation
            heart_rate = calculate_heart_rate(r_peaks, fs)
            print(f"Calculated Heart Rate: {heart_rate:.2f} BPM")

            # 4. Classification (based on average heart rate)
            arrhythmia_type = classify_arrhythmia(heart_rate)
            print(f"Classification: {arrhythmia_type}")

            # 5. Visualization
            plot_ecg(filtered_signal, r_peaks, fs,
                     f'ECG Signal for Record {record_name} - {arrhythmia_type} ({heart_rate:.2f} BPM)')

if __name__ == "__main__":
    main()

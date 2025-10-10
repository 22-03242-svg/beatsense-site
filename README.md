# 🫀 BeatSense: ECG Heartbeat Detection and Classification

BeatSense is a Jupyter Notebook project for analyzing ECG signals using the **MIT-BIH Arrhythmia Database**.  
It detects R-peaks, computes heart rate, classifies rhythms, and exports the results to CSV.

---

## 📂 Dataset

This project uses the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/).

📁 Place the dataset here:



---

## ⚙️ Features

✅ Bandpass filtering (0.5–40 Hz)  
✅ Automatic R-peak detection  
✅ Heart rate calculation  
✅ Rhythm classification (Normal / Bradycardia / Tachycardia)  
✅ CSV summary output  
✅ ECG signal visualization with detected peaks  

---

## 🚀 How to Run

1. Install required libraries  
   ```bash
   pip install wfdb numpy scipy pandas matplotlib tqdm

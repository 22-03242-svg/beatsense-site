from flask import Flask, render_template, request
import os
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# File upload and analysis route
@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['file']
    if not file:
        return render_template('index.html', result="No file uploaded.")

    try:
        # Try to read the uploaded file as CSV
        data = np.loadtxt(file, delimiter=',')
    except Exception as e:
        return render_template('index.html', result=f"Error reading file: {str(e)}")

    # --- Basic ECG analysis example ---
    mean_val = np.mean(data)
    std_val = np.std(data)
    max_val = np.max(data)
    min_val = np.min(data)

    result_text = (
        f"Mean amplitude: {mean_val:.3f}<br>"
        f"Standard deviation: {std_val:.3f}<br>"
        f"Max value: {max_val:.3f}<br>"
        f"Min value: {min_val:.3f}"
    )

    # --- Generate ECG plot ---
    plt.figure(figsize=(8, 3))
    plt.plot(data[:1000])  # Show first 1000 samples for readability
    plt.title("ECG Signal")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.tight_layout()

    # Convert plot to base64 for embedding in HTML
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = 'data:image/png;base64,' + base64.b64encode(img.getvalue()).decode()
    plt.close()

    return render_template('index.html', result=result_text, plot_url=plot_url)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

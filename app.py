from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import csv

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    data = np.loadtxt(file, delimiter=',')
    plt.figure()
    plt.plot(data)
    plt.title("Uploaded ECG Signal")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")

    # Convert plot to PNG image in base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return render_template('index.html', image=image_base64)

if __name__ == '__main__':
    app.run(debug=True)

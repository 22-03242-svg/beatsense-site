from flask import Flask, render_template, request, url_for
import os
from werkzeug.utils import secure_filename
from analysis import analyze_ecg_from_file # Import our new analysis function

# Initialize the Flask application
app = Flask(__name__)

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Create the uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define the route for the home page (your main HTML page)
@app.route('/')
def home():
    return render_template('index.html')

# Define the route that handles the file upload and analysis
@app.route('/analyze', methods=['POST'])
def analyze():
    if 'ecg_file' not in request.files:
        return "Error: No file part in the request.", 400
    
    file = request.files['ecg_file']

    if file.filename == '':
        return "Error: No file selected for uploading.", 400

    if file:
        # Secure the filename to prevent security issues
        filename = secure_filename(file.filename)
        # Save the uploaded file to the 'uploads' folder
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # --- THIS IS WHERE YOUR PYTHON CODE RUNS ---
        # Call the analysis function with the path to the uploaded file
        # We assume a default sampling frequency of 360 Hz for MIT-BIH data
        fs = 360 
        results = analyze_ecg_from_file(filepath, fs)

        # If analysis fails, show an error
        if results is None:
            return "Error: Could not process the uploaded file. Please ensure it is a valid single-column text or CSV file.", 500

        # Render the results page with the data from your analysis
        return render_template('results.html', results=results)

# This allows you to run the app by executing "python app.py"
if __name__ == '__main__':
    app.run(debug=True)

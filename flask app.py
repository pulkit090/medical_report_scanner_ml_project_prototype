from flask import Flask, render_template, request, redirect, url_for, jsonify
import json
from transformers import pipeline
import numpy as np
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load the pre-trained model for sentiment analysis
classifier = pipeline("sentiment-analysis")

# Function to analyze and assign severity
def analyze_report(report):
    analysis = classifier(report)[0]
    severity = 5 if analysis["label"] == "NEGATIVE" else 1
    return severity

# Function to process and prioritize reports
def process_reports():
    with open('extended_sample_data.json', 'r') as f:
        reports = json.load(f)

    for report in reports:
        report["severity"] = analyze_report(report["report"])

    scaler = MinMaxScaler()
    ages = np.array([report["age"] for report in reports]).reshape(-1, 1)
    scaled_ages = scaler.fit_transform(ages).flatten()

    for i, report in enumerate(reports):
        age_factor = scaled_ages[i] * 2  # Age has a weight of 2 in the final score
        critical_factor = 3 if report["critical"] else 0  # Critical condition adds 3 to the final score
        report["final_score"] = report["severity"] + age_factor + critical_factor

    sorted_reports = sorted(reports, key=lambda x: x["final_score"], reverse=True)
    return sorted_reports

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file:
        file.save('extended_sample_data.json')
    return redirect(url_for('result'))

@app.route('/result')
def result():
    sorted_reports = process_reports()
    return render_template('result.html', reports=sorted_reports)

if __name__ == '__main__':
    app.run(debug=True)

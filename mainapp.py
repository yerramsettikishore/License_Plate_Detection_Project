from flask import Flask, render_template, redirect
import subprocess
import time
import webbrowser

app = Flask(__name__)

# Define the ports for each project
FILE_DETECTION_PORT = 5001
LIVE_DETECTION_PORT = 5002

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/file_detection')
def run_file_detection():
    """Start File Detection project and open in browser"""
    subprocess.Popen(["python", "file_detection/app.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    time.sleep(3)  # Give time for the server to start
    webbrowser.open(f"http://127.0.0.1:{FILE_DETECTION_PORT}")

    return redirect(f"http://127.0.0.1:{FILE_DETECTION_PORT}")

@app.route('/live_detection')
def run_live_detection():
    """Start Live Detection project and open in browser"""
    subprocess.Popen(["python", "live_detection/app.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    time.sleep(3)  # Give time for the server to start
    webbrowser.open(f"http://127.0.0.1:{LIVE_DETECTION_PORT}")

    return redirect(f"http://127.0.0.1:{LIVE_DETECTION_PORT}")

if __name__ == '__main__':
    app.run(debug=True, port=5000)

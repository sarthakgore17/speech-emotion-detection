"""
Entry point matching README instructions: python serve.py
Runs the Flask application defined in main.py
"""
import os
import sys

# Ensure current directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import app, get_model

if __name__ == '__main__':
    try:
        get_model()
    except Exception as e:
        print(f"Warning: Model could not be preloaded: {e}")
    app.run(host='0.0.0.0', port=5000, debug=False)

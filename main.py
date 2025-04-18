from app import app  # noqa: F401

# The app is imported directly from app.py
# This file is used by gunicorn to run the application

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
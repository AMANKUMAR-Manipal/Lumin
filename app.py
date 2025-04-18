import os
import logging
import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
from werkzeug.utils import secure_filename
import yolo_detection as yolo

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default_secret_key")

# Configure upload settings
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize YOLOv3 model
yolo.initialize_yolo()

# Global variables for camera
camera = None
camera_active = False

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def gen_frames():
    """Generate camera frames with YOLO detection"""
    global camera, camera_active
    
    # Initialize camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera_active = True
    
    while camera_active:
        success, frame = camera.read()
        if not success:
            logger.error("Failed to capture frame from camera")
            break
        else:
            # Apply YOLO detection to the frame
            try:
                processed_frame = yolo.detect_objects(frame)
                ret, buffer = cv2.imencode('.jpg', processed_frame)
                if not ret:
                    continue
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e:
                logger.error(f"Error in frame processing: {str(e)}")
                continue

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route for camera feed"""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera')
def start_camera():
    """Start the camera stream"""
    global camera, camera_active
    if camera is None:
        camera = cv2.VideoCapture(0)
    camera_active = True
    return jsonify({"status": "success", "message": "Camera started"})

@app.route('/stop_camera')
def stop_camera():
    """Stop the camera stream"""
    global camera, camera_active
    camera_active = False
    if camera is not None:
        camera.release()
        camera = None
    return jsonify({"status": "success", "message": "Camera stopped"})

@app.route('/detect_image', methods=['POST'])
def detect_image():
    """Process uploaded image for object detection"""
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"})
    
    if file and allowed_file(file.filename):
        try:
            # Read and process the uploaded image
            in_memory_file = file.read()
            nparr = np.frombuffer(in_memory_file, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Apply YOLO detection
            processed_img = yolo.detect_objects(img)
            
            # Save processed image
            filename = secure_filename(file.filename)
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"detected_{filename}")
            cv2.imwrite(output_path, processed_img)
            
            # Return path to processed image
            return jsonify({
                "status": "success", 
                "image_path": f"/{output_path}",
                "message": "Image processed successfully"
            })
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return jsonify({"status": "error", "message": f"Error processing image: {str(e)}"})
    
    return jsonify({"status": "error", "message": "File type not allowed"})

@app.teardown_appcontext
def cleanup(exception=None):
    """Cleanup resources when app context ends"""
    global camera
    if camera is not None:
        camera.release()
        camera = None

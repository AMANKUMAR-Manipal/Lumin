import os
import json
import time
import uuid
import logging
from datetime import datetime
from flask import render_template, request, redirect, url_for, jsonify, flash
import numpy as np
import cv2
from werkzeug.utils import secure_filename

from app import app, db
from models import Detection, Image
from object_detection.cnn_detector import CNNDetector
from object_detection.knn_detector import KNNDetector
from object_detection.yolo_detector import YOLODetector
from object_detection.utils import draw_bounding_boxes

# Initialize detectors (lazy loading to avoid startup errors)
cnn_detector = None
knn_detector = None
yolo_detector = None

def get_cnn_detector():
    global cnn_detector
    if cnn_detector is None:
        cnn_detector = CNNDetector()
    return cnn_detector

def get_knn_detector():
    global knn_detector
    if knn_detector is None:
        knn_detector = KNNDetector()
    return knn_detector

def get_yolo_detector():
    global yolo_detector
    if yolo_detector is None:
        yolo_detector = YOLODetector()
    return yolo_detector

def allowed_file(filename):
    """Check if uploaded file has an allowed extension"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def landing():
    """Render the landing page"""
    return render_template('landing.html')

@app.route('/detect')
def index():
    """Render the object detection page"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Render the dashboard with detection statistics"""
    # Query database for detection statistics
    all_detections = Detection.query.all()
    all_images = Image.query.all()
    
    # Calculate basic stats
    stats = {
        'total_images': len(all_images),
        'total_objects': 0,
        'vehicle_count': 0,
        'person_count': 0,
        'person_confidence': 0,
        'class_counts': {},
        'algorithm_stats': {
            'cnn': {'count': 0, 'confidence': 0, 'time': 0},
            'knn': {'count': 0, 'confidence': 0, 'time': 0},
            'yolov3': {'count': 0, 'confidence': 0, 'time': 0}
        },
        'recent_detections': [],
        'recent_images': [],
        'processing_times': [],
        'time_labels': [],
        'distribution_labels': [],
        'distribution_data': []
    }
    
    # Process detections for statistics
    vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']
    object_class_counts = {}
    
    for detection in all_detections:
        objects = json.loads(detection.objects_detected)
        scores = json.loads(detection.confidence_scores)
        
        # Count total objects
        stats['total_objects'] += len(objects)
        
        # Count by algorithm
        algo = detection.algorithm
        if algo in stats['algorithm_stats']:
            stats['algorithm_stats'][algo]['count'] += len(objects)
            if scores:
                stats['algorithm_stats'][algo]['confidence'] += sum(scores) / len(scores)
            stats['algorithm_stats'][algo]['time'] += detection.detection_time
        
        # Count objects by class
        for obj in objects:
            if obj not in object_class_counts:
                object_class_counts[obj] = 0
            object_class_counts[obj] += 1
            
            # Count vehicles and pedestrians
            if obj in vehicle_classes:
                stats['vehicle_count'] += 1
            elif obj == 'person':
                stats['person_count'] += 1
                
                # Calculate average person confidence
                person_indices = [i for i, o in enumerate(objects) if o == 'person']
                if person_indices:
                    person_scores = [scores[i] for i in person_indices]
                    stats['person_confidence'] += sum(person_scores) / len(person_scores)
    
    # Calculate averages for algorithm stats
    for algo in stats['algorithm_stats']:
        count = stats['algorithm_stats'][algo]['count']
        if count > 0:
            stats['algorithm_stats'][algo]['confidence'] /= count
        
        # Get detection count
        algo_detections = [d for d in all_detections if d.algorithm == algo]
        if algo_detections:
            stats['algorithm_stats'][algo]['time'] /= len(algo_detections)
    
    # Normalize algorithm stats for radar chart
    max_count = max([stats['algorithm_stats'][a]['count'] for a in stats['algorithm_stats']]) or 1
    max_conf = max([stats['algorithm_stats'][a]['confidence'] for a in stats['algorithm_stats']]) or 1
    min_time = min([stats['algorithm_stats'][a]['time'] for a in stats['algorithm_stats']]) or 0.001
    max_time = max([stats['algorithm_stats'][a]['time'] for a in stats['algorithm_stats']]) or 0.001
    
    for algo in stats['algorithm_stats']:
        stats['algorithm_stats'][algo]['normalized_count'] = stats['algorithm_stats'][algo]['count'] / max_count
        stats['algorithm_stats'][algo]['normalized_confidence'] = stats['algorithm_stats'][algo]['confidence'] / max_conf
        # Invert time so faster is better (higher on radar chart)
        time_normalized = (stats['algorithm_stats'][algo]['time'] - min_time) / (max_time - min_time)
        stats['algorithm_stats'][algo]['normalized_speed'] = 1 - time_normalized
    
    # Finalize person confidence
    if stats['person_count'] > 0:
        stats['person_confidence'] /= stats['person_count']
    
    # Process class distribution for pie chart
    sorted_classes = sorted(object_class_counts.items(), key=lambda x: x[1], reverse=True)
    top_classes = sorted_classes[:6]  # Top 6 classes
    other_count = sum(count for cls, count in sorted_classes[6:]) if len(sorted_classes) > 6 else 0
    
    stats['distribution_labels'] = [cls for cls, _ in top_classes]
    stats['distribution_data'] = [count for _, count in top_classes]
    
    if other_count > 0:
        stats['distribution_labels'].append('Other')
        stats['distribution_data'].append(other_count)
    
    # Format for JSON
    stats['distribution_labels'] = json.dumps(stats['distribution_labels'])
    stats['distribution_data'] = json.dumps(stats['distribution_data'])
    
    # Get class counts for display
    stats['class_counts'] = {cls: count for cls, count in object_class_counts.items() if cls in vehicle_classes}
    
    # Get recent detections
    recent_detections = Detection.query.order_by(Detection.created_at.desc()).limit(5).all()
    for detection in recent_detections:
        objects = json.loads(detection.objects_detected)
        scores = json.loads(detection.confidence_scores)
        avg_confidence = sum(scores) / len(scores) if scores else 0
        
        stats['recent_detections'].append({
            'time': detection.created_at.strftime('%H:%M:%S'),
            'algorithm': detection.algorithm,
            'objects': objects,
            'confidence': avg_confidence
        })
    
    # Get processing times for line chart
    time_detections = Detection.query.order_by(Detection.created_at.desc()).limit(10).all()
    stats['processing_times'] = [detection.detection_time for detection in reversed(time_detections)]
    stats['time_labels'] = [f"#{i+1}" for i in range(len(stats['processing_times']))]
    
    # Convert to JSON for JavaScript
    stats['processing_times'] = json.dumps(stats['processing_times'])
    stats['time_labels'] = json.dumps(stats['time_labels'])
    
    # Get recent images with detections
    recent_images = []
    for detection in Detection.query.order_by(Detection.created_at.desc()).limit(6).all():
        # Extract filename from path
        path_parts = detection.image_path.split('/')
        if len(path_parts) > 1:
            filename = path_parts[-1]
            image_path = '/'.join(['uploads', filename])
            
            # Get detection results
            objects = json.loads(detection.objects_detected)
            
            recent_images.append({
                'path': image_path,
                'algorithm': detection.algorithm,
                'object_count': len(objects)
            })
    
    stats['recent_images'] = recent_images
    
    return render_template('dashboard.html', stats=stats)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle image upload and run object detection"""
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Generate a unique filename
        original_filename = secure_filename(file.filename)
        filename = f"{uuid.uuid4().hex}.{original_filename.rsplit('.', 1)[1].lower()}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Read image and get dimensions
        img = cv2.imread(file_path)
        height, width = img.shape[:2]
        
        # Store image information in database
        image = Image(
            filename=filename,
            original_filename=original_filename,
            file_path=file_path,
            file_size=os.path.getsize(file_path),
            width=width,
            height=height,
            mime_type=file.content_type
        )
        db.session.add(image)
        db.session.commit()
        
        # Get selected algorithms
        algorithms = request.form.getlist('algorithms')
        if not algorithms:  # If no algorithm selected, use all
            algorithms = ['cnn', 'knn', 'yolov3']
        
        # Process image with selected algorithms
        detection_results = {}
        
        for algo in algorithms:
            start_time = time.time()
            
            if algo == 'cnn':
                objects, scores, boxes = get_cnn_detector().detect(img)
            elif algo == 'knn':
                objects, scores, boxes = get_knn_detector().detect(img)
            elif algo == 'yolov3':
                objects, scores, boxes = get_yolo_detector().detect(img)
            
            end_time = time.time()
            detection_time = end_time - start_time
            
            # Save results to database
            detection = Detection(
                image_path=file_path,
                algorithm=algo,
                objects_detected=json.dumps(objects),
                detection_time=detection_time,
                confidence_scores=json.dumps(scores),
                bounding_boxes=json.dumps(boxes)
            )
            db.session.add(detection)
            
            # Create result image with bounding boxes
            output_filename = f"{uuid.uuid4().hex}_{algo}.jpg"
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            result_img = draw_bounding_boxes(img.copy(), boxes, objects, scores)
            cv2.imwrite(output_path, result_img)
            
            # Store results for display
            detection_results[algo] = {
                'objects': objects,
                'scores': scores,
                'boxes': boxes,
                'time': detection_time,
                'result_path': os.path.join('uploads', output_filename)
            }
        
        db.session.commit()
        
        return render_template('results.html', 
                              image_path=os.path.join('uploads', filename),
                              detection_results=detection_results,
                              width=width,
                              height=height)
        
    flash('Invalid file type. Please upload a JPG or PNG image.')
    return redirect(url_for('index'))

@app.route('/compare', methods=['POST'])
def compare_results():
    """Compare detection results from different algorithms"""
    data = request.json
    image_path = data.get('image_path')
    
    # Get detection results for this image
    detections = Detection.query.filter_by(image_path=image_path).all()
    
    comparison = {}
    for detection in detections:
        algo = detection.algorithm
        objects = json.loads(detection.objects_detected)
        scores = json.loads(detection.confidence_scores)
        time_taken = detection.detection_time
        
        # Count objects by class
        object_counts = {}
        for obj in objects:
            if obj in object_counts:
                object_counts[obj] += 1
            else:
                object_counts[obj] = 1
        
        # Calculate average confidence
        avg_confidence = sum(scores) / len(scores) if scores else 0
        
        comparison[algo] = {
            'object_counts': object_counts,
            'total_objects': len(objects),
            'avg_confidence': avg_confidence,
            'time_taken': time_taken
        }
    
    return jsonify(comparison)

@app.errorhandler(413)
def too_large(e):
    flash('File too large. Maximum size is 16MB.')
    return redirect(url_for('index')), 413

@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return render_template('index.html', error="An internal error occurred."), 500

import os
import logging
import cv2
import numpy as np
import random

# COCO class names for YOLOv3
with open('coco.names', 'r') as f:
    CLASSES = [line.strip() for line in f.readlines()]

# Colors for visualization
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(CLASSES), 3), dtype="uint8")

def initialize_yolo():
    """
    Initialize YOLOv3 model with pre-trained weights, configuration,
    and COCO class names
    """
    logging.info("Initializing YOLOv3 model")
    
    # Check if weights file exists (it's large, over 200MB)
    weights_path = os.path.join(os.getcwd(), 'yolov3.weights')
    config_path = os.path.join(os.getcwd(), 'yolov3.cfg')
    
    if not os.path.exists(weights_path) or not os.path.exists(config_path):
        logging.warning("YOLOv3 weights or config file not found. Using dummy detection.")
        return None
    
    try:
        # Load the model from weights and config
        net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        
        # Configure backend (CPU unless CUDA available)
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            logging.info("Using CUDA backend for YOLO")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            logging.info("Using CPU backend for YOLO")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
        # Get the output layer names
        layer_names = net.getLayerNames()
        unconnected_out_layers = net.getUnconnectedOutLayers()
        
        if isinstance(unconnected_out_layers[0], list):
            output_layers = [layer_names[i[0] - 1] for i in unconnected_out_layers]
        else:
            output_layers = [layer_names[i - 1] for i in unconnected_out_layers]
        
        return {
            'net': net,
            'output_layers': output_layers,
            'width': 416,
            'height': 416
        }
    except Exception as e:
        logging.error(f"Error initializing YOLOv3: {e}")
        return None

def dummy_detect(img):
    """
    A dummy detection function that draws example boxes
    to demonstrate how the real detection would work
    """
    logging.info("Using dummy detection for demonstration")
    height, width = img.shape[:2]
    result_img = img.copy()
    
    # Define some sample objects to detect (simulated)
    objects = [
        {'class': 'car', 'confidence': 0.92, 'box': [int(width*0.2), int(height*0.6), int(width*0.4), int(height*0.8)]},
        {'class': 'person', 'confidence': 0.85, 'box': [int(width*0.6), int(height*0.5), int(width*0.7), int(height*0.8)]},
        {'class': 'traffic light', 'confidence': 0.75, 'box': [int(width*0.8), int(height*0.2), int(width*0.85), int(height*0.3)]}
    ]
    
    # Add some randomization to make it more interesting
    if random.random() > 0.5:
        objects.append({'class': 'bicycle', 'confidence': 0.7, 'box': [int(width*0.4), int(height*0.7), int(width*0.5), int(height*0.85)]})
    
    if random.random() > 0.7:
        objects.append({'class': 'dog', 'confidence': 0.65, 'box': [int(width*0.1), int(height*0.7), int(width*0.2), int(height*0.85)]})
    
    # Draw boxes on the image
    for obj in objects:
        label = f"{obj['class']}: {obj['confidence']:.2f}"
        x, y, w, h = obj['box']
        color = [int(c) for c in COLORS[CLASSES.index(obj['class']) % len(COLORS)]]
        cv2.rectangle(result_img, (x, y), (w, h), color, 2)
        # Add text background
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(result_img, (x, y-25), (x+text_size[0], y), color, -1)
        # Add text
        cv2.putText(result_img, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return result_img, objects

def detect_objects(img):
    """
    Detect objects in an image using YOLOv3 or dummy detection
    
    Args:
        img: Image to process (numpy array)
        
    Returns:
        Image with detection boxes and labels drawn
    """
    # Initialize model (if not initialized)
    model = initialize_yolo()
    
    # If model initialization failed or weights not available, use dummy detection
    if model is None:
        return dummy_detect(img)
    
    height, width = img.shape[:2]
    result_img = img.copy()
    
    # Prepare image for neural network
    blob = cv2.dnn.blobFromImage(img, 0.00392, (model['width'], model['height']), (0, 0, 0), True, False)
    
    # Set the input and run forward pass
    model['net'].setInput(blob)
    outs = model['net'].forward(model['output_layers'])
    
    # Process detection results
    class_ids = []
    confidences = []
    boxes = []
    
    # Process each detection
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5:  # Only keep detections with confidence > 0.5
                # Convert YOLO coordinates to actual coordinates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, x + w, y + h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply non-maximum suppression to remove redundant overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    # Create detection objects
    objects = []
    
    # Draw final detections
    for i in indices:
        if isinstance(i, list):
            i = i[0]  # Handle different OpenCV versions that return nested lists
        
        box = boxes[i]
        x, y, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]
        class_id = class_ids[i]
        
        label = f"{CLASSES[class_id]}: {confidences[i]:.2f}"
        color = [int(c) for c in COLORS[class_id]]
        
        # Save object info
        objects.append({
            'class': CLASSES[class_id],
            'confidence': confidences[i],
            'box': [x, y, w, h]
        })
        
        # Draw bounding box
        cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 2)
        
        # Draw label background
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(result_img, (x, y - 25), (x + text_size[0], y), color, -1)
        
        # Draw label text
        cv2.putText(result_img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return result_img, objects
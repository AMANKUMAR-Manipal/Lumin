import cv2
import numpy as np
import os
import logging
import time

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Global variables for YOLO model
net = None
output_layers = None
classes = None
colors = None
use_dummy_detection = False  # Flag to indicate if we're using dummy detection

def initialize_yolo():
    """
    Initialize YOLOv3 model with pre-trained weights, configuration,
    and COCO class names
    """
    global net, output_layers, classes, colors, use_dummy_detection
    
    try:
        # Load COCO class names
        classes_path = "coco.names"
        
        # If file doesn't exist, create it with the COCO class names
        if not os.path.exists(classes_path):
            logger.info("Creating COCO classes file")
            with open(classes_path, 'w') as f:
                f.write('\n'.join([
                    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
                    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
                    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
                    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
                ]))
        
        with open(classes_path, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        
        # Generate random colors for each class
        np.random.seed(42)  # for reproducibility
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
        
        # Check for YOLOv3 weights and config files
        weights_path = "yolov3.weights"
        config_path = "yolov3.cfg"
        
        # Check if weights file exists, if not provide instructions to download
        if not os.path.exists(weights_path) or os.path.getsize(weights_path) < 1000000:  # Real weights is >200MB
            logger.warning(f"YOLO weights file not found or invalid at {weights_path}")
            logger.warning("Please download YOLOv3 weights from https://pjreddie.com/media/files/yolov3.weights")
            logger.warning("Using dummy detection mode for demonstration purposes")
            use_dummy_detection = True
            
            # Create a simple dummy config
            if not os.path.exists(config_path):
                with open(config_path, 'w') as f:
                    f.write("[net]\nbatch=1\nwidth=416\nheight=416\nchannels=3\n")
            return
        
        # Check if config file exists, if not create it
        if not os.path.exists(config_path):
            logger.info("Creating YOLOv3 config file")
            with open(config_path, 'w') as f:
                f.write("""
[net]
batch=1
subdivisions=1
width=416
height=416
channels=3
momentum=0.9
decay=0.0005
angle=0
saturation = 1.5
exposure = 1.5
hue=.1

learning_rate=0.001
burn_in=1000
max_batches = 500200
policy=steps
steps=400000,450000
scales=.1,.1

[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=leaky

# ... more layers would be here in a real config file
                
[yolo]
mask = 6,7,8
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=80
num=9
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1

[route]
layers = -4

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[upsample]
stride=2

[route]
layers = -1, 36

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters=255
activation=linear

[yolo]
mask = 3,4,5
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=80
num=9
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1

[route]
layers = -4

[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=leaky

[upsample]
stride=2

[route]
layers = -1, 11

[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters=255
activation=linear

[yolo]
mask = 0,1,2
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=80
num=9
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1
                """)
        
        try:
            # Load YOLO network
            logger.info("Loading YOLO network")
            net = cv2.dnn.readNet(weights_path, config_path)
            
            # Get output layer names
            layer_names = net.getLayerNames()
            try:
                # OpenCV 4.5.4+
                output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
            except:
                # Older OpenCV versions
                output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
            
            logger.info("YOLO initialization completed")
        except Exception as e:
            logger.error(f"Error loading YOLO model: {str(e)}")
            logger.warning("Falling back to dummy detection mode")
            use_dummy_detection = True
    
    except Exception as e:
        logger.error(f"Error initializing YOLO: {str(e)}")
        logger.warning("Falling back to dummy detection mode")
        use_dummy_detection = True

def dummy_detect(img):
    """
    A dummy detection function that draws example boxes
    to demonstrate how the real detection would work
    """
    global classes, colors
    
    height, width, _ = img.shape
    
    # Draw a "demo mode" text
    cv2.putText(
        img, 
        "DEMO MODE - YOLOv3 weights not loaded", 
        (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.7, 
        (0, 0, 255), 
        2
    )
    
    # Generate some "dummy" detections based on image regions
    detections = []
    
    # Add a fixed detection in the center
    center_x, center_y = width // 2, height // 2
    w, h = width // 4, height // 4
    x = center_x - w // 2
    y = center_y - h // 2
    
    # Use a deterministic "random" class based on time
    class_idx = int(time.time() / 3) % len(classes)
    confidence = 0.85
    
    detections.append({
        "box": [x, y, w, h],
        "class": class_idx,
        "confidence": confidence
    })
    
    # Add a few more detections in other areas
    regions = [
        [width // 4, height // 4, width // 6, height // 6],
        [width * 3 // 4, height // 4, width // 5, height // 5],
        [width // 4, height * 3 // 4, width // 5, height // 7],
    ]
    
    for i, (rx, ry, rw, rh) in enumerate(regions):
        if (i + int(time.time() / 2)) % 3 == 0:  # Only show some regions at a time
            class_idx = (class_idx + i + 1) % len(classes)
            confidence = 0.7 - (i * 0.1)
            detections.append({
                "box": [rx, ry, rw, rh],
                "class": class_idx,
                "confidence": confidence
            })
    
    # Draw the detections
    font = cv2.FONT_HERSHEY_SIMPLEX
    for det in detections:
        x, y, w, h = det["box"]
        class_id = det["class"]
        confidence = det["confidence"]
        
        label = str(classes[class_id])
        color = colors[class_id]
        
        # Draw rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        
        # Create label text with class name and confidence
        label_text = f"{label}: {confidence:.2f}"
        
        # Draw filled rectangle for text background
        text_size = cv2.getTextSize(label_text, font, 0.5, 2)[0]
        cv2.rectangle(img, (x, y - 25), (x + text_size[0], y), color, -1)
        
        # Draw text
        cv2.putText(img, label_text, (x, y - 7), font, 0.5, (0, 0, 0), 2)
    
    return img

def detect_objects(img):
    """
    Detect objects in an image using YOLOv3 or dummy detection
    
    Args:
        img: Image to process (numpy array)
        
    Returns:
        Image with detection boxes and labels drawn
    """
    global net, output_layers, classes, colors, use_dummy_detection
    
    if use_dummy_detection:
        return dummy_detect(img)
    
    if net is None:
        logger.error("YOLO model not initialized")
        return dummy_detect(img)
    
    try:
        height, width, _ = img.shape
        
        # Prepare the image for YOLO
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        
        # Set input to the network
        net.setInput(blob)
        
        # Run forward pass
        outs = net.forward(output_layers)
        
        # Process detection results
        class_ids = []
        confidences = []
        boxes = []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Filter detections with confidence > 0.5
                if confidence > 0.5:
                    # Calculate bounding box coordinates
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression to remove redundant overlapping boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        # Draw bounding boxes and labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = colors[class_ids[i]]
                
                # Draw rectangle
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                
                # Create label text with class name and confidence
                label_text = f"{label}: {confidence:.2f}"
                
                # Draw filled rectangle for text background
                text_size = cv2.getTextSize(label_text, font, 0.5, 2)[0]
                cv2.rectangle(img, (x, y - 25), (x + text_size[0], y), color, -1)
                
                # Draw text
                cv2.putText(img, label_text, (x, y - 7), font, 0.5, (0, 0, 0), 2)
        
        return img
    
    except Exception as e:
        logger.error(f"Error in object detection: {str(e)}")
        # Fall back to dummy detection if YOLO fails
        return dummy_detect(img)

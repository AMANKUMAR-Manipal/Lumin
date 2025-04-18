import cv2
import numpy as np
import random

def draw_bounding_boxes(image, boxes, labels, confidences, thickness=2):
    """
    Draw bounding boxes on the image
    
    Args:
        image: OpenCV image
        boxes: List of bounding boxes [x1, y1, x2, y2]
        labels: List of class labels
        confidences: List of confidence scores
        thickness: Line thickness
        
    Returns:
        Image with bounding boxes
    """
    # Generate random colors for each class
    colors = {}
    
    for i, (box, label, confidence) in enumerate(zip(boxes, labels, confidences)):
        # Create a consistent color for each class
        if label not in colors:
            colors[label] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
        
        color = colors[label]
        
        # Draw bounding box
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label and confidence
        label_text = f"{label}: {confidence:.2f}"
        label_size, baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y1 = max(y1, label_size[1])
        
        # Draw label background
        cv2.rectangle(
            image, 
            (x1, y1 - label_size[1] - 5), 
            (x1 + label_size[0], y1), 
            color, 
            -1
        )
        
        # Draw label text
        cv2.putText(
            image,
            label_text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
    
    return image

def preprocess_image(image, target_size=None):
    """
    Preprocess image for object detection
    
    Args:
        image: OpenCV image
        target_size: (width, height) tuple or None
        
    Returns:
        Preprocessed image
    """
    # Resize if target size is provided
    if target_size:
        image = cv2.resize(image, target_size)
    
    return image

def get_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
        
    Returns:
        IoU score
    """
    # Get coordinates of intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Intersection area
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    
    # IoU
    iou = intersection_area / union_area if union_area > 0 else 0
    
    return iou

def compare_detections(detections1, detections2, iou_threshold=0.5):
    """
    Compare detection results from two different algorithms
    
    Args:
        detections1: (labels, confidences, boxes) from algorithm 1
        detections2: (labels, confidences, boxes) from algorithm 2
        iou_threshold: Threshold for matching boxes
        
    Returns:
        Comparison metrics
    """
    labels1, confidences1, boxes1 = detections1
    labels2, confidences2, boxes2 = detections2
    
    # Match detections between algorithms
    matches = []
    
    for i, box1 in enumerate(boxes1):
        best_match = -1
        best_iou = iou_threshold
        
        for j, box2 in enumerate(boxes2):
            iou = get_iou(box1, box2)
            
            if iou > best_iou:
                best_iou = iou
                best_match = j
        
        if best_match != -1:
            matches.append((i, best_match, best_iou))
    
    # Calculate metrics
    metrics = {
        'matched_count': len(matches),
        'algo1_only': len(boxes1) - len(matches),
        'algo2_only': len(boxes2) - len(matches),
        'matched_classes': sum(1 for i, j, _ in matches if labels1[i] == labels2[j]),
        'avg_iou': sum(iou for _, _, iou in matches) / len(matches) if matches else 0
    }
    
    return metrics

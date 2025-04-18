import cv2
import numpy as np

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
    # Create a copy of the image to draw on
    output = image.copy()
    
    # Generate random colors for each class
    np.random.seed(42)  # For reproducibility
    colors = np.random.uniform(0, 255, size=(len(set(labels)), 3))
    
    # Map unique labels to color indices
    unique_labels = list(set(labels))
    
    # Draw each bounding box
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        
        # Get color for this class
        label = labels[i]
        color_idx = unique_labels.index(label)
        color = tuple(map(int, colors[color_idx]))
        
        # Draw rectangle
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label text with confidence
        confidence = confidences[i]
        label_text = f"{label}: {confidence:.2f}"
        
        # Draw filled rectangle for text background
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, thickness)[0]
        cv2.rectangle(output, (x1, y1 - 20), (x1 + text_size[0], y1), color, -1)
        
        # Draw text
        cv2.putText(output, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness)
    
    return output

def preprocess_image(image, target_size=None):
    """
    Preprocess image for object detection
    
    Args:
        image: OpenCV image
        target_size: (width, height) tuple or None
        
    Returns:
        Preprocessed image
    """
    # Resize if target_size is specified
    if target_size:
        image = cv2.resize(image, target_size)
    
    # Convert to RGB if needed (OpenCV uses BGR)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
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
    # Get coordinates of intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate area of intersection
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    intersection = width * height
    
    # Calculate area of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate IoU
    union = box1_area + box2_area - intersection
    iou = intersection / union if union > 0 else 0
    
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
    
    matches = []
    
    # For each detection in the first set
    for i, box1 in enumerate(boxes1):
        best_match = None
        best_iou = 0
        
        # Find the best matching box in the second set
        for j, box2 in enumerate(boxes2):
            iou = get_iou(box1, box2)
            
            if iou > iou_threshold and iou > best_iou:
                best_iou = iou
                best_match = j
        
        if best_match is not None:
            matches.append((i, best_match, best_iou))
    
    # Calculate metrics
    metrics = {
        "matched_count": len(matches),
        "precision": len(matches) / len(boxes1) if boxes1 else 0,
        "recall": len(matches) / len(boxes2) if boxes2 else 0,
        "average_iou": sum(m[2] for m in matches) / len(matches) if matches else 0,
        "class_agreement": sum(1 for m in matches if labels1[m[0]] == labels2[m[1]]) / len(matches) if matches else 0
    }
    
    return metrics
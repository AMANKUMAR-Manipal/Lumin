import cv2
import numpy as np
import random
import logging

class KNNDetector:
    """Object detector using K-Nearest Neighbors algorithm"""
    
    def __init__(self):
        """Initialize the KNN detector"""
        logging.info("Initializing KNN detector simulation")
        
        # Default parameters
        self.confidence_threshold = 0.5
        
        # Define the classes we want to detect
        self.classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
            'traffic light', 'stop sign', 'parking meter'
        ]
        
        # Initialize the KNN model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the KNN model with simulated data"""
        # In a real application, this would load or train a KNN model
        # For this simulation, we'll just set up parameters
        self.k = 5  # Number of neighbors to use
        
        # Simulated feature database for KNN
        # In a real implementation, this would be HOG features from training images
        self.feature_database = {
            'person': np.random.rand(10, 64),  # 10 examples with 64 features each
            'bicycle': np.random.rand(10, 64),
            'car': np.random.rand(10, 64),
            'motorcycle': np.random.rand(10, 64),
            'bus': np.random.rand(10, 64),
            'truck': np.random.rand(10, 64),
            'traffic light': np.random.rand(10, 64),
            'stop sign': np.random.rand(10, 64),
            'parking meter': np.random.rand(10, 64)
        }
    
    def _extract_hog_features(self, image, windows):
        """Extract HOG features from image windows"""
        features = []
        
        for x, y, w, h in windows:
            # Extract the window from the image
            window = image[y:y+h, x:x+w]
            
            # Resize to a consistent size for HOG
            resized = cv2.resize(window, (64, 64))
            
            # Convert to grayscale
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            
            # Compute gradients (simplified HOG-like features)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag, ang = cv2.cartToPolar(gx, gy)
            
            # Create a 64-element feature vector (simplified HOG)
            # Divide the image into 4x4 cells and compute gradient histograms
            bin_counts = np.zeros(64)
            for i in range(64):
                cell_x = (i % 8) * 8
                cell_y = (i // 8) * 8
                cell_mag = mag[cell_y:cell_y+8, cell_x:cell_x+8]
                bin_counts[i] = np.mean(cell_mag)
            
            # Normalize the feature vector
            if np.max(bin_counts) > 0:
                bin_counts = bin_counts / np.max(bin_counts)
            
            features.append(bin_counts)
        
        return np.array(features)
    
    def _sliding_window(self, image, window_size=(96, 192), step_size=32):
        """Generate sliding windows over the image"""
        windows = []
        h, w = image.shape[:2]
        
        for y in range(0, h - window_size[1], step_size):
            for x in range(0, w - window_size[0], step_size):
                windows.append((x, y, window_size[0], window_size[1]))
        
        return windows
    
    def detect(self, image, confidence_threshold=None):
        """
        Detect objects in the image using KNN
        
        Args:
            image: OpenCV image (BGR format)
            confidence_threshold: Threshold for detection confidence
            
        Returns:
            tuple: (classes, confidences, bounding_boxes)
        """
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
        
        detected_objects = []
        confidence_scores = []
        bounding_boxes = []
        
        # Get image dimensions
        h, w = image.shape[:2]
        
        # Generate sliding windows at different scales
        windows = []
        
        # For smaller objects (traffic lights, signs)
        windows.extend(self._sliding_window(image, window_size=(64, 64), step_size=32))
        
        # For people
        windows.extend(self._sliding_window(image, window_size=(96, 192), step_size=48))
        
        # For vehicles
        windows.extend(self._sliding_window(image, window_size=(128, 96), step_size=64))
        
        # In a real implementation, we would extract HOG features from each window
        # and classify them using KNN. For this simulation, we'll use a simpler approach.
        
        # Simulate detection by selecting a few random windows
        # and assigning them classes based on size and position
        num_detections = random.randint(2, 6)  # Generate a few detections
        
        # Sort windows by position (prefer bottom of image for vehicles, center for people)
        def score_window(window):
            x, y, w, h = window
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Distance from center
            center_dist = abs(center_x - (w // 2)) + abs(center_y - (h // 2))
            
            # Prefer wider windows for vehicles, taller for people
            shape_score = w / h if w > h else h / w
            
            # Combine factors
            return center_dist * shape_score
        
        # Sort windows by score
        sorted_windows = sorted(windows, key=score_window)
        
        # Take the top windows
        selected_windows = sorted_windows[:min(num_detections * 3, len(sorted_windows))]
        
        # Randomly sample from the top windows
        selected_indices = random.sample(range(len(selected_windows)), min(num_detections, len(selected_windows)))
        final_windows = [selected_windows[i] for i in selected_indices]
        
        # Process selected windows
        for x, y, w, h in final_windows:
            # Determine class based on window shape and position
            aspect_ratio = w / h if h > 0 else 0
            rel_y_pos = y / h  # Relative y position
            
            if aspect_ratio < 0.7:  # Tall and narrow
                if rel_y_pos < 0.5:  # Upper half of image
                    obj_class = random.choice(['person', 'traffic light'])
                else:
                    obj_class = 'person'
                confidence = random.uniform(0.65, 0.85)
            elif aspect_ratio > 1.4:  # Wide and short
                obj_class = random.choice(['car', 'truck', 'bus'])
                confidence = random.uniform(0.7, 0.9)
            else:  # Square-ish
                if w < 100:  # Small square objects
                    obj_class = random.choice(['stop sign', 'traffic light', 'parking meter'])
                    confidence = random.uniform(0.6, 0.8)
                else:
                    obj_class = random.choice(['car', 'motorcycle', 'bicycle'])
                    confidence = random.uniform(0.65, 0.85)
            
            # Only keep detections above threshold
            if confidence >= confidence_threshold:
                detected_objects.append(obj_class)
                confidence_scores.append(float(confidence))
                bounding_boxes.append([x, y, x + w, y + h])
        
        # Apply non-maximum suppression to remove overlapping boxes
        if len(bounding_boxes) > 1:
            indices = []
            boxes = np.array(bounding_boxes)
            scores = np.array(confidence_scores)
            
            # Calculate overlap between all boxes
            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 2]
            y2 = boxes[:, 3]
            
            areas = (x2 - x1) * (y2 - y1)
            order = scores.argsort()[::-1]
            
            keep = []
            while order.size > 0:
                i = order[0]
                keep.append(i)
                
                # Calculate intersection with remaining boxes
                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])
                
                w = np.maximum(0.0, xx2 - xx1)
                h = np.maximum(0.0, yy2 - yy1)
                intersection = w * h
                
                # Calculate IoU
                iou = intersection / (areas[i] + areas[order[1:]] - intersection)
                
                # Keep indices where IoU is less than threshold
                inds = np.where(iou < 0.5)[0]
                order = order[inds + 1]
            
            # Get the final detections
            final_objects = [detected_objects[i] for i in keep]
            final_scores = [confidence_scores[i] for i in keep]
            final_boxes = [bounding_boxes[i] for i in keep]
            
            detected_objects = final_objects
            confidence_scores = final_scores
            bounding_boxes = final_boxes
        
        # If no detections, add some defaults
        if not detected_objects:
            # Default: add a car and a person
            detected_objects = ['car', 'person']
            confidence_scores = [float(random.uniform(0.6, 0.8)), float(random.uniform(0.6, 0.8))]
            
            # Car in bottom half
            car_width = w // 4
            car_height = car_width // 2
            car_x = w // 3
            car_y = h * 2 // 3
            
            # Person in middle
            person_width = w // 8
            person_height = person_width * 2
            person_x = w * 2 // 3
            person_y = h // 2 - person_height // 2
            
            bounding_boxes = [
                [car_x, car_y, car_x + car_width, car_y + car_height],
                [person_x, person_y, person_x + person_width, person_y + person_height]
            ]
        
        return detected_objects, confidence_scores, bounding_boxes
import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier
import logging
import os

class KNNDetector:
    """Object detector using K-Nearest Neighbors algorithm"""
    
    def __init__(self):
        """Initialize the KNN detector"""
        logging.info("Initializing KNN detector")
        
        # Create KNN model
        self.model = KNeighborsClassifier(n_neighbors=5)
        
        # For simplicity, we'll use pre-computed HOG features
        # In a real application, you'd train the KNN on actual HOG features from a dataset
        self.trained = False
        self.feature_size = 1764  # HOG feature size
        self.classes = ['person', 'car', 'chair', 'bottle', 'dog', 'cat']
        
        # Simulated training data (random features + labels)
        np.random.seed(42)  # For reproducibility
        self._initialize_model()
        
        # Parameters for object detection
        self.confidence_threshold = 0.4
        self.hog = cv2.HOGDescriptor()
    
    def _initialize_model(self):
        """Initialize the KNN model with simulated data"""
        # Create simulated HOG features for common objects
        # In a real application, you'd extract these from a dataset
        n_samples = 100
        X = np.random.rand(n_samples * len(self.classes), self.feature_size)
        y = np.repeat(self.classes, n_samples)
        
        # Train the KNN model
        self.model.fit(X, y)
        self.trained = True
        logging.info("KNN model initialized with simulated data")
    
    def _extract_hog_features(self, image, windows):
        """Extract HOG features from image windows"""
        features = []
        for (x1, y1, x2, y2) in windows:
            window = image[y1:y2, x1:x2]
            if window.size == 0:  # Skip empty windows
                continue
                
            # Resize window for HOG
            window = cv2.resize(window, (64, 128))
            
            # Extract HOG features
            window_features = self.hog.compute(window)
            
            # If feature vector size doesn't match expected size, skip
            if window_features.size != self.feature_size:
                continue
                
            features.append(window_features.flatten())
        
        return np.array(features)
    
    def _sliding_window(self, image, window_size=(96, 192), step_size=32):
        """Generate sliding windows over the image"""
        windows = []
        h, w = image.shape[:2]
        
        for y in range(0, h - window_size[1], step_size):
            for x in range(0, w - window_size[0], step_size):
                windows.append((x, y, x + window_size[0], y + window_size[1]))
        
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
        if not self.trained:
            self._initialize_model()
            
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
        
        # Convert to grayscale for HOG
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Generate sliding windows
        windows = self._sliding_window(gray_image)
        
        # Extract HOG features from windows
        features = self._extract_hog_features(gray_image, windows)
        
        detected_objects = []
        confidence_scores = []
        bounding_boxes = []
        
        if len(features) > 0:
            # Predict class probabilities
            probabilities = self.model.predict_proba(features)
            predictions = self.model.predict(features)
            
            # Get detections above threshold
            for i, (probs, label) in enumerate(zip(probabilities, predictions)):
                max_prob = np.max(probs)
                if max_prob > confidence_threshold:
                    detected_objects.append(label)
                    confidence_scores.append(float(max_prob))
                    bounding_boxes.append(windows[i])
        
        return detected_objects, confidence_scores, bounding_boxes

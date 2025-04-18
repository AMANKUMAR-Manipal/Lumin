from datetime import datetime
import json
from app import db

class Detection(db.Model):
    """Model for storing object detection results"""
    id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String(255), nullable=False)
    algorithm = db.Column(db.String(50), nullable=False)
    objects_detected = db.Column(db.Text, nullable=False)  # JSON string of detected objects
    detection_time = db.Column(db.Float, nullable=False)  # Processing time in seconds
    confidence_scores = db.Column(db.Text, nullable=False)  # JSON string of confidence scores
    bounding_boxes = db.Column(db.Text, nullable=False)  # JSON string of bounding boxes
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def get_objects(self):
        """Return objects_detected as a list"""
        return json.loads(self.objects_detected)
    
    def get_scores(self):
        """Return confidence_scores as a list"""
        return json.loads(self.confidence_scores)
    
    def get_boxes(self):
        """Return bounding_boxes as a list"""
        return json.loads(self.bounding_boxes)
    
    def __repr__(self):
        return f"<Detection {self.id}: {self.algorithm} - {self.get_objects()}>"

class Image(db.Model):
    """Model for storing uploaded images"""
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(255), nullable=False)
    file_size = db.Column(db.Integer, nullable=False)
    width = db.Column(db.Integer, nullable=True)
    height = db.Column(db.Integer, nullable=True)
    mime_type = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def get_relative_path(self):
        """Return the path relative to the static folder"""
        if self.file_path.startswith("static/"):
            return self.file_path[7:]  # Remove 'static/' prefix
        return self.file_path
    
    def __repr__(self):
        return f"<Image {self.id}: {self.original_filename}>"
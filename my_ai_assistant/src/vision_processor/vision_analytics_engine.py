# TODO: add face detection
# TODO: emotion detection
# TODO: face recognition capabilities

import asyncio
from concurrent.futures import ProcessPoolExecutor
import torch
import cv2
import numpy as np
from ultralytics import YOLO

class VisionAnalyticsEngine:
    """
    Object Detection Engine using YOLOv9 with distance estimation capabilities.
    """
    # Constants for distance estimation
    KNOWN_WIDTH = 16.0  # Width of a reference object in inches
    KNOWN_DISTANCE = 50  # Reference distance (can be adjusted)

    def __init__(self, model_path='./models/yolov9-c.pt'):
        """
        Initialize the VisionAnalyticsEngine with a YOLO model.
        
        :param model_path: Path to the YOLOv9 model weights
        """
        # Load the YOLO model
        self.model = YOLO(model_path)
        
        # Estimate focal length (may need calibration)
        self.focal_length = 1000
        self.process_executor = ProcessPoolExecutor(max_workers=2)

    def calculate_distance(self, bbox_width, frame_width):
        """
        Calculate distance based on bounding box width.
        
        :param bbox_width: Width of the detected object's bounding box
        :param frame_width: Width of the frame
        :return: Estimated distance
        """
        return (self.KNOWN_WIDTH * self.focal_length) / bbox_width

    async def detect_objects_and_measure_distance_async(self, frame):
        """Asynchronous object detection with distance measurement"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.process_executor, 
            self.detect_objects_and_measure_distance, 
            frame
        )

    def detect_objects_and_measure_distance(self, frame):
        """Perform object detection and distance measurement on a frame"""
        results = self.model(frame)
        detection_results = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                conf = box.conf[0]
                cls = int(box.cls[0])

                if conf > 0.5:
                    label = self.model.names[cls]
                    bbox_width = x2 - x1
                    distance = self.calculate_distance(bbox_width, frame.shape[1])

                    detection_results.append({
                        'object': label,
                        'confidence': round(float(conf), 2),
                        'distance': round(distance, 2),
                        'bbox': (x1, y1, x2, y2)
                    })

                    # Optional: Draw bounding boxes and labels
                    self._draw_detection(frame, label, conf, x1, y1, x2, y2, distance)

        return frame, detection_results

    def _draw_detection(self, frame, label, conf, x1, y1, x2, y2, distance):
        """Draw detection results on frame"""
        color = (0, 255, 0)  # Green color
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        label_text = f'{label} {conf:.2f}'
        cv2.putText(frame, label_text, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        distance_label = f'Distance: {distance:.2f} inches'
        cv2.putText(frame, distance_label, (x1, y2 + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    def calibrate_focal_length(self, reference_image_path, known_object_width):
        """
        Calibrate focal length using a reference image.
        
        :param reference_image_path: Path to the reference image
        :param known_object_width: Known width of the reference object
        :return: Calculated focal length
        """
        # Read reference image
        ref_image = cv2.imread(reference_image_path)
        if ref_image is None:
            raise ValueError(f"Could not load reference image from {reference_image_path}")

        # Detect objects in reference image
        results = self.model(ref_image)
        
        # Find the first detection with sufficient confidence
        for r in results:
            boxes = r.boxes
            for box in boxes:
                conf = box.conf[0]
                if conf > 0.5:
                    # Calculate bounding box width
                    x1, y1, x2, y2 = box.xyxy[0]
                    bbox_width = x2 - x1

                    # Calculate focal length
                    self.focal_length = (bbox_width * self.KNOWN_DISTANCE) / known_object_width
                    return self.focal_length

        raise ValueError("No suitable object found for focal length calibration")

# def create_object_detector(model_path='./models/yolov9-c.pt'):
#     """
#     Factory function to create and initialize an VisionAnalyticsEngine instance.
    
#     :param model_path: Path to the YOLO model weights
#     :return: Initialized VisionAnalyticsEngine
#     """
#     detector = VisionAnalyticsEngine(model_path)
#     return detector

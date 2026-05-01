import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from tqdm import tqdm

import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Classes
# ============================================================================

class ObjectClass(Enum):
    """Supported railway object classes for open-vocabulary detection."""
    user_defined_classes = [
        "a lifted gate arm with red and white stripes",
        # "traffic sign",
        # "traffic light",
    ]
    
    @classmethod
    def get_labels(cls) -> List[str]:
        """Get all class labels as strings."""
        return cls.user_defined_classes.value

    @classmethod
    def get_color(cls, class_name: str) -> Tuple[int, int, int]:
        """Get BGR color for each class from tab20 colormap."""
        # Get tab20 colormap
        cmap = cm.get_cmap('tab20')
        
        # Map class names to color indices
        class_to_idx = {class_name: idx for idx, class_name in enumerate(cls.get_labels())}
        
        # Get color index for the class
        color_idx = class_to_idx.get(class_name, 0)
        
        # Get RGBA color from tab20
        rgba = cmap(color_idx)
        
        # Convert RGBA [0, 1] to BGR [0, 255] for OpenCV
        b = int(rgba[2] * 255)
        g = int(rgba[1] * 255)
        r = int(rgba[0] * 255)
        
        return (b, g, r)

@dataclass
class DetectionResult:
    """Standardized detection result format."""
    image_path: str
    image_name: str
    model_name: str
    width: int
    height: int
    detections: List[Dict[str, Any]]  # List of {class, confidence, x1, y1, x2, y2}


# ============================================================================
# Abstract Base Class
# ============================================================================

class ObjectDetector(ABC):
    """Abstract base class for all object detectors."""

    def __init__(self, model_name: str, confidence_threshold: float = 0.5):
        """
        Initialize detector.

        Args:
            model_name: Name of the model
            confidence_threshold: Minimum confidence score to keep detections
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = self._get_device()
        self.model = None

    @staticmethod
    def _get_device() -> str:
        """Get appropriate device (cuda or cpu)."""
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    @abstractmethod
    def load_model(self) -> None:
        """Load the model."""
        pass

    @abstractmethod
    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Perform detection on image.

        Args:
            image: Input image as numpy array (H, W, C) in BGR format

        Returns:
            Dict with keys: 'boxes', 'confidences', 'classes'
            - boxes: List of [x1, y1, x2, y2] in pixel coordinates
            - confidences: List of confidence scores
            - classes: List of class indices or names
        """
        pass


# ============================================================================
# YOLOE Detector
# ============================================================================

class YOLOEDetector(ObjectDetector):
    """Detector for YOLOE segmentation models."""

    SUPPORTED_MODELS = [
        "yoloe-26n-seg",
        "yoloe-26s-seg",
        "yoloe-26m-seg",
        "yoloe-26l-seg",
        "yoloe-26x-seg",
        "yoloe-v8s-seg",
        "yoloe-v8l-seg",
        "yoloe-11s-seg",
        "yoloe-11l-seg",
    ]

    def __init__(
        self,
        model_name: str,
        confidence_threshold: float = 0.5,
    ):
        """
        Initialize YOLOE detector.

        Args:
            model_name: Model name (e.g., 'yoloe-26n-seg')
            confidence_threshold: Confidence threshold
        """
        super().__init__(model_name, confidence_threshold)
        self.model_path = model_name + ".pt"
        self.load_model()

    def load_model(self) -> None:
        """Load YOLOE model."""
        from ultralytics import YOLO
        
        self.model = YOLO(self.model_path)  # type: ignore
        self.model.to(self.device)  # type: ignore
        logger.info(f"Loaded YOLOE model: {self.model_name} on {self.device}")

        # Set user-defined classes from ObjectClass enum
        class_labels = ObjectClass.get_labels()
        self.model.set_classes(class_labels)  # type: ignore
        logger.info(f"Set custom classes: {self.model.names}")  # type: ignore

    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect objects in image using YOLOE.

        Args:
            image: Input image (H, W, C) in BGR format

        Returns:
            Standardized detection dict
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Run inference
        results = self.model.predict(image, conf=self.confidence_threshold, verbose=False)
        result = results[0]

        boxes = []
        confidences = []
        classes = []

        # Extract detections
        if result.boxes is not None:  # type: ignore
            for box in result.boxes:  # type: ignore
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # type: ignore
                conf = box.conf[0].cpu().item()  # type: ignore
                cls_idx = int(box.cls[0].cpu().item())  # type: ignore

                boxes.append([float(x1), float(y1), float(x2), float(y2)])
                confidences.append(float(conf))
                classes.append(cls_idx)

        return {
            "boxes": boxes,
            "confidences": confidences,
            "classes": classes,
        }


# ============================================================================
# Grounding DINO Detector
# ============================================================================

class GroundingDINODetector(ObjectDetector):
    """Detector for Grounding DINO models using HuggingFace transformers."""

    SUPPORTED_MODELS = [
        "IDEA-Research/grounding-dino-tiny",
        "IDEA-Research/grounding-dino-base",
    ]

    def __init__(
        self,
        model_name: str,
        text_prompt: Optional[List[str]] = None,
        confidence_threshold: float = 0.5,
    ):
        """
        Initialize Grounding DINO detector using HuggingFace.

        Args:
            model_name: Model ID from HuggingFace (e.g., 'IDEA-Research/grounding-dino-tiny')
            text_prompt: List of class labels for detection
            confidence_threshold: Confidence threshold
        """
        super().__init__(model_name, confidence_threshold)
        self.text_prompt = text_prompt or ObjectClass.get_labels()
        self.processor = None
        self.load_model()

    def load_model(self) -> None:
        """Load Grounding DINO model from HuggingFace."""
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
            
        logger.info(f"Loaded Grounding DINO model: {self.model_name} on {self.device}")
        
    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect objects in image using Grounding DINO.

        Args:
            image: Input image (H, W, C) in BGR format

        Returns:
            Standardized detection dict
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Convert BGR to RGB for processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Get image dimensions
        h, w = image.shape[:2]

        # Prepare text labels as list of lists (batch format expected by processor)
        text_labels = [self.text_prompt]

        # Process inputs
        inputs = self.processor(
            images=pil_image,
            text=text_labels,
            return_tensors="pt"
        ).to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process results
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.confidence_threshold,
            text_labels=self.text_prompt,
            target_sizes=[(h, w)]
        )

        result = results[0]
        
        # Extract boxes, confidences, and classes
        result_boxes = []
        confidences = []
        classes = []

        for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
            # Box is already in [x1, y1, x2, y2] pixel coordinates
            x1, y1, x2, y2 = box.tolist()
            conf = score.item()
            
            result_boxes.append([float(x1), float(y1), float(x2), float(y2)])
            confidences.append(float(conf))
            classes.append(label)

        return {
            "boxes": result_boxes,
            "confidences": confidences,
            "classes": classes,
        }



# ============================================================================
# Detection Formatter
# ============================================================================

class DetectionFormatter:
    """Formats detector outputs into a standardized structure."""

    @staticmethod
    def standardize_detections(
        detection_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Convert detector output to standardized format.

        Args:
            detection_result: Output from detector

        Returns:
            List of detection dicts with keys: class, confidence, x1, y1, x2, y2
        """
        detections = []
        boxes = detection_result.get("boxes", [])
        confidences = detection_result.get("confidences", [])
        classes = detection_result.get("classes", [])

        for box, conf, cls in zip(boxes, confidences, classes):
            x1, y1, x2, y2 = box

            # Handle class - could be index or name
            if isinstance(cls, (int, np.integer)):
                # Map YOLO class indices to object class if needed
                class_name = ObjectClass.get_labels()[min(cls, len(ObjectClass.get_labels()) - 1)]
            else:
                class_name = str(cls)

            detection = {
                "class": class_name,
                "confidence": float(conf),
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "width": float(x2 - x1),
                "height": float(y2 - y1),
            }
            detections.append(detection)

        return detections


# ============================================================================
# Detection Visualizer
# ============================================================================

class DetectionVisualizer:
    """Draws bounding boxes on images with class-specific colors."""

    def __init__(self, font_size: int, line_thickness: int):
        """
        Initialize visualizer.

        Args:
            font_size: Font size for class labels
            line_thickness: Thickness of bounding box lines
        """
        self.font_size = font_size
        self.line_thickness = line_thickness

    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
    ) -> np.ndarray:
        """
        Draw bounding boxes on image.

        Args:
            image: Input image (H, W, C) in BGR format
            detections: List of detection dicts

        Returns:
            Image with drawn bounding boxes
        """
        output = image.copy()

        for detection in detections:
            x1 = int(detection["x1"])
            y1 = int(detection["y1"])
            x2 = int(detection["x2"])
            y2 = int(detection["y2"])
            class_name = detection["class"]
            confidence = detection["confidence"]

            # Get color for this class
            color = ObjectClass.get_color(class_name)

            # Draw rectangle
            cv2.rectangle(output, (x1, y1), (x2, y2), color, self.line_thickness)

            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size, baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
            )
            label_y = max(y1 - 5, label_size[1] + 5)
            cv2.rectangle(
                output,
                (x1, label_y - label_size[1] - 5),
                (x1 + label_size[0], label_y + baseline),
                color,
                cv2.FILLED,
            )
            cv2.putText(
                output,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return output


# ============================================================================
# Main Preprocessing Pipeline
# ============================================================================

def create_detector(
    model_name: str,
    confidence_threshold: float = 0.5,
) -> ObjectDetector:
    """
    Factory function to create appropriate detector.

    Args:
        model_name: Name of the model
        confidence_threshold: Confidence threshold

    Returns:
        Initialized ObjectDetector instance
    """
    if "dino" in model_name.lower():
        return GroundingDINODetector(
            model_name=model_name,
            confidence_threshold=confidence_threshold,
        )
    elif "yoloe" in model_name.lower():
        return YOLOEDetector(
            model_name=model_name,
            confidence_threshold=confidence_threshold,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def preprocess_image(
    cfg,
    model_name: str,
    confidence_threshold: float,
) -> pd.DataFrame:
    """
    Main preprocessing pipeline for object detection.

    Args:
        cfg: Configuration object with path information
        model_name: Name of detection model
        confidence_threshold: Minimum confidence to keep detections

    Returns:
        DataFrame with detection results
    """
    # Initialize detector
    detector = create_detector(
        model_name=model_name,
        confidence_threshold=confidence_threshold,
    )
    visualizer = DetectionVisualizer(font_size=12, line_thickness=2)

    # Load image sequence dataframe
    df_image_seq = pd.read_csv(cfg.path.df_image_seq)

    all_detections = []
    image_stats = {
        "total_images": 0,
        "images_with_detections": 0,
        "total_detections": 0,
        "detections_by_class": {},
    }

    # Process each image
    df_image_seq = df_image_seq.sort_values(by=['crossing_id', 'seq_id', 'img_pos'])
    df_image_seq = df_image_seq.iloc[:100]
    df_image_seq = df_image_seq[~df_image_seq[['crossing_id', 'seq_id']].duplicated()]
    for _, row in tqdm(df_image_seq.iterrows(), total=len(df_image_seq)):
        crossing_id = row['crossing_id']
        seq_id = row['seq_id']
        img_pos = row['img_pos']
        img_id = row['img_id']
        image_rel_path = os.path.join(crossing_id, seq_id, f"{str(img_pos).zfill(4)}_{img_id}.jpg")
        image_path = os.path.join(cfg.path.dir_image_seq, image_rel_path)

        image_stats["total_images"] += 1

        image = cv2.imread(image_path)
        h, w = image.shape[:2]

        detection_result = detector.detect(image)
        
        # Normalize detections
        detections = DetectionFormatter.standardize_detections(detection_result=detection_result)

        if detections:
            image_stats["images_with_detections"] += 1
            image_stats["total_detections"] += len(detections)

            # Update class statistics
            for detection in detections:
                class_name = detection["class"]
                image_stats["detections_by_class"][class_name] = image_stats["detections_by_class"].get(class_name, 0) + 1

            # Save annotated image
            output_image = visualizer.draw_detections(image, detections)
            output_path = os.path.join(cfg.path.dir_image_preprocessed, model_name, image_rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, output_image)

            # Add to results
            for detection in detections:
                result_row = {
                    "crossing_id": crossing_id,
                    "seq_id": seq_id,
                    "img_pos": img_pos,
                    "img_id": img_id,
                    **detection,
                }
                all_detections.append(result_row)

    # Create output dataframe
    df_preprocessed = pd.DataFrame(all_detections)

    # Save dataframe
    output_csv_path = os.path.join(cfg.path.dir_image_preprocessed, model_name, cfg.path.df_image_preprocessed.split('/')[-1])
    df_preprocessed.to_csv(output_csv_path, index=False)

    # Print summary statistics
    _print_summary_stats(image_stats, len(df_image_seq))

    return df_preprocessed


def _print_summary_stats(image_stats: Dict[str, Any], total_rows: int) -> None:
    """Print summary statistics of preprocessing."""
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total rows in df_image_seq: {total_rows}")
    print(f"Successfully processed: {image_stats['total_images']}")
    print(f"Images with detections: {image_stats['images_with_detections']}")
    print(f"Total detections: {image_stats['total_detections']}")

    if image_stats["total_images"] > 0:
        print(
            f"Average detections per image: "
            f"{image_stats['total_detections'] / image_stats['total_images']:.2f}"
        )

    if image_stats["detections_by_class"]:
        print("\nDetections by class:")
        for class_name, count in sorted(
            image_stats["detections_by_class"].items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {class_name}: {count}")

    print("=" * 60 + "\n")


# ============================================================================
# Utility Functions
# ============================================================================

def get_supported_models() -> Dict[str, List[str]]:
    """Get list of supported models by type."""
    return {
        "yoloe": YOLOEDetector.SUPPORTED_MODELS,
        "grounding_dino": GroundingDINODetector.SUPPORTED_MODELS,
    }


if __name__ == "__main__":
    # Example usage
    print("Module imported successfully. Use preprocess_images() to start pipeline.")

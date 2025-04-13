"""
Zero-shot PCB defect detection using Vision-Language Models.
"""
import os
import json
import numpy as np
from typing import List, Dict, Any, Union, Optional
from PIL import Image
import matplotlib.pyplot as plt

from .model import PCBDefectVLM

class PCBDefectDetector:
    """Zero-shot PCB defect detection with prompt-based categorization."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize the PCB defect detector.

        Args:
            model_name: Hugging Face model identifier for the VLM
        """
        self.vlm = PCBDefectVLM(model_name=model_name)
        self.defect_categories = []
        self.defect_prompts = {}

    def load_defect_categories(self, json_path: str) -> None:
        """
        Load defect categories and prompts from a JSON file.

        Args:
            json_path: Path to the JSON file containing defect categories
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Defect categories file not found at {json_path}")

        with open(json_path, 'r') as f:
            data = json.load(f)

        self.defect_categories = [item['category'] for item in data['defects']]

        # Store the detailed prompts for each category
        self.defect_prompts = {}
        for item in data['defects']:
            self.defect_prompts[item['category']] = item['prompts']

    def get_prompts_for_detection(self, enhance_with_domain: bool = True) -> List[str]:
        """
        Generate prompts for zero-shot detection.

        Args:
            enhance_with_domain: Whether to enhance prompts with domain-specific language

        Returns:
            List of formatted prompts for the model
        """
        if not self.defect_categories:
            raise ValueError("Defect categories not loaded. Call load_defect_categories first.")

        detection_prompts = []

        for category in self.defect_categories:
            # Get the most generic prompt for this category
            base_prompt = self.defect_prompts[category][0]

            if enhance_with_domain:
                # Format with PCB/semiconductor domain knowledge
                prompt = f"A PCB with {base_prompt}"
                prompt_alt = f"A printed circuit board showing {base_prompt}"
                detection_prompts.extend([prompt, prompt_alt])
            else:
                detection_prompts.append(base_prompt)

        # Always add a "normal" category
        detection_prompts.append("A normal PCB with no defects")
        detection_prompts.append("A perfectly manufactured printed circuit board")

        return detection_prompts

    def detect(self, image_path: str, threshold: float = 0.2,
               top_k: int = 3, enhance_prompts: bool = True) -> Dict[str, Any]:
        """
        Detect PCB defects in an image using zero-shot classification.

        Args:
            image_path: Path to the PCB image
            threshold: Confidence threshold for detection
            top_k: Number of top categories to return
            enhance_prompts: Whether to enhance prompts with domain-specific language

        Returns:
            Detection results with categories and confidence scores
        """
        # Get formatted prompts
        prompts = self.get_prompts_for_detection(enhance_with_domain=enhance_prompts)

        # Perform zero-shot classification
        raw_results = self.vlm.classify(image_path, prompts)

        # Post-process results to combine similar categories
        processed_results = self._process_results(raw_results)

        # Get top k results above threshold
        top_results = {k: v for k, v in sorted(
            processed_results.items(),
            key=lambda item: item[1],
            reverse=True
        ) if v >= threshold}

        # Limit to top k
        top_k_results = dict(list(top_results.items())[:top_k])

        # Determine if the PCB is defective
        is_defective = not any(k.lower().find("normal") >= 0 for k in list(top_k_results.keys())[:1])

        return {
            "is_defective": is_defective,
            "defects": top_k_results,
            "all_scores": processed_results
        }

    def _process_results(self, raw_results: Dict[str, float]) -> Dict[str, float]:
          """
          Process raw classification results to combine similar categories.

          Args:
              raw_results: Raw classification results

          Returns:
              Processed results with combined categories
          """
          processed = {}

          # Group by category and take maximum score
          for prompt, score in raw_results.items():
              # Extract the category from the prompt using more robust matching
              category = None
              for cat in self.defect_categories:
                  # Use word-level matching for better accuracy
                  if any(word.lower() in prompt.lower().split() for word in cat.lower().split()):
                      category = cat
                      break

              # Handle "normal" prompts with improved logic
              if "normal" in prompt.lower() or "no defects" in prompt.lower():
                  category = "Normal"

              if category:
                  if category in processed:
                      processed[category] = max(processed[category], score)
                  else:
                      processed[category] = score

          # Add normalization to make scores more comparable
          if processed:
              total = sum(processed.values())
              if total > 0:  # Avoid division by zero
                  processed = {k: v/total for k, v in processed.items()}

          return processed

    def batch_detect(self, image_paths: List[str], threshold: float = 0.2,
                    top_k: int = 3, enhance_prompts: bool = True) -> List[Dict[str, Any]]:
        """
        Detect PCB defects in multiple images.

        Args:
            image_paths: List of paths to PCB images
            threshold: Confidence threshold for detection
            top_k: Number of top categories to return
            enhance_prompts: Whether to enhance prompts with domain-specific language

        Returns:
            List of detection results for each image
        """
        results = []
        for image_path in image_paths:
            result = self.detect(
                image_path=image_path,
                threshold=threshold,
                top_k=top_k,
                enhance_prompts=enhance_prompts
            )
            results.append(result)

        return results

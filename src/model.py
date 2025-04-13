"""
Vision Language Model for zero-shot PCB defect detection.
"""
import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import List, Dict, Union, Tuple

class PCBDefectVLM:
    """PCB defect detection using Vision-Language Models."""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize the Vision Language Model for PCB defect detection.
        
        Args:
            model_name: Hugging Face model identifier for the VLM
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
    def load_image(self, image_path: str) -> Image.Image:
        """
        Load and prepare an image for inference.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            PIL Image object
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        return Image.open(image_path).convert("RGB")
    
    def classify(self, image: Union[str, Image.Image], categories: List[str]) -> Dict[str, float]:
        """
        Perform zero-shot classification on PCB image.
        
        Args:
            image: Path to image or PIL Image object
            categories: List of defect categories as text prompts
            
        Returns:
            Dictionary of category -> probability mappings
        """
        if isinstance(image, str):
            image = self.load_image(image)
            
        # Prepare text prompts for the model
        text_inputs = self.processor(
            text=categories,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Prepare image for the model
        image_inputs = self.processor(
            images=image,
            return_tensors="pt"
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            image_features = self.model.get_image_features(**image_inputs)
            text_features = self.model.get_text_features(**text_inputs)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            
            # Calculate similarity scores
            logits_per_image = (100.0 * image_features @ text_features.T).squeeze(0)
            probs = logits_per_image.softmax(dim=0)
            
        # Create and return results dictionary
        results = {}
        for category, prob in zip(categories, probs.cpu().numpy()):
            results[category] = float(prob)
            
        return results
    
    def classify_batch(self, images: List[Union[str, Image.Image]], categories: List[str]) -> List[Dict[str, float]]:
        """
        Perform zero-shot classification on a batch of PCB images.
        
        Args:
            images: List of image paths or PIL Image objects
            categories: List of defect categories as text prompts
            
        Returns:
            List of dictionaries mapping categories to probabilities
        """
        # Load images if paths are provided
        loaded_images = []
        for img in images:
            if isinstance(img, str):
                loaded_images.append(self.load_image(img))
            else:
                loaded_images.append(img)
                
        # Prepare text prompts for the model
        text_inputs = self.processor(
            text=categories,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Prepare images for the model
        image_inputs = self.processor(
            images=loaded_images,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            image_features = self.model.get_image_features(**image_inputs)
            text_features = self.model.get_text_features(**text_inputs)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            
            # Calculate similarity scores
            logits_per_image = (100.0 * image_features @ text_features.T)
            probs = logits_per_image.softmax(dim=-1)
            
        # Create and return results
        results = []
        for i, prob_set in enumerate(probs.cpu().numpy()):
            result = {}
            for category, prob in zip(categories, prob_set):
                result[category] = float(prob)
            results.append(result)
            
        return results

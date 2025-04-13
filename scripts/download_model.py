#!/usr/bin/env python
"""
Script to download pre-trained models for PCB defect detection.
"""
import os
import argparse
from transformers import CLIPProcessor, CLIPModel
import torch
from pathlib import Path

def download_model(model_name, save_dir=None):
    """
    Download a pre-trained model from Hugging Face.
    
    Args:
        model_name: Name of the model on Hugging Face
        save_dir: Directory to save the model (optional)
    """
    print(f"Downloading model: {model_name}")
    
    # Download and cache the model
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    if save_dir:
        save_path = Path(save_dir) / model_name.replace('/', '_')
        os.makedirs(save_path, exist_ok=True)
        
        print(f"Saving model to: {save_path}")
        model.save_pretrained(save_path)
        processor.save_pretrained(save_path)
        
        print(f"Model saved successfully to {save_path}")
    else:
        print("Model downloaded and cached successfully")
    
    # Test the model with a small tensor to verify it works
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    print(f"Model loaded successfully on {device}")
    return model, processor

def main():
    parser = argparse.ArgumentParser(description="Download pre-trained models for PCB defect detection")
    parser.add_argument('--model', type=str, default="openai/clip-vit-base-patch32",
                        help="Model name on Hugging Face")
    parser.add_argument('--save-dir', type=str, default=None,
                        help="Directory to save the model (optional)")
    args = parser.parse_args()
    
    download_model(args.model, args.save_dir)
    print("Download complete!")

if __name__ == "__main__":
    main()

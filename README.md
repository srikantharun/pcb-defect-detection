# PCB Defect Detection Implementation Guide

This guide provides detailed instructions for implementing and extending the zero-shot PCB defect detection system.

## Table of Contents

1. [System Overview](#system-overview)
2. [Setting Up Your Environment](#setting-up-your-environment)
3. [Adding Test Images](#adding-test-images)
4. [Custom Defect Categories](#custom-defect-categories)
5. [Tuning Detection Parameters](#tuning-detection-parameters)
6. [Using Alternative Models](#using-alternative-models)
7. [Batch Processing](#batch-processing)
8. [Creating Custom Visualizations](#creating-custom-visualizations)
9. [API Integration](#api-integration)
10. [Troubleshooting](#troubleshooting)

## System Overview

This system uses Vision-Language Models (VLMs) for zero-shot detection of PCB defects. The key components are:

1. **Vision-Language Model**: A pre-trained model (default: CLIP) that understands both images and text
2. **Prompt Engineering**: Natural language descriptions of defect types
3. **Zero-Shot Classification**: Matching images to text descriptions without specific training

## Setting Up Your Environment

### Prerequisites

- Python 3.8+ 
- CUDA-compatible GPU (recommended but not required)
- Git

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pcb-defect-detection.git
   cd pcb-defect-detection
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the pre-trained model:
   ```bash
   python scripts/download_model.py
   ```

## Adding Test Images

### Image Requirements

- Format: JPEG, PNG, or BMP
- Resolution: At least 640x480 pixels recommended
- Content: Clear images of PCBs with good lighting

### Directory Structure

Place your test images in the appropriate directories:

```
data/
└── images/
    ├── normal/        # Non-defective PCB images
    │   ├── pcb_01.jpg
    │   └── pcb_02.jpg
    └── defective/     # Defective PCB images
        ├── solder_bridge_01.jpg
        └── missing_component_01.jpg
```

### Image Naming Conventions

While not required, it's helpful to name defective images with their defect type for easier validation, e.g., `solder_bridge_01.jpg`, `missing_component_02.jpg`.

## Custom Defect Categories

### Defect Category JSON Format

The system uses a JSON file to define defect categories and their associated prompts:

```json
{
    "defects": [
        {
            "category": "Category Name",
            "prompts": [
                "primary description of the defect",
                "alternative description 1",
                "alternative description 2"
            ],
            "severity": "high|medium|low",
            "description": "Detailed explanation of the defect."
        }
    ]
}
```

### Adding New Categories

1. Edit `data/prompts/defect_categories.json`
2. Add a new entry to the `defects` array
3. Provide multiple prompt variations to improve detection accuracy
4. Save the file

### Prompt Engineering Tips

- Use clear, descriptive language
- Include visual characteristics of the defect
- Provide multiple alternative descriptions
- Consider both technical and visual aspects
- Test different prompt variations to find the most effective ones

## Tuning Detection Parameters

### Key Parameters

- `threshold`: Confidence threshold for detection (default: 0.2)
- `top_k`: Number of top categories to return (default: 3)
- `enhance_prompts`: Whether to enhance prompts with domain knowledge (default: True)

### Tuning Process

1. Start with default parameters
2. If getting too many false positives, increase the threshold
3. If missing defects, decrease the threshold
4. Adjust `top_k` based on how many categories you want to consider
5. Experiment with `enhance_prompts=False` if domain-specific prompts are causing issues

Example:
```python
results = detector.detect(
    image_path="path/to/image.jpg",
    threshold=0.3,  # Increased from default 0.2
    top_k=2,        # Only interested in the top 2 defects
    enhance_prompts=True
)
```

## Using Alternative Models

### Available Models

The system supports any Vision-Language Model from Hugging Face that's compatible with the CLIP architecture. Some options include:

- `openai/clip-vit-base-patch32` (default)
- `openai/clip-vit-large-patch14` (higher accuracy, slower)
- `facebook/flava-full` (alternative architecture)
- `google/siglip-base-patch16-224` (alternative architecture)

### Changing Models

```python
# Initialize with a different model
detector = PCBDefectDetector(model_name="openai/clip-vit-large-patch14")
```

### Model Selection Considerations

- Larger models generally provide better accuracy but are slower
- Some models may be better at certain types of visual defects
- Consider your computational resources when selecting a model

## Batch Processing

For processing multiple images, use the `batch_detect` method:

```python
# Get all image paths
import glob
image_paths = glob.glob("data/images/defective/*.jpg")

# Batch detection
results = detector.batch_detect(
    image_paths=image_paths,
    threshold=0.2,
    top_k=3,
    enhance_prompts=True
)

# Process results
for image_path, result in zip(image_paths, results):
    print(f"Results for {image_path}:")
    print(f"Is defective: {result['is_defective']}")
    for category, score in result['defects'].items():
        print(f"  - {category}: {score:.4f}")
```

## Creating Custom Visualizations

The `visualization.py` module provides several visualization functions:

### Single Image Visualization

```python
from src.visualization import visualize_detection

visualize_detection(
    image_path="data/images/defective/solder_bridge_01.jpg",
    results=results,
    output_path="results/detection_result.png"
)
```

### Comparison Grid

```python
from src.visualization import create_comparison_grid

create_comparison_grid(
    image_paths=image_paths,
    results=results,
    output_path="results/comparison_grid.png",
    grid_size=(3, 3)  # 3x3 grid
)
```

### Defect Distribution

```python
from src.visualization import visualize_defect_distribution

visualize_defect_distribution(
    results=results,
    output_path="results/defect_distribution.png"
)
```

### Customizing Visualizations

You can modify the visualization functions in `src/visualization.py` to create custom visualizations tailored to your needs.

## API Integration

You can integrate this system into a larger application or service:

### Web Service Example

```python
from flask import Flask, request, jsonify
from src.zero_shot import PCBDefectDetector

app = Flask(__name__)
detector = PCBDefectDetector()
detector.load_defect_categories("data/prompts/defect_categories.json")

@app.route('/detect', methods=['POST'])
def detect_defect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
        
    image = request.files['image']
    image_path = f"uploads/{image.filename}"
    image.save(image_path)
    
    results = detector.detect(
        image_path=image_path,
        threshold=float(request.form.get('threshold', 0.2)),
        top_k=int(request.form.get('top_k', 3))
    )
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
```

## Troubleshooting

### Common Issues and Solutions

#### Model Loading Errors

**Issue**: Error when loading the model (`OSError` or `ValueError`)
**Solution**: 
- Check internet connection
- Ensure you have enough disk space
- Try downloading the model manually using `transformers.AutoModel.from_pretrained`

#### Poor Detection Results

**Issue**: Incorrect or low-confidence detections
**Solution**:
- Try different prompt formulations
- Adjust the threshold parameter
- Use a larger model
- Improve image quality/lighting

#### Memory Errors

**Issue**: CUDA out of memory or similar errors
**Solution**:
- Use a smaller model
- Process images in smaller batches
- Reduce image resolution
- Use CPU mode if GPU memory is limited

#### Slow Performance

**Issue**: Processing takes too long
**Solution**:
- Use a smaller model
- Process images in batches
- Consider model quantization for faster inference
- Ensure you're using GPU acceleration if available

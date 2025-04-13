"""
Visualization tools for PCB defect detection results.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Any, Tuple, Union, Optional

def visualize_detection(image_path: str, results: Dict[str, Any], 
                        output_path: Optional[str] = None,
                        show: bool = True) -> None:
    """
    Visualize defect detection results with confidence scores.
    
    Args:
        image_path: Path to the PCB image
        results: Detection results from PCBDefectDetector
        output_path: Optional path to save the visualization
        show: Whether to display the plot
    """
    # Load image
    img = Image.open(image_path)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot image
    ax1.imshow(np.array(img))
    ax1.set_title("PCB Image")
    ax1.axis('off')
    
    # Plot defect scores
    defects = results['defects']
    categories = list(defects.keys())
    scores = list(defects.values())
    
    # Sort by score in descending order
    sorted_indices = np.argsort(scores)[::-1]
    categories = [categories[i] for i in sorted_indices]
    scores = [scores[i] for i in sorted_indices]
    
    # Set colors based on defect status
    colors = ['red' if category.lower() != "normal" else 'green' for category in categories]
    
    # Plot horizontal bar chart
    y_pos = np.arange(len(categories))
    bars = ax2.barh(y_pos, scores, color=colors, alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(categories)
    ax2.set_xlim(0, 1.0)
    ax2.set_xlabel('Confidence Score')
    
    # Determine overall status
    if results['is_defective']:
        status_text = "DEFECTIVE"
        status_color = "red"
    else:
        status_text = "NORMAL"
        status_color = "green"
        
    ax2.set_title(f"Detection Results: {status_text}", color=status_color, fontweight='bold')
    
    # Add score values
    for bar, score in zip(bars, scores):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{score:.2f}', va='center')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def create_comparison_grid(image_paths: List[str], results: List[Dict[str, Any]],
                          output_path: Optional[str] = None,
                          grid_size: Optional[Tuple[int, int]] = None,
                          show: bool = True) -> None:
    """
    Create a grid of PCB images with their detection results.
    
    Args:
        image_paths: List of paths to PCB images
        results: List of detection results
        output_path: Optional path to save the visualization
        grid_size: Optional tuple of (rows, cols) for the grid layout
        show: Whether to display the plot
    """
    n_images = len(image_paths)
    
    if grid_size is None:
        # Calculate grid size based on number of images
        cols = min(4, n_images)
        rows = (n_images + cols - 1) // cols
    else:
        rows, cols = grid_size
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    
    # Handle single row or column case
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot each image with its top defect
    for i, (image_path, result) in enumerate(zip(image_paths, results)):
        if i >= rows * cols:
            break
            
        row, col = i // cols, i % cols
        ax = axes[row, col]
        
        # Load and display image
        img = Image.open(image_path)
        ax.imshow(np.array(img))
        
        # Get top defect
        defects = result['defects']
        if defects:
            top_defect = list(defects.keys())[0]
            top_score = list(defects.values())[0]
            
            # Set color based on defect status
            color = 'red' if result['is_defective'] else 'green'
            
            # Set title with top defect and score
            ax.set_title(f"{top_defect}\n({top_score:.2f})", color=color)
        else:
            ax.set_title("No defects detected")
            
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(n_images, rows * cols):
        row, col = i // cols, i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Comparison grid saved to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def visualize_defect_distribution(results: List[Dict[str, Any]], 
                                 output_path: Optional[str] = None,
                                 show: bool = True) -> None:
    """
    Visualize the distribution of defect types across multiple images.
    
    Args:
        results: List of detection results
        output_path: Optional path to save the visualization
        show: Whether to display the plot
    """
    # Count defect occurrences
    defect_counts = {}
    
    for result in results:
        if result['is_defective']:
            # Get top defect for each image
            top_defect = list(result['defects'].keys())[0]
            if top_defect in defect_counts:
                defect_counts[top_defect] += 1
            else:
                defect_counts[top_defect] = 1
    
    # Sort defects by frequency
    sorted_defects = sorted(defect_counts.items(), key=lambda x: x[1], reverse=True)
    categories = [item[0] for item in sorted_defects]
    counts = [item[1] for item in sorted_defects]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bar chart
    bars = ax.bar(categories, counts, color='crimson', alpha=0.7)
    
    # Add count labels on top of bars
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
               str(count), ha='center')
    
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Defect Types')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Defect distribution visualization saved to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def create_annotated_image(image_path: str, results: Dict[str, Any],
                          output_path: Optional[str] = None) -> Image.Image:
    """
    Create an annotated version of the PCB image with defect information.
    
    Args:
        image_path: Path to the PCB image
        results: Detection results
        output_path: Optional path to save the annotated image
        
    Returns:
        Annotated PIL Image
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, use default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        small_font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Draw a header at the top
    width, height = img.size
    header_height = 40
    
    # Create header background
    if results['is_defective']:
        header_color = (220, 50, 50)  # Red for defective
        status_text = "DEFECTIVE"
    else:
        header_color = (50, 180, 50)  # Green for normal
        status_text = "NORMAL"
        
    draw.rectangle([(0, 0), (width, header_height)], fill=header_color)
    
    # Draw status text
    text_width = len(status_text) * 12  # Approximate width
    draw.text(((width - text_width) // 2, 10), status_text, fill='white', font=font)
    
    # Draw defect information
    y_pos = header_height + 10
    for category, score in results['defects'].items():
        # Skip if it's normal and there are other defects
        if category.lower() == "normal" and len(results['defects']) > 1:
            continue
            
        text = f"{category}: {score:.2f}"
        draw.text((10, y_pos), text, fill='black', font=small_font)
        y_pos += 25
    
    # Save if output path is provided
    if output_path:
        img.save(output_path)
        print(f"Annotated image saved to {output_path}")
    
    return img

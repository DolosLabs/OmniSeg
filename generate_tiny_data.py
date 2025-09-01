#!/usr/bin/env python3
"""
Generate a tiny synthetic dataset for quick testing and development.
This creates a small COCO-format dataset with simple geometric shapes
instead of requiring external downloads.
"""

import os
import random
import json
import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils

# -------------------------
# Config
# -------------------------
BASE_DIR = "SSL_Instance_Segmentation/coco2017"
IMAGES_DIR = {
    "train": os.path.join(BASE_DIR, "train2017"),
    "val": os.path.join(BASE_DIR, "val2017"),
    "test": os.path.join(BASE_DIR, "test2017")
}
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "annotations")

CANVAS_SIZE = (64, 64)
SHAPES_PER_IMAGE = (2, 5)  # min-max
N_TRAIN = 1000
N_VAL = 200
N_TEST = 200
MIN_MASK_PIXELS = 10  # Skip tiny masks

# Shape categories
CATEGORIES = [
    {"id": 1, "name": "circle", "supercategory": "shape"},
    {"id": 2, "name": "rectangle", "supercategory": "shape"},
    {"id": 3, "name": "triangle", "supercategory": "shape"}
]

random.seed(42)
np.random.seed(42)

# Create directories
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
for d in IMAGES_DIR.values():
    os.makedirs(d, exist_ok=True)

print("Generating tiny synthetic dataset for quick training...")

# -------------------------
# Helper functions
# -------------------------
def get_bbox(mask, x_offset=0, y_offset=0):
    """Get bounding box from binary mask"""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    
    return [x_min + x_offset, y_min + y_offset, x_max - x_min + 1, y_max - y_min + 1]

def get_rle(mask):
    """Convert binary mask to RLE format"""
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle

def draw_circle(draw, center, radius, color):
    """Draw a filled circle"""
    x, y = center
    draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)
    return True

def draw_rectangle(draw, center, width, height, color):
    """Draw a filled rectangle"""
    x, y = center
    draw.rectangle([x-width//2, y-height//2, x+width//2, y+height//2], fill=color)
    return True

def draw_triangle(draw, center, size, color):
    """Draw a filled triangle"""
    x, y = center
    points = [
        (x, y - size),
        (x - size, y + size//2),
        (x + size, y + size//2)
    ]
    draw.polygon(points, fill=color)
    return True

def generate_shapes_image():
    """Generate a single image with random shapes"""
    # Create blank canvas
    canvas = Image.new('RGB', CANVAS_SIZE, color=(0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    
    # Number of shapes to draw
    num_shapes = random.randint(*SHAPES_PER_IMAGE)
    
    shapes_info = []
    
    for shape_id in range(num_shapes):
        # Choose random shape type
        shape_type = random.choice(['circle', 'rectangle', 'triangle'])
        category_id = next(cat['id'] for cat in CATEGORIES if cat['name'] == shape_type)
        
        # Random color (grayscale for simplicity)
        color_val = random.randint(50, 255)
        color = (color_val, color_val, color_val)
        
        # Random position (avoid edges)
        margin = 30
        center_x = random.randint(margin, CANVAS_SIZE[0] - margin)
        center_y = random.randint(margin, CANVAS_SIZE[1] - margin)
        center = (center_x, center_y)
        
        # Create mask for this shape
        shape_canvas = Image.new('L', CANVAS_SIZE, color=0)
        shape_draw = ImageDraw.Draw(shape_canvas)
        
        if shape_type == 'circle':
            radius = random.randint(10, 25)
            draw_circle(draw, center, radius, color)
            draw_circle(shape_draw, center, radius, 255)
        elif shape_type == 'rectangle':
            width = random.randint(20, 50)
            height = random.randint(20, 50)
            draw_rectangle(draw, center, width, height, color)
            draw_rectangle(shape_draw, center, width, height, 255)
        elif shape_type == 'triangle':
            size = random.randint(15, 30)
            draw_triangle(draw, center, size, color)
            draw_triangle(shape_draw, center, size, 255)
        
        # Convert to numpy for processing
        mask_array = np.array(shape_canvas) > 0
        
        # Skip if too small
        if np.sum(mask_array) < MIN_MASK_PIXELS:
            continue
        
        # Get bounding box and RLE
        bbox = get_bbox(mask_array)
        rle = get_rle(mask_array)
        area = int(np.sum(mask_array))
        
        shapes_info.append({
            'category_id': category_id,
            'bbox': bbox,
            'segmentation': rle,
            'area': area
        })
    
    return canvas, shapes_info

# -------------------------
# Generate COCO JSON + images
# -------------------------
splits = {"train": N_TRAIN, "val": N_VAL, "test": N_TEST}

for split_name, num_images in splits.items():
    print(f"Generating {split_name} split: {num_images} images...")
    
    coco = {
        "info": {"description": "Tiny synthetic dataset for OmniSeg testing"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": CATEGORIES
    }
    
    annotation_id = 1
    
    for img_idx in range(num_images):
        image_id = img_idx + 1
        filename = f"{image_id:012d}.jpg"
        image_path = os.path.join(IMAGES_DIR[split_name], filename)
        
        # Generate image with shapes
        canvas, shapes_info = generate_shapes_image()
        
        # Save image
        canvas.save(image_path, 'JPEG')
        
        # Add image info
        coco["images"].append({
            "id": image_id,
            "width": CANVAS_SIZE[0],
            "height": CANVAS_SIZE[1],
            "file_name": filename
        })
        
        # Add annotations for each shape
        for shape_info in shapes_info:
            coco["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": shape_info['category_id'],
                "segmentation": shape_info['segmentation'],
                "area": shape_info['area'],
                "bbox": shape_info['bbox'],
                "iscrowd": 0
            })
            annotation_id += 1
    
    # Save annotation file
    annotation_file = os.path.join(ANNOTATIONS_DIR, f"instances_{split_name}2017.json")
    with open(annotation_file, 'w') as f:
        json.dump(coco, f)
    
    print(f"  Saved {len(coco['images'])} images and {len(coco['annotations'])} annotations")

print(f"\nTiny dataset generated successfully!")
print(f"Dataset location: {BASE_DIR}")
print(f"Train: {N_TRAIN} images, Val: {N_VAL} images, Test: {N_TEST} images")
print(f"Categories: {len(CATEGORIES)} (circle, rectangle, triangle)")
print(f"\nYou can now train with: python train.py --backbone resnet --head maskrcnn --fast_dev_run")

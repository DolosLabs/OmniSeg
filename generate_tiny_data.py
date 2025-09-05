#!/usr/bin/env python3
"""
Synthetic COCO dataset generator optimized for LW-DETR with DINOv3 backbone.
- Image size: 128x128 (16x16 patches → 8x8 feature map)
- Object size: 16-32 pixels
- Non-overlapping shapes
- COCO RLE masks
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
IMAGES_DIR = {split: os.path.join(BASE_DIR, f"{split}2017") for split in ["train", "val", "test"]}
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "annotations")

CANVAS_SIZE = (128, 128)
SHAPES_PER_IMAGE = (1, 5)  # min-max shapes
N_TRAIN = 4096
N_VAL = 400
N_TEST = 400
MIN_MASK_PIXELS = 20  # skip tiny masks

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

# -------------------------
# Helper functions
# -------------------------
def get_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0: return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1)]

def get_rle(mask):
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle

def draw_circle(draw, center, radius, color):
    x, y = center
    draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)

def draw_rectangle(draw, center, width, height, color):
    x, y = center
    draw.rectangle([x-width//2, y-height//2, x+width//2, y+height//2], fill=color)

def draw_triangle(draw, center, size, color):
    x, y = center
    points = [(x, y-size), (x-size, y+size//2), (x+size, y+size//2)]
    draw.polygon(points, fill=color)

def sample_non_overlapping_positions(num_shapes, grid_size=(4,4), margin=8):
    """Place shapes in distinct grid cells to avoid overlap"""
    rows, cols = grid_size
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    random.shuffle(cells)
    chosen_cells = cells[:num_shapes]

    positions = []
    for r, c in chosen_cells:
        cell_w, cell_h = CANVAS_SIZE[0]//cols, CANVAS_SIZE[1]//rows
        x_min, x_max = c*cell_w + margin, (c+1)*cell_w - margin
        y_min, y_max = r*cell_h + margin, (r+1)*cell_h - margin
        center_x = random.randint(x_min, x_max)
        center_y = random.randint(y_min, y_max)
        positions.append((center_x, center_y))
    return positions

def generate_shapes_image():
    canvas = Image.new('RGB', CANVAS_SIZE, (0,0,0))
    draw = ImageDraw.Draw(canvas)

    num_shapes = random.randint(*SHAPES_PER_IMAGE)
    positions = sample_non_overlapping_positions(num_shapes)
    shapes_info = []

    for center in positions:
        shape_type = random.choice(['circle','rectangle','triangle'])
        category_id = next(cat['id'] for cat in CATEGORIES if cat['name']==shape_type)
        color = tuple(random.randint(50,255) for _ in range(3))

        shape_canvas = Image.new('L', CANVAS_SIZE, 0)
        shape_draw = ImageDraw.Draw(shape_canvas)

        if shape_type=='circle':
            radius = random.randint(16,32)
            draw_circle(draw, center, radius, color)
            draw_circle(shape_draw, center, radius, 255)
        elif shape_type=='rectangle':
            width,height = random.randint(16,32), random.randint(16,32)
            draw_rectangle(draw, center, width, height, color)
            draw_rectangle(shape_draw, center, width, height, 255)
        else: # triangle
            size = random.randint(16,32)
            draw_triangle(draw, center, size, color)
            draw_triangle(shape_draw, center, size, 255)

        mask_array = np.array(shape_canvas) > 0
        if np.sum(mask_array) < MIN_MASK_PIXELS:
            continue

        shapes_info.append({
            'category_id': category_id,
            'bbox': get_bbox(mask_array),
            'segmentation': get_rle(mask_array),
            'area': int(np.sum(mask_array))
        })

    return canvas, shapes_info

# -------------------------
# Generate dataset
# -------------------------
splits = {"train": N_TRAIN, "val": N_VAL, "test": N_TEST}

for split_name,num_images in splits.items():
    coco = {"info":{"description":"LW-DETR tiny synthetic dataset for DINOv3"},
            "licenses":[],"images":[],"annotations":[],"categories":CATEGORIES}
    ann_id = 1

    for img_idx in range(num_images):
        image_id = img_idx+1
        filename = f"{image_id:012d}.jpg"
        image_path = os.path.join(IMAGES_DIR[split_name], filename)

        canvas, shapes_info = generate_shapes_image()
        canvas.save(image_path,'JPEG')

        coco["images"].append({"id":image_id,"width":CANVAS_SIZE[0],"height":CANVAS_SIZE[1],"file_name":filename})

        for s in shapes_info:
            coco["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": s['category_id'],
                "segmentation": s['segmentation'],
                "area": s['area'],
                "bbox": s['bbox'],
                "iscrowd": 0
            })
            ann_id += 1

    with open(os.path.join(ANNOTATIONS_DIR,f"instances_{split_name}2017.json"),'w') as f:
        json.dump(coco,f)

print("DINOv3-optimized LW-DETR tiny COCO dataset generated successfully!")

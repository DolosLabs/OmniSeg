import os
import random
import json
import numpy as np
from PIL import Image
import torch
import torchvision
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
DIGITS_PER_IMAGE = (2, 5)  # min-max
N_TRAIN = 2000
N_VAL = 400
N_TEST = 400
MIN_MASK_PIXELS = 5  # Skip tiny masks

random.seed(42)
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
for d in IMAGES_DIR.values():
    os.makedirs(d, exist_ok=True)

# -------------------------
# Helper functions
# -------------------------
def get_bbox(mask, x_offset, y_offset):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if rows.any() and cols.any():
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        return [int(x_min + x_offset), int(y_min + y_offset),
                int(x_max - x_min + 1), int(y_max - y_min + 1)]
    else:
        return [x_offset, y_offset, mask.shape[1], mask.shape[0]]

def get_rle(mask):
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def main():
    # -------------------------
    # Download MNIST
    # -------------------------
    mnist_train = torchvision.datasets.MNIST(root="./data", train=True, download=True)
    mnist_test = torchvision.datasets.MNIST(root="./data", train=False, download=True)
    all_data = list(mnist_train) + list(mnist_test)
    random.shuffle(all_data)

    splits = {
        "train": all_data[:N_TRAIN],
        "val": all_data[N_TRAIN:N_TRAIN+N_VAL],
        "test": all_data[N_TRAIN+N_VAL:N_TRAIN+N_VAL+N_TEST]
    }

    # -------------------------
    # Generate COCO JSON + images
    # -------------------------
    category_ids = {str(i): i+1 for i in range(10)}

for split_name, data in splits.items():
    coco = {
        "info": {"description": "MNIST Multi-digit dataset"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": i+1, "name": str(i)} for i in range(10)]
    }
    ann_id = 1
    img_idx = 0

    for _ in range(len(data)):
        canvas = Image.new("L", CANVAS_SIZE, 0)
        chosen_digits = random.choices(data, k=random.randint(*DIGITS_PER_IMAGE))
        instance_masks = []
        instance_labels = []

        for digit_img, label in chosen_digits:
            scale = random.randint(20,28)
            digit_resized = digit_img.resize((scale, scale))
            digit_arr = np.array(digit_resized)

            x_offset = random.randint(0, CANVAS_SIZE[0]-scale)
            y_offset = random.randint(0, CANVAS_SIZE[1]-scale)

            mask_full = np.zeros(CANVAS_SIZE, dtype=np.uint8)
            digit_mask = (digit_arr > 0).astype(np.uint8)
            if digit_mask.sum() < MIN_MASK_PIXELS:
                continue  # skip tiny mask
            mask_full[y_offset:y_offset+scale, x_offset:x_offset+scale] = digit_mask
            instance_masks.append(mask_full)
            instance_labels.append(category_ids[str(label)])

            bbox = get_bbox(digit_arr, x_offset, y_offset)
            segmentation = get_rle(mask_full)

            coco["annotations"].append({
                "id": int(ann_id),
                "image_id": int(img_idx),
                "category_id": int(category_ids[str(label)]),
                "bbox": [int(x) for x in bbox],
                "area": int(mask_full.sum()),
                "iscrowd": 0,
                "segmentation": segmentation
            })
            ann_id += 1
            canvas.paste(digit_resized, (x_offset, y_offset))

        # skip empty images
        if len(instance_masks) == 0:
            continue

        file_name = f"{img_idx:05d}.jpg"
        canvas.save(os.path.join(IMAGES_DIR[split_name], file_name))
        coco["images"].append({
            "id": int(img_idx),
            "file_name": file_name,
            "height": CANVAS_SIZE[1],
            "width": CANVAS_SIZE[0]
        })
        img_idx += 1

    ann_file_path = os.path.join(ANNOTATIONS_DIR, f"instances_{split_name}2017.json")
    with open(ann_file_path, "w") as f:
        json.dump(coco, f)
    print(f"{split_name.upper()}: {len(coco['images'])} images, {len(coco['annotations'])} annotations saved")

    print("MNIST -> Multi-digit COCO conversion complete!")


if __name__ == "__main__":
    main()


# -------------------------
# PyTorch Dataset
# -------------------------
class MNISTCOCODataset(torch.utils.data.Dataset):
    def __init__(self, split="train", transforms=None):
        self.images_dir = IMAGES_DIR[split]
        self.coco_json = os.path.join(ANNOTATIONS_DIR, f"instances_{split}2017.json")
        with open(self.coco_json) as f:
            self.coco = json.load(f)

        self.img_id_to_anns = {}
        for ann in self.coco["annotations"]:
            self.img_id_to_anns.setdefault(ann["image_id"], []).append(ann)

        self.ids = [img["id"] for img in self.coco["images"]]
        self.transforms = transforms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = next(x for x in self.coco["images"] if x["id"]==img_id)
        img_path = os.path.join(self.images_dir, img_info["file_name"])
        img = Image.open(img_path).convert("L")
        img = (torch.tensor(np.array(img), dtype=torch.float32)/255.0 - 0.5) / 0.5  # normalize [-1,1]

        anns = self.img_id_to_anns.get(img_id, [])
        masks, labels = [], []
        for ann in anns:
            mask = mask_utils.decode(ann["segmentation"])
            masks.append(mask)
            labels.append(ann["category_id"])

        if masks:
            masks_tensor = torch.tensor(np.stack(masks), dtype=torch.uint8)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            masks_tensor = torch.zeros((0, *CANVAS_SIZE), dtype=torch.uint8)
            labels_tensor = torch.tensor([], dtype=torch.int64)

        target = {"image_id": img_id, "masks": masks_tensor, "labels": labels_tensor}
        if self.transforms:
            img, target = self.transforms(img, target)
        return img, target

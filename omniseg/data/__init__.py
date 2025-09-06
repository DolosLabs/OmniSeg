"""Data utilities and dataset classes for OmniSeg."""

import os
import random
import zipfile
import urllib.request
from typing import List, Dict, Tuple, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.v2 as T
from torchvision import tv_tensors
from PIL import Image
from pycocotools.coco import COCO
import tqdm
import pytorch_lightning as pl
from pytorch_lightning.utilities.combined_loader import CombinedLoader

def download_coco2017(root_dir=".", splits=['train', 'val', 'test']):
    """Download COCO 2017 dataset."""
    base_dir = os.path.join(root_dir, 'coco2017')
    annotations_dir = os.path.join(base_dir, 'annotations')
    
    class TqdmUpTo(tqdm.tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None: 
                self.total = tsize
            self.update(b * bsize - self.n)
    
    def download_url(url, output_path, desc):
        with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=desc) as t:
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
    
    if not os.path.exists(annotations_dir):
        print("Downloading COCO 2017 annotations...")
        os.makedirs(base_dir, exist_ok=True)
        url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        zip_path = os.path.join(base_dir, 'annotations.zip')
        download_url(url, zip_path, "annotations_trainval2017.zip")
        print("Extracting annotations...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(base_dir)
        os.remove(zip_path)
    else:
        print("COCO 2017 annotations already exist.")
    
    for split in splits:
        images_dir = os.path.join(base_dir, f'{split}2017')
        if not os.path.exists(images_dir):
            print(f"Downloading COCO {split}2017 images...")
            url = f"http://images.cocodataset.org/zips/{split}2017.zip"
            zip_path = os.path.join(base_dir, f'{split}2017.zip')
            download_url(url, zip_path, f"{split}2017.zip")
            print(f"Extracting {split}2017 images...")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(base_dir)
            os.remove(zip_path)
        else:
            print(f"COCO {split}2017 images already exist.")
    
    return base_dir


def get_transforms(augment=False, image_size=224):
    """Get image transforms for training/validation."""
    transforms = []
    if augment:
        transforms.append(T.RandomHorizontalFlip(p=0.5))
        transforms.append(T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1))

    transforms.append(T.Resize((image_size, image_size), antialias=True))
    
    # --- ✅ CORRECTED CODE ---
    # 1. Convert PIL Image to a v2 tensor-like Image object
    transforms.append(T.ToImage()) 
    # 2. Convert dtype to float and scale values from [0, 255] to [0.0, 1.0]
    transforms.append(T.ToDtype(torch.float32, scale=True))
    # -------------------------

    # Now, Normalize receives a tensor as expected.
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    
    return T.Compose(transforms)

# --- 2. Fixed and Integrated Dataset Class ---
# This version is corrected to use the transformation pipeline properly.

class SemiCOCODataset(Dataset):
    """
    Semi-supervised COCO dataset that correctly applies v2 transforms.
    """
    
    # MODIFIED: Added 'transform=None' to accept the pipeline
    def __init__(self, images_dir, ann_file=None, is_unlabeled=False, num_images=-1, transform=None):
        self.images_dir = images_dir
        self.is_unlabeled = is_unlabeled
        self.transform = transform
        
        # --- ✅ FIX: Initialize attributes here ---
        # This ensures they always exist on the object instance.
        self.coco = None
        self.cat2label = {}
        self.label2cat = {}
        # ----------------------------------------
        
        if self.is_unlabeled:
            self.img_files = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.jpg')]
            if num_images > 0 and len(self.img_files) > num_images:
                random.seed(42)
                self.img_files = random.sample(self.img_files, num_images)
                print(f"--- Using a random subset of {len(self.img_files)} unlabeled images. ---")
        else:
            # Now, populate the attributes with real data
            self.coco = COCO(ann_file)
            all_img_ids = sorted(self.coco.getImgIds())
            cat_ids = self.coco.getCatIds()
            self.cat2label = {cat_id: i for i, cat_id in enumerate(cat_ids)}
            self.label2cat = {v: k for k, v in self.cat2label.items()} # Now this correctly fills the empty dict
            
            self.img_ids = [img_id for img_id in all_img_ids if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=False)) > 0]
            if num_images > 0 and len(self.img_ids) > num_images:
                random.seed(42)
                self.img_ids = random.sample(self.img_ids, num_images)
                print(f"--- Using a random subset of {len(self.img_ids)} labeled images. ---")

    def __len__(self):
        return len(self.img_files) if self.is_unlabeled else len(self.img_ids)

    # In your SemiCOCODataset class
    def __getitem__(self, idx):
        # --- Handle Unlabeled Data ---
        if self.is_unlabeled:
            image = Image.open(self.img_files[idx]).convert("RGB")
            if self.transform:
                image, _ = self.transform(image, {})
            return image, {}
    
        # --- Handle Labeled Data ---
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        image = Image.open(os.path.join(self.images_dir, img_info['file_name'])).convert("RGB")
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)
    
        masks, labels = [], []
        for ann in anns:
            mask = self.coco.annToMask(ann)
            if mask.sum() > 0:
                masks.append(mask)
                labels.append(self.cat2label[ann['category_id']])
    
        if not masks:
            return self.__getitem__((idx + 1) % len(self))

        target = {
            "image_id": torch.tensor([img_id]),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks": tv_tensors.Mask(np.array(masks, dtype=np.uint8))
        }
    
        if self.transform:
            image, target = self.transform(image, target)
            
        return image, target



class COCODataModule(pl.LightningDataModule):
    """COCO data module for PyTorch Lightning."""
    def __init__(self, project_dir, batch_size=32, num_workers=2, image_size=224,
                 num_labeled_images=-1, num_unlabeled_images=-1):
        super().__init__()
        self.project_dir = project_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.num_labeled_images = num_labeled_images
        self.num_unlabeled_images = num_unlabeled_images
        
        # Note: self.train_aug and self.val_aug are now defined in setup()

    def prepare_data(self):
        """Downloads data if needed. Ran once per node."""
        download_coco2017(root_dir=self.project_dir, splits=['train', 'val', 'test'])
    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to handle batches of images and targets.
        Images are stacked into a single tensor, while targets (dictionaries)
        are gathered into a list.
        """
        # Separate images and targets from the batch
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]
    
        # Stack images into a single tensor (B, C, H, W)
        images = torch.stack(images, 0)
    
        # Return the batched images and the list of targets
        return images, targets
    def setup(self, stage=None):
        """Sets up datasets. Ran on each GPU."""
        # MODIFIED: Define transforms here as a best practice
        self.train_aug = get_transforms(augment=True, image_size=self.image_size)
        self.val_aug = get_transforms(augment=False, image_size=self.image_size)
        
        base_dir = os.path.join(self.project_dir, 'coco2017')
        train_images_dir = os.path.join(base_dir, 'train2017')
        val_images_dir = os.path.join(base_dir, 'val2017')
        test_images_dir = os.path.join(base_dir, 'test2017') # Used for unlabeled data
        train_ann_file = os.path.join(base_dir, 'annotations', 'instances_train2017.json')
        val_ann_file = os.path.join(base_dir, 'annotations', 'instances_val2017.json')

        # MODIFIED: Pass the correct transform to each dataset instance
        self.labeled_ds = SemiCOCODataset(
            train_images_dir, train_ann_file, 
            num_images=self.num_labeled_images, 
            transform=self.train_aug
        )
        self.unlabeled_ds = SemiCOCODataset(
            test_images_dir, is_unlabeled=True, 
            num_images=self.num_unlabeled_images, 
            transform=self.train_aug
        )
        self.val_ds = SemiCOCODataset(
            val_images_dir, val_ann_file, 
            transform=self.val_aug
        )
        
        # These assignments remain the same
        self.cat2label = self.labeled_ds.cat2label
        self.label2cat = self.labeled_ds.label2cat
        self.coco_gt_val = self.val_ds.coco

    def train_dataloader(self):
        labeled_loader = DataLoader(
            self.labeled_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True,
            collate_fn=self.collate_fn 
        )
        unlabeled_loader = DataLoader(
            self.unlabeled_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True,
            collate_fn=self.collate_fn 
        )
        return CombinedLoader({"labeled": labeled_loader, "unlabeled": unlabeled_loader}, mode="max_size_cycle")

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=1, shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn 
        )
        


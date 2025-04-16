import os
import json
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image

class DAVISDataset(Dataset):
    def __init__(self, davis_root, split='train', transform=None, max_sequences=None, max_frames_per_seq=None):
        self.davis_root = davis_root
        self.transform = transform
        self.split = split
        self.max_sequences = max_sequences
        self.max_frames_per_seq = max_frames_per_seq
        
        # Check if DAVIS directory exists
        if not os.path.exists(davis_root):
            raise RuntimeError(f"DAVIS dataset directory not found: {davis_root}")
        
        # Paths for images and annotations
        self.img_path = os.path.join(davis_root, 'JPEGImages', '480p')
        self.ann_path = os.path.join(davis_root, 'Annotations', '480p')
        
        # Check if required directories exist
        if not os.path.exists(self.img_path):
            raise RuntimeError(f"JPEGImages/480p directory not found in {davis_root}")
        if not os.path.exists(self.ann_path):
            raise RuntimeError(f"Annotations/480p directory not found in {davis_root}")
        
        # Get all video sequences
        self.sequences = sorted([d for d in os.listdir(self.img_path) 
                               if os.path.isdir(os.path.join(self.img_path, d))])
        
        if not self.sequences:
            raise RuntimeError(f"No video sequences found in {self.img_path}")
            
        if self.max_sequences is not None:
            self.sequences = self.sequences[:max_sequences]
        
        # Create class mapping
        self.class_mapping = {}
        self.images = []
        self.annotations = []
        
        # Build dataset
        self._build_dataset()
        
        if len(self.images) == 0:
            raise RuntimeError("No valid frames found in the dataset")
            
        print(f"Dataset loaded with {len(self.sequences)} sequences and {len(self.images)} total frames")
    
    def _build_dataset(self):
        class_id = 1  # Start from 1 as 0 is background
        
        for seq in self.sequences:
            seq_img_path = os.path.join(self.img_path, seq)
            seq_ann_path = os.path.join(self.ann_path, seq)
            
            if not os.path.isdir(seq_img_path):
                continue
                
            # Get all frames
            frames = sorted([f for f in os.listdir(seq_img_path) 
                           if f.endswith('.jpg') and 
                           os.path.exists(os.path.join(seq_ann_path, f.replace('.jpg', '.png')))])
            
            if not frames:
                print(f"Warning: No valid frames found in sequence {seq}")
                continue
                
            if self.max_frames_per_seq is not None:
                frames = frames[:self.max_frames_per_seq]
            
            valid_frames = []
            for frame in frames:
                img_file = os.path.join(seq_img_path, frame)
                ann_file = os.path.join(seq_ann_path, frame.replace('.jpg', '.png'))
                
                # Verify files are readable
                try:
                    with Image.open(img_file) as img:
                        img.verify()
                    with Image.open(ann_file) as mask:
                        mask.verify()
                    valid_frames.append((img_file, ann_file))
                except Exception as e:
                    print(f"Warning: Error reading files for frame {frame} in sequence {seq}: {e}")
                    continue
            
            if valid_frames:
                self.images.extend([vf[0] for vf in valid_frames])
                self.annotations.extend([vf[1] for vf in valid_frames])
                self.class_mapping[seq] = class_id
                class_id += 1
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        try:
            # Load image
            img_path = self.images[idx]
            img = Image.open(img_path).convert("RGB")
            
            # Load mask
            mask_path = self.annotations[idx]
            mask = Image.open(mask_path)
            
            # Convert to numpy array
            mask = np.array(mask)
            
            # Get sequence name from path
            seq = img_path.split(os.sep)[-2]
            class_id = self.class_mapping[seq]
            
            # Get bounding boxes from mask
            obj_ids = np.unique(mask)
            obj_ids = obj_ids[1:]  # Remove background
            
            if len(obj_ids) == 0:
                # If no objects in mask, create a dummy box
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)
                masks = torch.zeros((0, mask.shape[0], mask.shape[1]), dtype=torch.uint8)
            else:
                masks = mask == obj_ids[:, None, None]
                
                # Get bounding boxes
                boxes = []
                for i in range(len(obj_ids)):
                    pos = np.where(masks[i])
                    if len(pos[0]) == 0:  # Skip if mask is empty
                        continue
                    xmin = np.min(pos[1])
                    xmax = np.max(pos[1])
                    ymin = np.min(pos[0])
                    ymax = np.max(pos[0])
                    # Ensure box has valid dimensions
                    if xmax > xmin and ymax > ymin:
                        boxes.append([xmin, ymin, xmax, ymax])
                
                if not boxes:  # If no valid boxes were found
                    boxes = torch.zeros((0, 4), dtype=torch.float32)
                    labels = torch.zeros((0,), dtype=torch.int64)
                    masks = torch.zeros((0, mask.shape[0], mask.shape[1]), dtype=torch.uint8)
                else:
                    boxes = torch.as_tensor(boxes, dtype=torch.float32)
                    labels = torch.ones((len(boxes),), dtype=torch.int64) * class_id
                    masks = torch.as_tensor(masks, dtype=torch.uint8)
            
            image_id = torch.tensor([idx])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
            
            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["masks"] = masks
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd
            
            if self.transform is not None:
                img, target = self.transform(img, target)
            
            return img, target
        except Exception as e:
            print(f"Error loading item {idx}: {e}")
            # Return a valid but empty sample
            img = torch.zeros((3, 480, 854), dtype=torch.float32)  # Default DAVIS dimensions
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0, 480, 854), dtype=torch.uint8),
                "image_id": torch.tensor([idx]),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64)
            }
            return img, target
    
    def get_num_classes(self):
        return len(self.class_mapping) + 1  # +1 for background 
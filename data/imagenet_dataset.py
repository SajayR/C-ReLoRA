"""
ImageNet dataset loader for DinoV2 LoRA training.
"""

import os
from typing import Optional, Tuple, Callable, Any
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json


class ImageNetDataset(Dataset):
    """
    ImageNet dataset loader that works with the standard ImageNet directory structure:
    root/
        train/
            class_name/
                image1.jpg
                image2.jpg
                ...
        val/
            class_name/
                image1.jpg
                image2.jpg
                ...
    """
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        load_class_names: bool = True
    ):
        """
        Args:
            root: Root directory of ImageNet dataset
            split: Either 'train' or 'val'
            transform: Transform to apply to images
            target_transform: Transform to apply to targets
            load_class_names: Whether to load human-readable class names
        """
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        self.split_dir = os.path.join(root, split)
        if not os.path.exists(self.split_dir):
            raise RuntimeError(f"Dataset split directory {self.split_dir} not found")
        
        # Get all class directories
        self.classes = sorted([d for d in os.listdir(self.split_dir) 
                              if os.path.isdir(os.path.join(self.split_dir, d))])
        
        if not self.classes:
            raise RuntimeError(f"No class directories found in {self.split_dir}")
        
        # Create class to index mapping
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # Build samples list
        self.samples = []
        self._build_samples()
        
        print(f"Found {len(self.samples)} images in {len(self.classes)} classes for {split} split")
    
    def _build_samples(self):
        """Build list of (image_path, class_index) tuples."""
        for class_name in self.classes:
            class_dir = os.path.join(self.split_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            # Get all image files in this class directory
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    path = os.path.join(class_dir, filename)
                    self.samples.append((path, class_idx))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index: Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        
        # Load image
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return sample, target


def get_dinov2_transforms(image_size: int = 224, is_training: bool = True):
    """
    Get transforms for DinoV2 preprocessing.
    
    Args:
        image_size: Target image size
        is_training: Whether this is for training (includes augmentation)
    
    Returns:
        torchvision.transforms.Compose: Transform pipeline
    """
    if is_training:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def create_imagenet_dataloaders(
    data_root: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create ImageNet train and validation dataloaders.
    
    Args:
        data_root: Root directory containing train/ and val/ subdirectories
        batch_size: Batch size for dataloaders
        image_size: Target image size
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory
    
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Create transforms
    train_transform = get_dinov2_transforms(image_size, is_training=True)
    val_transform = get_dinov2_transforms(image_size, is_training=False)
    
    # Create datasets
    train_dataset = ImageNetDataset(
        root=data_root,
        split="train",
        transform=train_transform
    )
    
    val_dataset = ImageNetDataset(
        root=data_root,
        split="val",
        transform=val_transform
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_dataloader, val_dataloader


if __name__ == "__main__":
    # Test the dataset loader
    data_root = "/speedy/ImageNet"
    
    train_loader, val_loader = create_imagenet_dataloaders(
        data_root=data_root,
        batch_size=16,
        num_workers=2
    )
    
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    
    # Test loading a batch
    for batch_idx, (images, targets) in enumerate(train_loader):
        print(f"Batch {batch_idx}: images shape {images.shape}, targets shape {targets.shape}")
        print(f"Target range: {targets.min().item()} - {targets.max().item()}")
        break

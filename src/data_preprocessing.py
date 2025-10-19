"""
Data Preprocessing Module for CNN Image Classification

This module handles all data-related operations including:
1. Loading datasets (CIFAR-10)
2. Data normalization and standardization
3. Data augmentation for improving model generalization
4. Creating data loaders for efficient batch processing

Mathematical Concepts:
- Normalization: (x - mean) / std
- Data Augmentation: Geometric and color transformations
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import os


class DataPreprocessor:
    """
    Handles all data preprocessing operations for the CNN model.
    
    Key concepts:
    - Normalization: Scales pixel values to have mean=0, std=1
    - Augmentation: Creates variations of training data to prevent overfitting
    - Batching: Groups images together for efficient GPU processing
    """
    
    def __init__(self, data_dir='./data', batch_size=64, num_workers=2):
        """
        Initialize the data preprocessor.
        
        Args:
            data_dir (str): Directory to store/load datasets
            batch_size (int): Number of images per batch (affects GPU memory)
                            - Larger batch = more stable gradients but needs more memory
                            - Smaller batch = more noise in gradients but faster iterations
            num_workers (int): Number of subprocesses for data loading
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # CIFAR-10 statistics (pre-computed across entire dataset)
        # These are the mean and std for each RGB channel
        # Used for normalization: normalized_value = (value - mean) / std
        self.mean = (0.4914, 0.4822, 0.4465)  # RGB means
        self.std = (0.2470, 0.2435, 0.2616)   # RGB standard deviations
        
        # Class names for CIFAR-10
        self.classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck')
    
    def get_transforms(self, train=True):
        """
        Get image transformations for training or testing.
        
        Why normalize?
        - Neural networks work better with normalized inputs (mean=0, std=1)
        - Prevents certain features from dominating due to scale differences
        - Helps gradient descent converge faster and more reliably
        
        Mathematical formula for normalization:
        x_normalized = (x - μ) / σ
        where μ is mean, σ is standard deviation
        
        Args:
            train (bool): If True, apply data augmentation (only for training)
        
        Returns:
            transforms.Compose: Composition of image transformations
        """
        if train:
            # Training transformations with data augmentation
            # Data augmentation creates variations of training images to:
            # 1. Increase effective dataset size
            # 2. Make model more robust to variations
            # 3. Reduce overfitting
            transform = transforms.Compose([
                # Random horizontal flip with 50% probability
                # Makes model learn left-right invariance
                # e.g., a flipped car is still a car
                transforms.RandomHorizontalFlip(p=0.5),
                
                # Random cropping with padding
                # Teaches model to recognize objects at different positions
                # Padding of 4 pixels, then crop back to 32x32
                transforms.RandomCrop(32, padding=4),
                
                # Random rotation (±15 degrees)
                # Helps model handle slight rotations in real-world images
                transforms.RandomRotation(15),
                
                # Color jittering (random brightness, contrast, saturation)
                # Makes model robust to lighting conditions
                # brightness: ±20%, contrast: ±20%, saturation: ±20%
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                
                # Convert PIL Image to PyTorch Tensor
                # Changes shape from (H, W, C) to (C, H, W)
                # Scales pixel values from [0, 255] to [0.0, 1.0]
                transforms.ToTensor(),
                
                # Normalize using dataset statistics
                # Formula: output[channel] = (input[channel] - mean[channel]) / std[channel]
                # Result: Each channel has approximately mean=0, std=1
                transforms.Normalize(self.mean, self.std)
            ])
        else:
            # Test transformations (no augmentation)
            # We only normalize test data, no augmentation because:
            # - We want to evaluate on original images
            # - Augmentation is only for training to improve generalization
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
        
        return transform
    
    def load_data(self, val_split=0.1):
        """
        Load CIFAR-10 dataset and create train/validation/test splits.
        
        Dataset Information:
        - CIFAR-10: 60,000 32x32 color images in 10 classes
        - Training: 50,000 images (split into train/val)
        - Testing: 10,000 images
        - Each image: 32x32 pixels with 3 color channels (RGB)
        
        Args:
            val_split (float): Fraction of training data to use for validation
                             - Validation helps us tune hyperparameters
                             - Prevents overfitting to training data
        
        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        print("Loading CIFAR-10 dataset...")
        
        # Load training data with augmentation
        # download=True will download dataset if not present
        train_dataset_full = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.get_transforms(train=True)
        )
        
        # Load test data without augmentation
        test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.get_transforms(train=False)
        )
        
        # Split training data into train and validation sets
        # Why validation set?
        # - Training set: Used to update model weights
        # - Validation set: Used to tune hyperparameters and prevent overfitting
        # - Test set: Final evaluation (never used during training)
        
        train_size = int((1 - val_split) * len(train_dataset_full))
        val_size = len(train_dataset_full) - train_size
        
        # random_split creates random subsets
        # Uses a fixed seed for reproducibility
        train_dataset, val_dataset = random_split(
            train_dataset_full,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create DataLoaders for efficient batch processing
        # DataLoader benefits:
        # 1. Automatic batching: Groups samples into batches
        # 2. Shuffling: Randomizes order (prevents model from learning order)
        # 3. Parallel loading: Uses multiple workers for faster data loading
        # 4. Automatic tensor conversion
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # Shuffle training data each epoch
            num_workers=self.num_workers,
            pin_memory=True  # Speeds up GPU transfer
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # No need to shuffle validation data
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # No need to shuffle test data
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        print(f"Dataset loaded successfully!")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        print(f"Number of classes: {len(self.classes)}")
        print(f"Batch size: {self.batch_size}")
        
        return train_loader, val_loader, test_loader
    
    def denormalize(self, tensor):
        """
        Reverse the normalization to display images correctly.
        
        Mathematical formula (inverse of normalization):
        x_original = (x_normalized * σ) + μ
        
        This is useful for visualizing images after they've been normalized.
        
        Args:
            tensor (torch.Tensor): Normalized image tensor
        
        Returns:
            torch.Tensor: Denormalized image tensor
        """
        # Create denormalization transform
        # We reverse: (x - mean) / std
        # By doing: x * std + mean
        mean = torch.tensor(self.mean).view(3, 1, 1)
        std = torch.tensor(self.std).view(3, 1, 1)
        
        return tensor * std + mean
    
    def get_class_name(self, class_idx):
        """
        Get class name from class index.
        
        Args:
            class_idx (int): Class index (0-9 for CIFAR-10)
        
        Returns:
            str: Class name
        """
        return self.classes[class_idx]


# Example usage and testing
if __name__ == "__main__":
    # This code runs only when this file is executed directly
    # Useful for testing the module independently
    
    print("="*60)
    print("Testing Data Preprocessing Module")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(batch_size=32)
    
    # Load data
    train_loader, val_loader, test_loader = preprocessor.load_data()
    
    # Get a batch of training data
    images, labels = next(iter(train_loader))
    
    print(f"\nBatch Information:")
    print(f"Image batch shape: {images.shape}")  # Should be [batch_size, 3, 32, 32]
    print(f"Label batch shape: {labels.shape}")  # Should be [batch_size]
    print(f"Image data type: {images.dtype}")
    print(f"Label data type: {labels.dtype}")
    print(f"Image value range: [{images.min():.3f}, {images.max():.3f}]")
    
    # Display some class labels
    print(f"\nFirst 10 labels in batch: {labels[:10].tolist()}")
    print(f"Corresponding classes: {[preprocessor.get_class_name(label.item()) for label in labels[:10]]}")
    
    print("\n" + "="*60)
    print("Data preprocessing module test completed successfully!")
    print("="*60)

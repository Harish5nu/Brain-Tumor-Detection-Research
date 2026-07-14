"""
Preprocessing Pipeline for Brain Tumor Thesis
Cross-Dataset Generalization & Uncertainty-Aware Explainable Learning

Converts 4-class datasets to binary classification (Tumor vs No Tumor)
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from config import (
    BRISC_TRAIN, BRISC_TEST,
    NICKPARVAR_TRAIN, NICKPARVAR_TEST,
    CLASSES, IMAGE_SIZE, BATCH_SIZE, RANDOM_SEED,
    MEAN, STD, DEVICE, RESULTS_DIR
)

print("="*70)
print("🔄 PREPROCESSING PIPELINE")
print("="*70)
print(f"Device: {DEVICE}")
print(f"Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"Batch Size: {BATCH_SIZE}")
print("="*70)


# ============================================
# Dataset Class
# ============================================

class BrainMRIDataset(Dataset):
    """Custom Dataset for Brain MRI Images (Binary Classification)"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Label
        label = self.labels[idx]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


# ============================================
# Helper Functions
# ============================================

def get_image_paths_and_labels(base_path, class_list):
    """
    Get all image paths and binary labels (0 = No Tumor, 1 = Tumor)
    """
    tumor_classes = ['glioma', 'meningioma', 'pituitary']
    no_tumor_class = ['no_tumor']
    
    image_paths = []
    labels = []
    
    for class_name in class_list:
        folder_path = os.path.join(base_path, class_name)
        if not os.path.exists(folder_path):
            continue
        
        extensions = ('.jpg', '.jpeg', '.png')
        paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                 if f.lower().endswith(extensions)]
        
        # Binary label: 1 = Tumor, 0 = No Tumor
        if class_name in tumor_classes:
            label = 1
        else:
            label = 0
        
        image_paths.extend(paths)
        labels.extend([label] * len(paths))
    
    return image_paths, labels


def get_transforms():
    """Get data augmentation transforms"""
    
    # Training transforms (with augmentation)
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    
    # Validation/Test transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    
    return train_transform, val_transform


def get_class_weights(labels):
    """Calculate class weights for imbalanced dataset"""
    class_counts = [labels.count(0), labels.count(1)]
    total = sum(class_counts)
    weights = torch.tensor([total / class_counts[0], total / class_counts[1]], dtype=torch.float32)
    return weights


# ============================================
# Main Preprocessing Function
# ============================================

def prepare_brisc_data():
    """
    Prepare BRISC dataset for training and internal testing
    Returns: train_loader, val_loader, test_loader, class_weights
    """
    
    print("\n" + "="*70)
    print("📊 PREPARING BRISC DATASET (Training + Internal Test)")
    print("="*70)
    
    # Get all images and labels
    images, labels = get_image_paths_and_labels(BRISC_TRAIN, CLASSES)
    
    print(f"Total images: {len(images)}")
    print(f"  Tumor:     {labels.count(1)}")
    print(f"  No Tumor:  {labels.count(0)}")
    
    # Split: 80% train, 20% temp (for val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels, test_size=0.2, stratify=labels, random_state=RANDOM_SEED
    )
    
    # Split temp: 50% val, 50% test (10% each of total)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_SEED
    )
    
    print(f"\nData Split:")
    print(f"  Training:   {len(X_train)} ({len(X_train)/len(images)*100:.1f}%)")
    print(f"  Validation: {len(X_val)} ({len(X_val)/len(images)*100:.1f}%)")
    print(f"  Test:       {len(X_test)} ({len(X_test)/len(images)*100:.1f}%)")
    
    # Class distribution in splits
    def print_split_dist(y, name):
        tumor = sum(y)
        no_tumor = len(y) - tumor
        print(f"  {name} - Tumor: {tumor}, No Tumor: {no_tumor}")
    
    print(f"\nClass Distribution:")
    print_split_dist(y_train, "Train")
    print_split_dist(y_val, "Val")
    print_split_dist(y_test, "Test")
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    train_dataset = BrainMRIDataset(X_train, y_train, transform=train_transform)
    val_dataset = BrainMRIDataset(X_val, y_val, transform=val_transform)
    test_dataset = BrainMRIDataset(X_test, y_test, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"\nDataLoader Created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")
    
    # Calculate class weights
    class_weights = get_class_weights(y_train)
    print(f"\nClass Weights:")
    print(f"  No Tumor: {class_weights[0]:.4f}")
    print(f"  Tumor:    {class_weights[1]:.4f}")
    
    # Also return test images for evaluation
    test_images = X_test
    test_labels = y_test
    
    return train_loader, val_loader, test_loader, class_weights, test_images, test_labels


def prepare_nickparvar_data():
    """
    Prepare Nickparvar dataset for external testing
    Returns: test_loader
    """
    
    print("\n" + "="*70)
    print("📊 PREPARING NICKPARVAR DATASET (External Test)")
    print("="*70)
    
    # Get all images and labels from test set
    images, labels = get_image_paths_and_labels(NICKPARVAR_TEST, CLASSES)
    
    print(f"Total images: {len(images)}")
    print(f"  Tumor:     {labels.count(1)}")
    print(f"  No Tumor:  {labels.count(0)}")
    
    # Get transforms (validation only, no augmentation)
    _, val_transform = get_transforms()
    
    # Create dataset
    test_dataset = BrainMRIDataset(images, labels, transform=val_transform)
    
    # Create data loader
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"\nDataLoader Created:")
    print(f"  Test batches: {len(test_loader)}")
    
    return test_loader, images, labels


# ============================================
# Verify Batch Function
# ============================================

def verify_batch(loader, dataset_name):
    """Verify batch shape and values"""
    
    images, labels = next(iter(loader))
    
    print(f"\n📐 {dataset_name} - Batch Verification:")
    print(f"  Image shape: {images.shape}")
    print(f"  Label shape: {labels.shape}")
    print(f"  Label values: {labels[:10].tolist()}")
    print(f"  Image range: [{images.min():.2f}, {images.max():.2f}]")
    
    if images.shape[1] == 3 and images.shape[2] == IMAGE_SIZE and images.shape[3] == IMAGE_SIZE:
        print("  ✅ Batch shape is correct!")
    else:
        print("  ❌ Batch shape is incorrect!")
    
    return images, labels


# ============================================
# Visualize Augmentation
# ============================================

def visualize_augmentation(train_loader):
    """Visualize data augmentation effects"""
    
    print("\n" + "="*70)
    print("🖼️ DATA AUGMENTATION VISUALIZATION")
    print("="*70)
    
    # Get one batch
    images, labels = next(iter(train_loader))
    
    # Denormalize for display
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    images_denorm = images * std + mean
    images_denorm = torch.clamp(images_denorm, 0, 1)
    
    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    
    for i in range(min(16, len(images_denorm))):
        row, col = i // 4, i % 4
        img = images_denorm[i].permute(1, 2, 0).numpy()
        label = "TUMOR" if labels[i] == 1 else "NO TUMOR"
        axes[row, col].imshow(img)
        axes[row, col].set_title(f"Sample {i+1}: {label}", fontsize=10, fontweight='bold')
        axes[row, col].axis('off')
    
    plt.suptitle("Training Batch with Augmentation", fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "augmentation_samples.png")
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"✅ Augmentation visualization saved to: {save_path}")


# ============================================
# Main Execution
# ============================================

if __name__ == "__main__":
    
    # 1. Prepare BRISC dataset (Training + Internal Test)
    train_loader, val_loader, test_loader, class_weights, test_images, test_labels = prepare_brisc_data()
    
    # 2. Prepare Nickparvar dataset (External Test)
    nick_test_loader, nick_images, nick_labels = prepare_nickparvar_data()
    
    # 3. Verify batches
    verify_batch(train_loader, "BRISC Training")
    verify_batch(test_loader, "BRISC Test")
    verify_batch(nick_test_loader, "Nickparvar Test")
    
    # 4. Visualize augmentation
    visualize_augmentation(train_loader)
    
    # 5. Summary
    print("\n" + "="*70)
    print("✅ PREPROCESSING COMPLETE!")
    print("="*70)
    print(f"""
    Summary:
    ─────────
    BRISC (Training):
      Train:   {len(train_loader)} batches
      Val:     {len(val_loader)} batches
      Test:    {len(test_loader)} batches
    
    Nickparvar (External Test):
      Test:    {len(nick_test_loader)} batches
    
    Class Weights (BRISC Training):
      No Tumor: {class_weights[0]:.4f}
      Tumor:    {class_weights[1]:.4f}
    
    Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}
    Batch Size: {BATCH_SIZE}
    
    Next Step: Train Baseline Models
    """)
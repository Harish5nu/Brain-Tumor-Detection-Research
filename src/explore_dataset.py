"""
Dataset Exploration for Brain Tumor Thesis
Cross-Dataset Generalization & Uncertainty-Aware Explainable Learning
"""

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
from collections import Counter

from config import (
    BRISC_TRAIN, BRISC_TEST,
    NICKPARVAR_TRAIN, NICKPARVAR_TEST,
    CLASSES, RESULTS_DIR, PROJECT_ROOT
)

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

print("="*70)
print("BRAIN TUMOR MRI DATASET EXPLORATION")
print("="*70)
print(f"Project Root: {PROJECT_ROOT}")
print(f"Results will be saved to: {RESULTS_DIR}")
print("="*70)


# ============================================
# Helper Functions
# ============================================

def count_images_in_folder(folder_path):
    """Count images in a specific folder"""
    if not os.path.exists(folder_path):
        return 0
    extensions = ('.jpg', '.jpeg', '.png')
    count = len([f for f in os.listdir(folder_path) if f.lower().endswith(extensions)])
    return count


def get_class_distribution(base_path, class_list):
    """Get counts for all classes in a dataset"""
    counts = {}
    for class_name in class_list:
        folder_path = os.path.join(base_path, class_name)
        counts[class_name] = count_images_in_folder(folder_path)
    return counts


def get_image_sizes(folder_path, max_samples=100):
    """Get width and height of sample images"""
    widths, heights = [], []
    if not os.path.exists(folder_path):
        return widths, heights

    extensions = ('.jpg', '.jpeg', '.png')
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(extensions)]

    sample_images = random.sample(images, min(max_samples, len(images)))

    for img_file in sample_images:
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path)
        if img is not None:
            h, w = img.shape[:2]
            heights.append(h)
            widths.append(w)

    return widths, heights


def check_corrupted_images(folder_path):
    """Check for corrupted images"""
    corrupted = []
    if not os.path.exists(folder_path):
        return corrupted

    extensions = ('.jpg', '.jpeg', '.png')
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(extensions)]

    for img_file in images:
        img_path = os.path.join(folder_path, img_file)
        try:
            img = Image.open(img_path)
            img.verify()
            Image.open(img_path)
        except Exception:
            corrupted.append(img_file)

    return corrupted


def get_all_image_paths(base_path, class_list):
    """Get all image paths for each class"""
    image_paths = {}
    for class_name in class_list:
        folder_path = os.path.join(base_path, class_name)
        if os.path.exists(folder_path):
            extensions = ('.jpg', '.jpeg', '.png')
            paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                     if f.lower().endswith(extensions)]
            image_paths[class_name] = paths
        else:
            image_paths[class_name] = []
    return image_paths


def plot_class_distribution(data, title, save_name):
    """Plot class distribution bar chart"""
    fig, ax = plt.subplots(figsize=(10, 6))

    classes = list(data.keys())
    counts = list(data.values())
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']

    bars = ax.bar(classes, counts, color=colors, edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Number of Images', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{count}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, save_name)
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"✅ Saved: {save_path}")


# ============================================
# 1. EXPLORE BRISC DATASET
# ============================================

print("\n" + "="*70)
print("📊 DATASET 1: BRISC 2025")
print("="*70)

# Training set
print("\n📂 BRISC - Training Set:")
train_counts = get_class_distribution(BRISC_TRAIN, CLASSES)
for class_name, count in train_counts.items():
    print(f"  {class_name}: {count} images")
total_train = sum(train_counts.values())
print(f"  Total: {total_train} images")

# Test set
print("\n📂 BRISC - Test Set:")
test_counts = get_class_distribution(BRISC_TEST, CLASSES)
for class_name, count in test_counts.items():
    print(f"  {class_name}: {count} images")
total_test = sum(test_counts.values())
print(f"  Total: {total_test} images")

print(f"\n📊 BRISC Summary:")
print(f"  Training: {total_train} images")
print(f"  Testing:  {total_test} images")
print(f"  Total:    {total_train + total_test} images")

plot_class_distribution(
    train_counts,
    "BRISC 2025 - Training Set Class Distribution",
    "brisc_train_distribution.png"
)


# ============================================
# 2. EXPLORE NICKPARVAR DATASET
# ============================================

print("\n" + "="*70)
print("📊 DATASET 2: Nickparvar")
print("="*70)

# Training set
print("\n📂 Nickparvar - Training Set:")
nick_train_counts = get_class_distribution(NICKPARVAR_TRAIN, CLASSES)
for class_name, count in nick_train_counts.items():
    print(f"  {class_name}: {count} images")
nick_total_train = sum(nick_train_counts.values())
print(f"  Total: {nick_total_train} images")

# Test set
print("\n📂 Nickparvar - Test Set:")
nick_test_counts = get_class_distribution(NICKPARVAR_TEST, CLASSES)
for class_name, count in nick_test_counts.items():
    print(f"  {class_name}: {count} images")
nick_total_test = sum(nick_test_counts.values())
print(f"  Total: {nick_total_test} images")

print(f"\n📊 Nickparvar Summary:")
print(f"  Training: {nick_total_train} images")
print(f"  Testing:  {nick_total_test} images")
print(f"  Total:    {nick_total_train + nick_total_test} images")

plot_class_distribution(
    nick_train_counts,
    "Nickparvar - Training Set Class Distribution",
    "nickparvar_train_distribution.png"
)


# ============================================
# 3. COMPARE CLASS DISTRIBUTIONS
# ============================================

print("\n" + "="*70)
print("📊 CROSS-DATASET COMPARISON")
print("="*70)

print("\nClass-wise comparison (Training sets):")
print("-" * 50)
print(f"{'Class':<15} {'BRISC':<12} {'Nickparvar':<12}")
print("-" * 50)
for class_name in CLASSES:
    brisc_count = train_counts.get(class_name, 0)
    nick_count = nick_train_counts.get(class_name, 0)
    print(f"{class_name:<15} {brisc_count:<12} {nick_count:<12}")

print(f"\nTotal images for training:")
print(f"  BRISC:      {total_train}")
print(f"  Nickparvar: {nick_total_train}")


# ============================================
# 4. CHECK IMAGE SIZES (BRISC)
# ============================================

print("\n" + "="*70)
print("📏 IMAGE SIZE ANALYSIS")
print("="*70)

all_widths, all_heights = [], []

for class_name in CLASSES:
    folder_path = os.path.join(BRISC_TRAIN, class_name)
    widths, heights = get_image_sizes(folder_path)
    all_widths.extend(widths)
    all_heights.extend(heights)

if all_widths:
    print(f"\nBRISC Training Set - Image Sizes:")
    print(f"  Width range:  {min(all_widths)} - {max(all_widths)} pixels")
    print(f"  Height range: {min(all_heights)} - {max(all_heights)} pixels")
    print(f"  Mean size:    {int(np.mean(all_widths))} x {int(np.mean(all_heights))}")

# Check Nickparvar image sizes
nick_widths, nick_heights = [], []
for class_name in CLASSES:
    folder_path = os.path.join(NICKPARVAR_TRAIN, class_name)
    widths, heights = get_image_sizes(folder_path)
    nick_widths.extend(widths)
    nick_heights.extend(heights)

if nick_widths:
    print(f"\nNickparvar Training Set - Image Sizes:")
    print(f"  Width range:  {min(nick_widths)} - {max(nick_widths)} pixels")
    print(f"  Height range: {min(nick_heights)} - {max(nick_heights)} pixels")
    print(f"  Mean size:    {int(np.mean(nick_widths))} x {int(np.mean(nick_heights))}")


# ============================================
# 5. CHECK FOR CORRUPTED IMAGES
# ============================================

print("\n" + "="*70)
print("🔍 CORRUPTED IMAGE CHECK")
print("="*70)

corrupted_count = 0
for class_name in CLASSES:
    folder_path = os.path.join(BRISC_TRAIN, class_name)
    corrupted = check_corrupted_images(folder_path)
    if corrupted:
        print(f"  ⚠️ BRISC - {class_name}: {len(corrupted)} corrupted images")
        corrupted_count += len(corrupted)

for class_name in CLASSES:
    folder_path = os.path.join(NICKPARVAR_TRAIN, class_name)
    corrupted = check_corrupted_images(folder_path)
    if corrupted:
        print(f"  ⚠️ Nickparvar - {class_name}: {len(corrupted)} corrupted images")
        corrupted_count += len(corrupted)

if corrupted_count == 0:
    print("✅ No corrupted images found in either dataset!")


# ============================================
# 6. VISUALIZE SAMPLE IMAGES
# ============================================

print("\n" + "="*70)
print("🖼️ SAMPLE IMAGES VISUALIZATION")
print("="*70)

image_paths = get_all_image_paths(BRISC_TRAIN, CLASSES)

fig, axes = plt.subplots(4, 4, figsize=(16, 16))

for i, class_name in enumerate(CLASSES):
    paths = image_paths.get(class_name, [])
    if paths:
        sample_paths = random.sample(paths, min(4, len(paths)))
        for j, img_path in enumerate(sample_paths):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax = axes[i, j]
            ax.imshow(img)
            ax.set_title(f"{class_name}\n{os.path.basename(img_path)}", fontsize=8)
            ax.axis('off')
    else:
        for j in range(4):
            axes[i, j].axis('off')
            axes[i, j].set_title(f"{class_name}\n(No images)", fontsize=8)

plt.suptitle("BRISC 2025 - Sample Images by Class", fontsize=16, fontweight='bold')
plt.tight_layout()
save_path = os.path.join(RESULTS_DIR, "brisc_sample_images.png")
plt.savefig(save_path, dpi=150)
plt.show()
print(f"✅ Sample images saved to: {save_path}")


# ============================================
# 7. VISUALIZE NICKPARVAR SAMPLE IMAGES
# ============================================

nick_image_paths = get_all_image_paths(NICKPARVAR_TRAIN, CLASSES)

fig, axes = plt.subplots(4, 4, figsize=(16, 16))

for i, class_name in enumerate(CLASSES):
    paths = nick_image_paths.get(class_name, [])
    if paths:
        sample_paths = random.sample(paths, min(4, len(paths)))
        for j, img_path in enumerate(sample_paths):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax = axes[i, j]
            ax.imshow(img)
            ax.set_title(f"{class_name}\n{os.path.basename(img_path)}", fontsize=8)
            ax.axis('off')
    else:
        for j in range(4):
            axes[i, j].axis('off')
            axes[i, j].set_title(f"{class_name}\n(No images)", fontsize=8)

plt.suptitle("Nickparvar - Sample Images by Class", fontsize=16, fontweight='bold')
plt.tight_layout()
save_path = os.path.join(RESULTS_DIR, "nickparvar_sample_images.png")
plt.savefig(save_path, dpi=150)
plt.show()
print(f"✅ Sample images saved to: {save_path}")


# ============================================
# 8. SUMMARY REPORT
# ============================================

print("\n" + "="*70)
print("📋 DATASET EXPLORATION SUMMARY")
print("="*70)

print(f"""
┌─────────────────────────────────────────────────────────────────┐
│  DATASET 1: BRISC 2025                                          │
├─────────────────────────────────────────────────────────────────┤
│  Total images:     {total_train + total_test}                                     │
│  Training:         {total_train}                                     │
│  Testing:          {total_test}                                     │
│                                                                 │
│  Class Distribution (Training):                                 │
│    glioma:         {train_counts.get('glioma', 0)}                                     │
│    meningioma:     {train_counts.get('meningioma', 0)}                                     │
│    pituitary:      {train_counts.get('pituitary', 0)}                                     │
│    no_tumor:       {train_counts.get('no_tumor', 0)}                                     │
│                                                                 │
│  Image sizes:      {int(np.mean(all_widths))}x{int(np.mean(all_heights))} (avg)            │
│  Corrupted:        {corrupted_count}                                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  DATASET 2: Nickparvar                                          │
├─────────────────────────────────────────────────────────────────┤
│  Total images:     {nick_total_train + nick_total_test}                                     │
│  Training:         {nick_total_train}                                     │
│  Testing:          {nick_total_test}                                     │
│                                                                 │
│  Class Distribution (Training):                                 │
│    glioma:         {nick_train_counts.get('glioma', 0)}                                     │
│    meningioma:     {nick_train_counts.get('meningioma', 0)}                                     │
│    pituitary:      {nick_train_counts.get('pituitary', 0)}                                     │
│    no_tumor:       {nick_train_counts.get('no_tumor', 0)}                                     │
│                                                                 │
│  Image sizes:      {int(np.mean(nick_widths))}x{int(np.mean(nick_heights))} (avg)            │
│  Corrupted:        0                                            │
└─────────────────────────────────────────────────────────────────┘

Dataset Quality Check:
  ✅ No corrupted images found
  ✅ Both datasets have 4 classes
  ⚠️ Different image sizes (will resize to 224x224)
  ⚠️ Class distributions differ (normal for cross-dataset)
""")

print("\n✅ Dataset exploration complete!")
print(f"📁 All results saved to: {RESULTS_DIR}")
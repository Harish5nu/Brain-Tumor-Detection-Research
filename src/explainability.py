"""
Explainability with Grad-CAM
Visualizes where the model focuses when making predictions
"""

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.cm as cm

from config import MODELS_DIR, RESULTS_DIR, DEVICE, IMAGE_SIZE, MEAN, STD
from preprocessing import prepare_brisc_data, prepare_nickparvar_data

print("="*70)
print("🔍 EXPLAINABILITY: GRAD-CAM VISUALIZATION")
print("="*70)
print(f"Device: {DEVICE}")
print("="*70)

# ============================================
# 1. Grad-CAM Class
# ============================================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping
    """
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Save activations from forward pass"""
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        """Save gradients from backward pass"""
        self.gradients = grad_output[0]
    
    def generate_heatmap(self, input_image, target_class=None):
        """
        Generate Grad-CAM heatmap for a given image
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]
        
        # Global Average Pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weighted combination of activations
        heatmap = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            heatmap += w * activations[i, :, :]
        
        # ReLU activation (positive influences only)
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize heatmap
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        
        return heatmap

# ============================================
# 2. Load Models
# ============================================

print("\n🏗️ Loading models for Grad-CAM...")

# Load DenseNet121 (best model)
densenet = models.densenet121(weights=None)
num_features = densenet.classifier.in_features
densenet.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 2)
)
densenet.load_state_dict(torch.load(os.path.join(MODELS_DIR, "optimized_LR_1e-3.pth")))
densenet = densenet.to(DEVICE)
densenet.eval()
print("✅ DenseNet121 (Optimized) loaded")

# Get the last convolutional layer (features[-1] for DenseNet)
target_layer = densenet.features[-1]
grad_cam = GradCAM(densenet, target_layer)
print(f"✅ Grad-CAM initialized (target layer: {target_layer.__class__.__name__})")

# ============================================
# 3. Helper Functions
# ============================================

def load_and_preprocess_image(img_path):
    """Load and preprocess an image for the model"""
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
    
    # Normalize
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
    
    return image_tensor, image

def overlay_heatmap(original_image, heatmap, alpha=0.6):
    """Overlay heatmap on original image"""
    # Resize heatmap to image size
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    
    # Convert heatmap to colormap
    heatmap_colored = cm.jet(heatmap)[:, :, :3] * 255
    heatmap_colored = heatmap_colored.astype(np.uint8)
    
    # Overlay
    overlay = cv2.addWeighted(original_image.astype(np.uint8), 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlay

def visualize_gradcam(image_path, save_name, target_class=None):
    """Generate and visualize Grad-CAM for a single image"""
    
    # Load and preprocess image
    image_tensor, original_image = load_and_preprocess_image(image_path)
    
    # Get prediction
    with torch.no_grad():
        output = densenet(image_tensor)
        prob = torch.softmax(output, dim=1)
        pred_class = torch.argmax(output, dim=1).item()
        confidence = prob[0][pred_class].item() * 100
    
    # Generate heatmap
    heatmap = grad_cam.generate_heatmap(image_tensor, target_class)
    
    # Overlay
    overlay = overlay_heatmap(original_image, heatmap)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image", fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(heatmap, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM Heatmap", fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay\nPrediction: {['No Tumor', 'Tumor'][pred_class]} ({confidence:.1f}%)", 
                     fontweight='bold')
    axes[2].axis('off')
    
    plt.suptitle(f"Grad-CAM Visualization\nFile: {os.path.basename(image_path)}", 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, save_name), dpi=300)
    plt.show()
    
    return pred_class, confidence

def get_sample_images(loader, label=0, num_samples=4):
    """Get sample image paths from a dataloader"""
    image_paths = []
    for images, labels in loader:
        for i, img in enumerate(images):
            if labels[i] == label:
                # Since we don't have paths in the dataloader, we need to load from test data
                pass
    return image_paths

# ============================================
# 4. Load Test Data
# ============================================

print("\n📂 Loading test data...")
train_loader, val_loader, test_loader, class_weights, test_images, test_labels = prepare_brisc_data()
nick_test_loader, nick_images, nick_labels = prepare_nickparvar_data()

print(f"BRISC Test: {len(test_images)} images")
print(f"Nickparvar Test: {len(nick_images)} images")

# ============================================
# 5. Visualize Grad-CAM on BRISC Test Images
# ============================================

print("\n" + "="*70)
print("🖼️ GRAD-CAM ON BRISC TEST IMAGES")
print("="*70)

# Get sample images for each class
brisc_tumor_paths = [p for p, l in zip(test_images, test_labels) if l == 1]
brisc_no_tumor_paths = [p for p, l in zip(test_images, test_labels) if l == 0]

# Tumor samples
print("\n📊 Tumor samples (BRISC):")
for i, img_path in enumerate(brisc_tumor_paths[:3]):
    print(f"  Image {i+1}: {os.path.basename(img_path)}")
    pred_class, confidence = visualize_gradcam(
        img_path, 
        f"gradcam_brisc_tumor_{i+1}.png"
    )

# No Tumor samples
print("\n📊 No Tumor samples (BRISC):")
for i, img_path in enumerate(brisc_no_tumor_paths[:3]):
    print(f"  Image {i+1}: {os.path.basename(img_path)}")
    pred_class, confidence = visualize_gradcam(
        img_path, 
        f"gradcam_brisc_no_tumor_{i+1}.png"
    )

# ============================================
# 6. Visualize Grad-CAM on Nickparvar Images
# ============================================

print("\n" + "="*70)
print("🖼️ GRAD-CAM ON NICKPARVAR TEST IMAGES")
print("="*70)

nick_tumor_paths = [p for p, l in zip(nick_images, nick_labels) if l == 1]
nick_no_tumor_paths = [p for p, l in zip(nick_images, nick_labels) if l == 0]

# Tumor samples
print("\n📊 Tumor samples (Nickparvar):")
for i, img_path in enumerate(nick_tumor_paths[:3]):
    print(f"  Image {i+1}: {os.path.basename(img_path)}")
    pred_class, confidence = visualize_gradcam(
        img_path, 
        f"gradcam_nickparvar_tumor_{i+1}.png"
    )

# No Tumor samples
print("\n📊 No Tumor samples (Nickparvar):")
for i, img_path in enumerate(nick_no_tumor_paths[:3]):
    print(f"  Image {i+1}: {os.path.basename(img_path)}")
    pred_class, confidence = visualize_gradcam(
        img_path, 
        f"gradcam_nickparvar_no_tumor_{i+1}.png"
    )

# ============================================
# 7. Comparison Grid
# ============================================

print("\n" + "="*70)
print("📊 GRAD-CAM COMPARISON GRID")
print("="*70)

def create_comparison_grid():
    """Create a grid comparing Grad-CAM on BRISC vs Nickparvar"""
    
    # Get one image from each dataset (tumor)
    brisc_sample = brisc_tumor_paths[0] if brisc_tumor_paths else None
    nick_sample = nick_tumor_paths[0] if nick_tumor_paths else None
    
    if brisc_sample and nick_sample:
        fig, axes = plt.subplots(2, 4, figsize=(16, 10))
        
        # BRISC samples
        for i, img_path in enumerate([brisc_sample]):
            image_tensor, original_image = load_and_preprocess_image(img_path)
            heatmap = grad_cam.generate_heatmap(image_tensor)
            overlay = overlay_heatmap(original_image, heatmap)
            
            axes[0, i].imshow(original_image)
            axes[0, i].set_title("BRISC: Original", fontweight='bold')
            axes[0, i].axis('off')
            
            axes[0, i+1].imshow(overlay)
            axes[0, i+1].set_title("BRISC: Grad-CAM", fontweight='bold')
            axes[0, i+1].axis('off')
        
        # Nickparvar samples
        for i, img_path in enumerate([nick_sample]):
            image_tensor, original_image = load_and_preprocess_image(img_path)
            heatmap = grad_cam.generate_heatmap(image_tensor)
            overlay = overlay_heatmap(original_image, heatmap)
            
            axes[1, i].imshow(original_image)
            axes[1, i].set_title("Nickparvar: Original", fontweight='bold')
            axes[1, i].axis('off')
            
            axes[1, i+1].imshow(overlay)
            axes[1, i+1].set_title("Nickparvar: Grad-CAM", fontweight='bold')
            axes[1, i+1].axis('off')
        
        plt.suptitle("Grad-CAM Comparison: BRISC vs Nickparvar", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "gradcam_comparison_grid.png"), dpi=300)
        plt.show()
        print(f"✅ Grad-CAM comparison grid saved to: {RESULTS_DIR}/gradcam_comparison_grid.png")

create_comparison_grid()

print("\n" + "="*70)
print("✅ GRAD-CAM VISUALIZATION COMPLETE!")
print("="*70)
print(f"""
All Grad-CAM visualizations saved to: {RESULTS_DIR}

Files created:
  - gradcam_brisc_tumor_*.png
  - gradcam_brisc_no_tumor_*.png
  - gradcam_nickparvar_tumor_*.png
  - gradcam_nickparvar_no_tumor_*.png
  - gradcam_comparison_grid.png

Interpretation:
  🔴 Red/Hot areas = Where the model focused
  🔵 Blue/Cool areas = Where the model ignored

  ✅ If heatmap is on tumor region → Model is looking at the right place!
  ❌ If heatmap is outside tumor region → Model might be using wrong features!
""")
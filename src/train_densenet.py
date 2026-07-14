"""
Train DenseNet121 Baseline Model for Brain Tumor Classification
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
import seaborn as sns
import pandas as pd

from config import MODELS_DIR, RESULTS_DIR, DEVICE, NUM_EPOCHS, LEARNING_RATE
from preprocessing import prepare_brisc_data, prepare_nickparvar_data

print("="*70)
print("🧠 TRAINING DENSENET121 BASELINE MODEL")
print("="*70)
print(f"Device: {DEVICE}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Learning Rate: {LEARNING_RATE}")
print("="*70)

# ============================================
# 1. Load Data
# ============================================

print("\n📂 Loading data...")
train_loader, val_loader, test_loader, class_weights, test_images, test_labels = prepare_brisc_data()
nick_test_loader, nick_images, nick_labels = prepare_nickparvar_data()

# ============================================
# 2. Build Model
# ============================================

print("\n🏗️ Building DenseNet121 model...")

# Load pre-trained DenseNet121
model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

# Freeze all layers first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last few layers for fine-tuning
for param in model.features[-4:].parameters():
    param.requires_grad = True

# Replace classifier head
num_features = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 2)  # 2 classes: No Tumor (0), Tumor (1)
)

model = model.to(DEVICE)

# Count trainable parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ============================================
# 3. Loss Function & Optimizer
# ============================================

criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

# ============================================
# 4. Training Loop
# ============================================

print("\n" + "="*70)
print("🚀 STARTING TRAINING")
print("="*70)

train_losses = []
val_losses = []
train_accs = []
val_accs = []
best_val_acc = 0
patience_counter = 0
early_stopping_patience = 10

for epoch in range(NUM_EPOCHS):
    # ---------- Training ----------
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
    for images, labels in loop:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        loop.set_postfix(loss=loss.item(), acc=100*correct/total)
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    
    # ---------- Validation ----------
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]")
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loop.set_postfix(loss=loss.item(), acc=100*correct/total)
    
    val_loss = val_loss / len(val_loader)
    val_acc = 100 * correct / total
    
    # Store history
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    
    # Update scheduler
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    # Print summary
    print(f"\n📊 Epoch {epoch+1}/{NUM_EPOCHS}")
    print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
    print(f"   LR: {current_lr:.6f}")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(MODELS_DIR, "densenet121_best.pth"))
        print(f"   ✅ Best model saved! (Val Acc: {val_acc:.2f}%)")
        patience_counter = 0
    else:
        patience_counter += 1
    
    # Early stopping
    if patience_counter >= early_stopping_patience:
        print(f"\n⚠️ Early stopping triggered after {epoch+1} epochs")
        break
    
    print("-" * 50)

print(f"\n✅ Training complete!")
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

# ============================================
# 5. Training History Plots
# ============================================

print("\n📈 Generating training plots...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(train_losses, label='Train Loss', marker='o', linewidth=2)
ax1.plot(val_losses, label='Val Loss', marker='s', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('DenseNet121 - Training & Validation Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(train_accs, label='Train Acc', marker='o', linewidth=2)
ax2.plot(val_accs, label='Val Acc', marker='s', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('DenseNet121 - Training & Validation Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "densenet121_training.png"), dpi=300)
plt.show()
print(f"✅ Training plot saved to: {RESULTS_DIR}/densenet121_training.png")

# ============================================
# 6. Evaluation Function
# ============================================

def evaluate_model(model, test_loader, dataset_name):
    """Evaluate model on test set"""
    
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Testing on {dataset_name}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Results dictionary
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels
    }
    
    return results

# ============================================
# 7. Evaluate on BRISC Test Set
# ============================================

print("\n" + "="*70)
print("📊 EVALUATING ON BRISC TEST SET")
print("="*70)

# Load best model
model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "densenet121_best.pth")))
brisc_results = evaluate_model(model, test_loader, "BRISC Test")

print(f"\nBRISC Test Results:")
print(f"  Accuracy:  {brisc_results['accuracy']*100:.2f}%")
print(f"  Precision: {brisc_results['precision']*100:.2f}%")
print(f"  Recall:    {brisc_results['recall']*100:.2f}%")
print(f"  F1-Score:  {brisc_results['f1']*100:.2f}%")
print(f"  AUC-ROC:   {brisc_results['auc']*100:.2f}%")

# ============================================
# 8. Evaluate on Nickparvar (Cross-Dataset)
# ============================================

print("\n" + "="*70)
print("📊 CROSS-DATASET EVALUATION: Nickparvar (External Test)")
print("="*70)

nick_results = evaluate_model(model, nick_test_loader, "Nickparvar Test")

print(f"\nNickparvar Test Results:")
print(f"  Accuracy:  {nick_results['accuracy']*100:.2f}%")
print(f"  Precision: {nick_results['precision']*100:.2f}%")
print(f"  Recall:    {nick_results['recall']*100:.2f}%")
print(f"  F1-Score:  {nick_results['f1']*100:.2f}%")
print(f"  AUC-ROC:   {nick_results['auc']*100:.2f}%")

# Performance drop
drop = (brisc_results['accuracy'] - nick_results['accuracy']) * 100
print(f"\n📉 Performance Drop: {drop:.2f}%")

# ============================================
# 9. Confusion Matrix Visualization
# ============================================

def plot_confusion_matrix(cm, dataset_name, save_name):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Tumor', 'Tumor'],
                yticklabels=['No Tumor', 'Tumor'],
                annot_kws={'size': 16})
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title(f'DenseNet121 - Confusion Matrix ({dataset_name})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, save_name), dpi=300)
    plt.show()
    print(f"✅ Confusion matrix saved to: {RESULTS_DIR}/{save_name}")

plot_confusion_matrix(brisc_results['confusion_matrix'], "BRISC Test", "densenet121_brisc_cm.png")
plot_confusion_matrix(nick_results['confusion_matrix'], "Nickparvar Test", "densenet121_nickparvar_cm.png")

# ============================================
# 10. Save Results
# ============================================

results_df = pd.DataFrame({
    'Dataset': ['BRISC Test', 'Nickparvar Test'],
    'Accuracy': [brisc_results['accuracy'], nick_results['accuracy']],
    'Precision': [brisc_results['precision'], nick_results['precision']],
    'Recall': [brisc_results['recall'], nick_results['recall']],
    'F1-Score': [brisc_results['f1'], nick_results['f1']],
    'AUC-ROC': [brisc_results['auc'], nick_results['auc']]
})

print("\n" + "="*70)
print("📋 RESULTS SUMMARY")
print("="*70)
print(results_df.to_string(index=False))

results_df.to_csv(os.path.join(RESULTS_DIR, "densenet121_results.csv"), index=False)
print(f"\n✅ Results saved to: {RESULTS_DIR}/densenet121_results.csv")

# ============================================
# 11. ROC Curves
# ============================================

fig, ax = plt.subplots(figsize=(8, 6))

for name, results in [('BRISC', brisc_results), ('Nickparvar', nick_results)]:
    fpr, tpr, _ = roc_curve(results['labels'], results['probabilities'])
    ax.plot(fpr, tpr, label=f'{name} (AUC = {results["auc"]*100:.2f}%)', linewidth=2)

ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('DenseNet121 - ROC Curves', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "densenet121_roc.png"), dpi=300)
plt.show()
print(f"✅ ROC curves saved to: {RESULTS_DIR}/densenet121_roc.png")

print("\n" + "="*70)
print("✅ DENSENET121 TRAINING COMPLETE!")
print("="*70)
print(f"""
Summary:
---------
Best Validation Accuracy: {best_val_acc:.2f}%
BRISC Test Accuracy:      {brisc_results['accuracy']*100:.2f}%
Nickparvar Test Accuracy: {nick_results['accuracy']*100:.2f}%
Performance Drop:         {drop:.2f}%

Results saved to: {RESULTS_DIR}
Model saved to:   {MODELS_DIR}/densenet121_best.pth
""")
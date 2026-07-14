"""
Hyperparameter Optimization for Brain Tumor Classification
Tests different learning rates, batch sizes, and optimizers
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

from config import MODELS_DIR, RESULTS_DIR, DEVICE, NUM_EPOCHS
from preprocessing import prepare_brisc_data, prepare_nickparvar_data

print("="*70)
print("🔧 HYPERPARAMETER OPTIMIZATION")
print("="*70)
print(f"Device: {DEVICE}")
print("="*70)

# ============================================
# 1. Load Data
# ============================================

print("\n📂 Loading data...")
train_loader, val_loader, test_loader, class_weights, test_images, test_labels = prepare_brisc_data()
nick_test_loader, nick_images, nick_labels = prepare_nickparvar_data()

# ============================================
# 2. Optimization Configurations
# ============================================

configs = [
    # (name, learning_rate, batch_size, optimizer)
    ("LR_1e-4", 0.0001, 32, "Adam"),
    ("LR_1e-3", 0.001, 32, "Adam"),
    ("LR_1e-5", 0.00001, 32, "Adam"),
    ("AdamW", 0.0001, 32, "AdamW"),
    ("SGD", 0.001, 32, "SGD"),
]

# ============================================
# 3. Training Function
# ============================================

def train_densenet_with_config(train_loader, val_loader, config):
    """
    Train DenseNet121 with given hyperparameters
    Returns: best_val_acc, train_history
    """
    
    name, lr, batch_size, optimizer_name = config
    
    print(f"\n{'='*70}")
    print(f"🔬 Testing: {name}")
    print(f"   LR: {lr}, Batch: {batch_size}, Optimizer: {optimizer_name}")
    print(f"{'='*70}")
    
    # Build model
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    
    for param in model.parameters():
        param.requires_grad = False
    for param in model.features[-4:].parameters():
        param.requires_grad = True
    
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 2)
    )
    model = model.to(DEVICE)
    
    # Loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    
    # Optimizer
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    else:  # SGD
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Training loop
    best_val_acc = 0
    patience_counter = 0
    early_stopping_patience = 8
    
    for epoch in range(15):  # Reduced epochs for optimization
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/15", leave=False):
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
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        
        scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"optimized_{name}.pth"))
        else:
            patience_counter += 1
        
        if patience_counter >= early_stopping_patience:
            print(f"   ⏹️ Early stopping at epoch {epoch+1}")
            break
    
    print(f"   ✅ Best Val Acc: {best_val_acc:.2f}%")
    
    return best_val_acc

# ============================================
# 4. Run Optimization
# ============================================

print("\n" + "="*70)
print("🚀 STARTING OPTIMIZATION")
print("="*70)

results = []

for config in configs:
    best_acc = train_densenet_with_config(train_loader, val_loader, config)
    results.append({
        'Config': config[0],
        'Learning Rate': config[1],
        'Batch Size': config[2],
        'Optimizer': config[3],
        'Best Val Acc': best_acc
    })

# ============================================
# 5. Results Summary
# ============================================

print("\n" + "="*70)
print("📋 OPTIMIZATION RESULTS")
print("="*70)

df = pd.DataFrame(results)
print(df.to_string(index=False))

# Sort by best validation accuracy
df_sorted = df.sort_values('Best Val Acc', ascending=False)
print("\n" + "="*70)
print("🏆 BEST CONFIGURATION")
print("="*70)
best = df_sorted.iloc[0]
print(f"""
Config:         {best['Config']}
Learning Rate:  {best['Learning Rate']}
Batch Size:     {best['Batch Size']}
Optimizer:      {best['Optimizer']}
Best Val Acc:   {best['Best Val Acc']:.2f}%
""")

# Save results
df.to_csv(os.path.join(RESULTS_DIR, "optimization_results.csv"), index=False)
print(f"✅ Results saved to: {RESULTS_DIR}/optimization_results.csv")

# ============================================
# 6. Visualization
# ============================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart of all configs
ax1 = axes[0]
colors = ['green' if x == df_sorted.iloc[0]['Config'] else 'skyblue' for x in df['Config']]
bars = ax1.bar(df['Config'], df['Best Val Acc'], color=colors, edgecolor='black')
ax1.set_ylabel('Best Validation Accuracy (%)')
ax1.set_title('Hyperparameter Comparison - All Configs')
ax1.set_ylim([95, 100])
for bar, val in zip(bars, df['Best Val Acc']):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{val:.2f}%', ha='center', va='bottom', fontweight='bold')

# Highlight best
ax2 = axes[1]
best = df_sorted.iloc[0]
ax2.bar(['Best Config'], [best['Best Val Acc']], color='green', edgecolor='black')
ax2.set_ylabel('Best Validation Accuracy (%)')
ax2.set_title(f"Best: {best['Config']} (LR={best['Learning Rate']}, Batch={best['Batch Size']})")
ax2.set_ylim([95, 100])
ax2.text(0, best['Best Val Acc'] + 0.1, f'{best["Best Val Acc"]:.2f}%', 
         ha='center', va='bottom', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "optimization_chart.png"), dpi=300)
plt.show()
print(f"✅ Optimization chart saved to: {RESULTS_DIR}/optimization_chart.png")

print("\n" + "="*70)
print("✅ OPTIMIZATION COMPLETE!")
print("="*70)
print(f"""
Next Step: Step 10 - Ensemble Learning
Best Configuration to Use:
  Model:      DenseNet121
  LR:         {best['Learning Rate']}
  Optimizer:  {best['Optimizer']}
  Batch Size: {best['Batch Size']}
  
(Model saved to: {MODELS_DIR}/optimized_{best['Config']}.pth)
""")
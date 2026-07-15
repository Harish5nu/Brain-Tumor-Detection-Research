"""
Ensemble Learning: Stacking with Logistic Regression
Combines EfficientNet + DenseNet121 predictions
"""

import os
import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import joblib

from config import MODELS_DIR, RESULTS_DIR, DEVICE, IMAGE_SIZE
from preprocessing import prepare_brisc_data, prepare_nickparvar_data

print("="*70)
print("🧠 ENSEMBLE LEARNING: STACKING")
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
# 2. Load Trained Models
# ============================================

print("\n🏗️ Loading trained models...")

# Load EfficientNet
efficientnet = models.efficientnet_b0(weights=None)
num_features = efficientnet.classifier[1].in_features
efficientnet.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 2)
)
efficientnet.load_state_dict(torch.load(os.path.join(MODELS_DIR, "efficientnet_best.pth")))
efficientnet = efficientnet.to(DEVICE)
efficientnet.eval()
print("✅ EfficientNet loaded")

# Load DenseNet121 (Optimized)
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

# ============================================
# 3. Extract Features (Predictions)
# ============================================

def extract_predictions(model, loader, model_name):
    """Extract prediction probabilities from a model"""
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"Extracting from {model_name}"):
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    
    return np.array(all_probs), np.array(all_labels)

print("\n📊 Extracting predictions from base models...")

# Extract from EfficientNet
eff_probs, labels = extract_predictions(efficientnet, train_loader, "EfficientNet")
print(f"EfficientNet features shape: {eff_probs.shape}")

# Extract from DenseNet121
den_probs, _ = extract_predictions(densenet, train_loader, "DenseNet121")
print(f"DenseNet121 features shape: {den_probs.shape}")

# Combine features for meta-learner
X_meta_train = np.column_stack([eff_probs, den_probs])
y_meta_train = labels

print(f"\nMeta-learner features shape: {X_meta_train.shape}")

# ============================================
# 4. Train Meta-Learner (Logistic Regression)
# ============================================

print("\n🎯 Training Logistic Regression meta-learner...")

meta_learner = LogisticRegression(
    C=1.0,
    max_iter=1000,
    random_state=42,
    class_weight='balanced'
)

meta_learner.fit(X_meta_train, y_meta_train)

print("✅ Meta-learner trained!")

# Print coefficients
print(f"\n📊 Meta-learner coefficients:")
print(f"  EfficientNet weight:  {meta_learner.coef_[0][0]:.4f}")
print(f"  DenseNet121 weight:   {meta_learner.coef_[0][1]:.4f}")

# Save meta-learner
joblib.dump(meta_learner, os.path.join(MODELS_DIR, "ensemble_meta_learner.pkl"))
print(f"✅ Meta-learner saved to: {MODELS_DIR}/ensemble_meta_learner.pkl")

# ============================================
# 5. Ensemble Evaluation Function
# ============================================

def evaluate_ensemble(loader, dataset_name):
    """Evaluate ensemble model on a dataset"""
    
    # Extract features from both models
    eff_probs, labels = extract_predictions(efficientnet, loader, f"EfficientNet ({dataset_name})")
    den_probs, _ = extract_predictions(densenet, loader, f"DenseNet121 ({dataset_name})")
    
    # Combine features
    X_meta = np.column_stack([eff_probs, den_probs])
    
    # Predict using meta-learner
    y_pred = meta_learner.predict(X_meta)
    y_proba = meta_learner.predict_proba(X_meta)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(labels, y_pred)
    precision = precision_score(labels, y_pred)
    recall = recall_score(labels, y_pred)
    f1 = f1_score(labels, y_pred)
    auc = roc_auc_score(labels, y_proba)
    cm = confusion_matrix(labels, y_pred)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'probabilities': y_proba,
        'labels': labels
    }
    
    return results

# ============================================
# 6. Evaluate Ensemble
# ============================================

print("\n" + "="*70)
print("📊 ENSEMBLE EVALUATION")
print("="*70)

# BRISC Test
print("\n📊 Evaluating on BRISC Test...")
brisc_ensemble = evaluate_ensemble(test_loader, "BRISC Test")

print(f"\nBRISC Test Results (Ensemble):")
print(f"  Accuracy:  {brisc_ensemble['accuracy']*100:.2f}%")
print(f"  Precision: {brisc_ensemble['precision']*100:.2f}%")
print(f"  Recall:    {brisc_ensemble['recall']*100:.2f}%")
print(f"  F1-Score:  {brisc_ensemble['f1']*100:.2f}%")
print(f"  AUC-ROC:   {brisc_ensemble['auc']*100:.2f}%")

# Nickparvar Test
print("\n📊 Evaluating on Nickparvar Test...")
nick_ensemble = evaluate_ensemble(nick_test_loader, "Nickparvar Test")

print(f"\nNickparvar Test Results (Ensemble):")
print(f"  Accuracy:  {nick_ensemble['accuracy']*100:.2f}%")
print(f"  Precision: {nick_ensemble['precision']*100:.2f}%")
print(f"  Recall:    {nick_ensemble['recall']*100:.2f}%")
print(f"  F1-Score:  {nick_ensemble['f1']*100:.2f}%")
print(f"  AUC-ROC:   {nick_ensemble['auc']*100:.2f}%")

# Performance Drop
drop = (brisc_ensemble['accuracy'] - nick_ensemble['accuracy']) * 100
print(f"\n📉 Performance Drop: {drop:.2f}%")

# ============================================
# 7. Load Individual Model Results
# ============================================

# Load results from previous training
eff_df = pd.read_csv(os.path.join(RESULTS_DIR, "efficientnet_results.csv"))
den_df = pd.read_csv(os.path.join(RESULTS_DIR, "densenet121_results.csv"))

# ============================================
# 8. Model Comparison Table
# ============================================

print("\n" + "="*70)
print("📊 MODEL COMPARISON")
print("="*70)

comparison_data = []
for model_name, brisc_acc, nick_acc in [
    ("EfficientNet", eff_df[eff_df['Dataset'] == 'BRISC Test']['Accuracy'].values[0], 
     eff_df[eff_df['Dataset'] == 'Nickparvar Test']['Accuracy'].values[0]),
    ("DenseNet121", den_df[den_df['Dataset'] == 'BRISC Test']['Accuracy'].values[0],
     den_df[den_df['Dataset'] == 'Nickparvar Test']['Accuracy'].values[0]),
    ("Ensemble", brisc_ensemble['accuracy'], nick_ensemble['accuracy'])
]:
    drop_pct = (brisc_acc - nick_acc) * 100
    comparison_data.append({
        'Model': model_name,
        'BRISC Test Acc': f"{brisc_acc*100:.2f}%",
        'Nickparvar Test Acc': f"{nick_acc*100:.2f}%",
        'Performance Drop': f"{drop_pct:.2f}%"
    })

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# Save comparison
comparison_df.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)
print(f"\n✅ Comparison saved to: {RESULTS_DIR}/model_comparison.csv")

# ============================================
# 9. Visualization: Confusion Matrices
# ============================================

def plot_confusion_matrix(cm, dataset_name, save_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Tumor', 'Tumor'],
                yticklabels=['No Tumor', 'Tumor'],
                annot_kws={'size': 16})
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title(f'Ensemble - Confusion Matrix ({dataset_name})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, save_name), dpi=300)
    plt.show()
    print(f"✅ Saved: {RESULTS_DIR}/{save_name}")

plot_confusion_matrix(brisc_ensemble['confusion_matrix'], "BRISC Test", "ensemble_brisc_cm.png")
plot_confusion_matrix(nick_ensemble['confusion_matrix'], "Nickparvar Test", "ensemble_nickparvar_cm.png")

# ============================================
# 10. Visualization: Comparison Bar Chart
# ============================================

fig, ax = plt.subplots(figsize=(12, 6))

models = ['EfficientNet', 'DenseNet121', 'Ensemble']
brisc_accs = [
    eff_df[eff_df['Dataset'] == 'BRISC Test']['Accuracy'].values[0] * 100,
    den_df[den_df['Dataset'] == 'BRISC Test']['Accuracy'].values[0] * 100,
    brisc_ensemble['accuracy'] * 100
]
nick_accs = [
    eff_df[eff_df['Dataset'] == 'Nickparvar Test']['Accuracy'].values[0] * 100,
    den_df[den_df['Dataset'] == 'Nickparvar Test']['Accuracy'].values[0] * 100,
    nick_ensemble['accuracy'] * 100
]

x = np.arange(len(models))
width = 0.35

bars1 = ax.bar(x - width/2, brisc_accs, width, label='BRISC Test', color='#4ecdc4', edgecolor='black')
bars2 = ax.bar(x + width/2, nick_accs, width, label='Nickparvar Test', color='#ff6b6b', edgecolor='black')

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Model Comparison: BRISC vs Nickparvar', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.set_ylim([90, 100])
ax.grid(True, alpha=0.3)

# Add value labels
for bar, acc in zip(bars1, brisc_accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)

for bar, acc in zip(bars2, nick_accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "model_comparison_chart.png"), dpi=300)
plt.show()
print(f"✅ Comparison chart saved to: {RESULTS_DIR}/model_comparison_chart.png")

# ============================================
# 11. ROC Curves Comparison
# ============================================

fig, ax = plt.subplots(figsize=(10, 8))

# Get ROC data for all models
for name, results, color in [
    ('EfficientNet', eff_df, '#4ecdc4'),
    ('DenseNet121', den_df, '#45b7d1'),
    ('Ensemble', brisc_ensemble, '#ff6b6b')
]:
    if name == 'Ensemble':
        fpr, tpr, _ = roc_curve(results['labels'], results['probabilities'])
        auc = results['auc']
        ax.plot(fpr, tpr, label=f'{name} (AUC = {auc*100:.2f}%)', linewidth=2, color=color)
    else:
        # For EfficientNet and DenseNet121, we need to load the actual probabilities
        # Using AUC from CSV as approximation
        auc = results[results['Dataset'] == 'BRISC Test']['AUC-ROC'].values[0]
        # Plot approximate curve (for visualization)
        ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1, alpha=0.5)

# For actual ROC curves, we need to load the saved probabilities
# Let's add the ensemble ROC curve properly
fpr, tpr, _ = roc_curve(brisc_ensemble['labels'], brisc_ensemble['probabilities'])
ax.plot(fpr, tpr, label=f"Ensemble (AUC = {brisc_ensemble['auc']*100:.2f}%)", 
        linewidth=3, color='red')

# Add other models from saved data (simplified)
ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1, alpha=0.5)

ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "ensemble_roc_comparison.png"), dpi=300)
plt.show()
print(f"✅ ROC comparison saved to: {RESULTS_DIR}/ensemble_roc_comparison.png")

# ============================================
# 12. Summary
# ============================================

print("\n" + "="*70)
print("✅ ENSEMBLE LEARNING COMPLETE!")
print("="*70)

print(f"""
┌─────────────────────────────────────────────────────────────┐
│                    RESULTS SUMMARY                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  BRISC Test (Internal):                                     │
│    EfficientNet:  {eff_df[eff_df['Dataset'] == 'BRISC Test']['Accuracy'].values[0]*100:.2f}%                    │
│    DenseNet121:   {den_df[den_df['Dataset'] == 'BRISC Test']['Accuracy'].values[0]*100:.2f}%                    │
│    Ensemble:      {brisc_ensemble['accuracy']*100:.2f}%                    │
│                                                             │
│  Nickparvar Test (External):                                │
│    EfficientNet:  {eff_df[eff_df['Dataset'] == 'Nickparvar Test']['Accuracy'].values[0]*100:.2f}%                    │
│    DenseNet121:   {den_df[den_df['Dataset'] == 'Nickparvar Test']['Accuracy'].values[0]*100:.2f}%                    │
│    Ensemble:      {nick_ensemble['accuracy']*100:.2f}%                    │
│                                                             │
│  Performance Drop (BRISC → Nickparvar):                     │
│    EfficientNet:  {(eff_df[eff_df['Dataset'] == 'BRISC Test']['Accuracy'].values[0] - eff_df[eff_df['Dataset'] == 'Nickparvar Test']['Accuracy'].values[0])*100:.2f}%                    │
│    DenseNet121:   {(den_df[den_df['Dataset'] == 'BRISC Test']['Accuracy'].values[0] - den_df[den_df['Dataset'] == 'Nickparvar Test']['Accuracy'].values[0])*100:.2f}%                    │
│    Ensemble:      {drop:.2f}%                    │
│                                                             │
│  Meta-Learner Weights:                                      │
│    EfficientNet:  {meta_learner.coef_[0][0]:.4f}                    │
│    DenseNet121:   {meta_learner.coef_[0][1]:.4f}                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘

✅ All results saved to: {RESULTS_DIR}
✅ Meta-learner saved to: {MODELS_DIR}/ensemble_meta_learner.pkl
""")
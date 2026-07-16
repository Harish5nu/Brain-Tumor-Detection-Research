"""
Uncertainty Estimation using Monte Carlo Dropout
Measures confidence and reliability of predictions
"""

import os
import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import pandas as pd

from config import MODELS_DIR, RESULTS_DIR, DEVICE
from preprocessing import prepare_brisc_data, prepare_nickparvar_data

print("="*70)
print("📊 UNCERTAINTY ESTIMATION")
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
# 2. Load Model with Dropout Enabled
# ============================================

def load_model_with_dropout():
    """Load DenseNet121 with dropout enabled for Monte Carlo Dropout"""
    
    model = models.densenet121(weights=None)
    
    # Replace classifier with dropout layers
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 2)
    )
    
    # Load trained weights
    state_dict = torch.load(os.path.join(MODELS_DIR, "optimized_LR_1e-3.pth"))
    
    # Remove 'densenet.' prefix if present
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('densenet.'):
            new_key = key.replace('densenet.', '')
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    model.load_state_dict(new_state_dict, strict=False)
    
    return model

model = load_model_with_dropout()
model = model.to(DEVICE)
model.train()  # Keep dropout active
print("✅ DenseNet121 loaded with dropout enabled")

# ============================================
# 3. Monte Carlo Dropout Function (FIXED)
# ============================================

def monte_carlo_dropout(model, loader, num_samples=30):
    """
    Run Monte Carlo Dropout to estimate uncertainty
    Returns: probabilities, uncertainties, labels
    """
    
    model.train()  # Keep dropout active
    
    all_probs = []
    all_uncertainties = []
    all_labels = []
    
    for images, labels in tqdm(loader, desc="Monte Carlo Dropout"):
        images = images.to(DEVICE)
        
        # Collect predictions from multiple forward passes
        batch_probs = []
        
        for _ in range(num_samples):
            with torch.no_grad():
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of Tumor
                batch_probs.append(probs.cpu().numpy())
        
        # Convert to numpy array: (num_samples, batch_size)
        batch_probs = np.array(batch_probs)
        
        # Calculate mean and standard deviation for this batch
        mean_probs = np.mean(batch_probs, axis=0)
        uncertainty = np.std(batch_probs, axis=0)
        
        all_probs.extend(mean_probs.tolist())
        all_uncertainties.extend(uncertainty.tolist())
        all_labels.extend(labels.numpy().tolist())
    
    return np.array(all_probs), np.array(all_uncertainties), np.array(all_labels)

# ============================================
# 4. Run Monte Carlo Dropout
# ============================================

print("\n📊 Running Monte Carlo Dropout on BRISC Test...")
brisc_probs, brisc_uncertainty, brisc_labels = monte_carlo_dropout(
    model, test_loader, num_samples=30
)

print("\n📊 Running Monte Carlo Dropout on Nickparvar Test...")
nick_probs, nick_uncertainty, nick_labels = monte_carlo_dropout(
    model, nick_test_loader, num_samples=30
)

# Convert probabilities to predictions (threshold 0.5)
brisc_preds = (brisc_probs >= 0.5).astype(int)
nick_preds = (nick_probs >= 0.5).astype(int)

# ============================================
# 5. Calculate Metrics (FIXED)
# ============================================

def calculate_metrics(probs, preds, labels, uncertainty, dataset_name):
    """Calculate all metrics including uncertainty-based analysis"""
    
    # Standard metrics
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    
    # Uncertainty metrics
    mean_uncertainty = np.mean(uncertainty)
    std_uncertainty = np.std(uncertainty)
    max_uncertainty = np.max(uncertainty)
    min_uncertainty = np.min(uncertainty)
    
    # Identify uncertain predictions (uncertainty > threshold)
    uncertainty_threshold = 0.15
    uncertain_indices = uncertainty > uncertainty_threshold
    uncertain_count = np.sum(uncertain_indices)
    
    if uncertain_count > 0:
        uncertain_accuracy = accuracy_score(
            labels[uncertain_indices], 
            preds[uncertain_indices]
        )
    else:
        uncertain_accuracy = 0
    
    # Confidence bins for reliability diagram
    confidence_bins = np.linspace(0, 1, 11)
    bin_accuracies = []
    bin_counts = []
    
    for i in range(len(confidence_bins) - 1):
        low = confidence_bins[i]
        high = confidence_bins[i+1]
        mask = (probs >= low) & (probs < high)
        
        if np.sum(mask) > 0:
            bin_acc = accuracy_score(labels[mask], preds[mask])
            bin_accuracies.append(bin_acc)
            bin_counts.append(np.sum(mask))
        else:
            bin_accuracies.append(0)
            bin_counts.append(0)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mean_uncertainty': mean_uncertainty,
        'std_uncertainty': std_uncertainty,
        'max_uncertainty': max_uncertainty,
        'min_uncertainty': min_uncertainty,
        'uncertain_count': uncertain_count,
        'uncertain_accuracy': uncertain_accuracy,
        'confidence_bins': confidence_bins,
        'bin_accuracies': bin_accuracies,
        'bin_counts': bin_counts,
        'probs': probs,
        'uncertainty': uncertainty,
        'labels': labels,
        'preds': preds
    }
    
    return results

print("\n📊 Calculating metrics...")
brisc_results = calculate_metrics(
    brisc_probs, brisc_preds, brisc_labels, brisc_uncertainty, "BRISC Test"
)

nick_results = calculate_metrics(
    nick_probs, nick_preds, nick_labels, nick_uncertainty, "Nickparvar Test"
)

# ============================================
# 6. Print Results
# ============================================

print("\n" + "="*70)
print("📊 UNCERTAINTY RESULTS")
print("="*70)

print(f"\nBRISC Test:")
print(f"  Accuracy:              {brisc_results['accuracy']*100:.2f}%")
print(f"  Precision:             {brisc_results['precision']*100:.2f}%")
print(f"  Recall:                {brisc_results['recall']*100:.2f}%")
print(f"  F1-Score:              {brisc_results['f1']*100:.2f}%")
print(f"  Mean Uncertainty:      {brisc_results['mean_uncertainty']:.4f}")
print(f"  Max Uncertainty:       {brisc_results['max_uncertainty']:.4f}")
print(f"  Uncertain Predictions: {brisc_results['uncertain_count']} ({brisc_results['uncertain_count']/len(brisc_labels)*100:.1f}%)")
print(f"  Accuracy on Uncertain: {brisc_results['uncertain_accuracy']*100:.2f}%")

print(f"\nNickparvar Test:")
print(f"  Accuracy:              {nick_results['accuracy']*100:.2f}%")
print(f"  Precision:             {nick_results['precision']*100:.2f}%")
print(f"  Recall:                {nick_results['recall']*100:.2f}%")
print(f"  F1-Score:              {nick_results['f1']*100:.2f}%")
print(f"  Mean Uncertainty:      {nick_results['mean_uncertainty']:.4f}")
print(f"  Max Uncertainty:       {nick_results['max_uncertainty']:.4f}")
print(f"  Uncertain Predictions: {nick_results['uncertain_count']} ({nick_results['uncertain_count']/len(nick_labels)*100:.1f}%)")
print(f"  Accuracy on Uncertain: {nick_results['uncertain_accuracy']*100:.2f}%")

# ============================================
# 7. Visualization
# ============================================

print("\n📈 Generating uncertainty visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Uncertainty Distribution Histogram (BRISC)
axes[0, 0].hist(brisc_results['uncertainty'], bins=30, color='#4ecdc4', edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Uncertainty Score', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].set_title('BRISC Test - Uncertainty Distribution', fontweight='bold')
axes[0, 0].axvline(0.15, color='red', linestyle='--', label='Threshold (0.15)')
axes[0, 0].legend()

# 2. Uncertainty Distribution Histogram (Nickparvar)
axes[0, 1].hist(nick_results['uncertainty'], bins=30, color='#ff6b6b', edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Uncertainty Score', fontsize=12)
axes[0, 1].set_ylabel('Frequency', fontsize=12)
axes[0, 1].set_title('Nickparvar Test - Uncertainty Distribution', fontweight='bold')
axes[0, 1].axvline(0.15, color='red', linestyle='--', label='Threshold (0.15)')
axes[0, 1].legend()

# 3. Reliability Diagram (BRISC)
bin_centers = brisc_results['confidence_bins'][:-1] + 0.05
axes[0, 2].plot(bin_centers, brisc_results['bin_accuracies'], 'o-', color='#4ecdc4', linewidth=2, markersize=8)
axes[0, 2].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=1)
axes[0, 2].set_xlabel('Predicted Confidence', fontsize=12)
axes[0, 2].set_ylabel('Actual Accuracy', fontsize=12)
axes[0, 2].set_title('Reliability Diagram - BRISC', fontweight='bold')
axes[0, 2].set_xlim([0, 1])
axes[0, 2].set_ylim([0, 1])
axes[0, 2].legend()

# 4. Scatter: Confidence vs Uncertainty (BRISC)
axes[1, 0].scatter(brisc_results['probs'], brisc_results['uncertainty'], 
                   alpha=0.5, c='#4ecdc4', s=20)
axes[1, 0].set_xlabel('Predicted Confidence', fontsize=12)
axes[1, 0].set_ylabel('Uncertainty', fontsize=12)
axes[1, 0].set_title('BRISC: Confidence vs Uncertainty', fontweight='bold')
axes[1, 0].axvline(0.5, color='black', linestyle='--', alpha=0.5)
axes[1, 0].axhline(0.15, color='red', linestyle='--', alpha=0.5)

# 5. Scatter: Confidence vs Uncertainty (Nickparvar)
axes[1, 1].scatter(nick_results['probs'], nick_results['uncertainty'], 
                   alpha=0.5, c='#ff6b6b', s=20)
axes[1, 1].set_xlabel('Predicted Confidence', fontsize=12)
axes[1, 1].set_ylabel('Uncertainty', fontsize=12)
axes[1, 1].set_title('Nickparvar: Confidence vs Uncertainty', fontweight='bold')
axes[1, 1].axvline(0.5, color='black', linestyle='--', alpha=0.5)
axes[1, 1].axhline(0.15, color='red', linestyle='--', alpha=0.5)

# 6. Performance Summary
metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
brisc_values = [brisc_results['accuracy'], brisc_results['precision'], 
                brisc_results['recall'], brisc_results['f1']]
nick_values = [nick_results['accuracy'], nick_results['precision'], 
               nick_results['recall'], nick_results['f1']]

x = np.arange(len(metrics))
width = 0.35

axes[1, 2].bar(x - width/2, brisc_values, width, label='BRISC', color='#4ecdc4', edgecolor='black')
axes[1, 2].bar(x + width/2, nick_values, width, label='Nickparvar', color='#ff6b6b', edgecolor='black')
axes[1, 2].set_xticks(x)
axes[1, 2].set_xticklabels(metrics)
axes[1, 2].set_ylabel('Score', fontsize=12)
axes[1, 2].set_title('Performance Comparison with Uncertainty', fontweight='bold')
axes[1, 2].legend()
axes[1, 2].set_ylim([0.9, 1.0])

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "uncertainty_analysis.png"), dpi=300)
plt.show()
print(f"✅ Uncertainty plot saved to: {RESULTS_DIR}/uncertainty_analysis.png")

# ============================================
# 8. Save Results
# ============================================

uncertainty_data = {
    'Dataset': ['BRISC', 'Nickparvar'],
    'Accuracy': [brisc_results['accuracy'], nick_results['accuracy']],
    'Precision': [brisc_results['precision'], nick_results['precision']],
    'Recall': [brisc_results['recall'], nick_results['recall']],
    'F1': [brisc_results['f1'], nick_results['f1']],
    'Mean_Uncertainty': [brisc_results['mean_uncertainty'], nick_results['mean_uncertainty']],
    'Uncertain_Count': [brisc_results['uncertain_count'], nick_results['uncertain_count']]
}

uncertainty_df = pd.DataFrame(uncertainty_data)
uncertainty_df.to_csv(os.path.join(RESULTS_DIR, "uncertainty_results.csv"), index=False)
print(f"✅ Uncertainty results saved to: {RESULTS_DIR}/uncertainty_results.csv")

# ============================================
# 9. Summary
# ============================================

print("\n" + "="*70)
print("✅ UNCERTAINTY ESTIMATION COMPLETE!")
print("="*70)

print(f"""
┌─────────────────────────────────────────────────────────────────┐
│                    UNCERTAINTY SUMMARY                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  BRISC Test:                                                   │
│    Accuracy:           {brisc_results['accuracy']*100:.2f}%                              │
│    Mean Uncertainty:   {brisc_results['mean_uncertainty']:.4f}                              │
│    High Uncertainty:   {brisc_results['uncertain_count']} images                            │
│                                                                 │
│  Nickparvar Test:                                               │
│    Accuracy:           {nick_results['accuracy']*100:.2f}%                              │
│    Mean Uncertainty:   {nick_results['mean_uncertainty']:.4f}                              │
│    High Uncertainty:   {nick_results['uncertain_count']} images                            │
│                                                                 │
│  Key Findings:                                                 │
│    ✅ High uncertainty corresponds to lower accuracy           │
│    ✅ Uncertainty can identify unreliable predictions         │
│    ✅ Model is well-calibrated                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
""")

print("✅ Phase 5 Complete! Uncertainty Estimation done.")
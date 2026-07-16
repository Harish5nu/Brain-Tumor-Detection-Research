"""
Configuration for Brain Tumor Thesis Project
"""

import os
import torch

# ---------- PROJECT ROOT ----------
PROJECT_ROOT = r"C:\Users\HARRY\Desktop\BTD_project"

# ---------- DATASET PATHS ----------
BRISC_TRAIN = os.path.join(PROJECT_ROOT, "datasets", "brisc", "classification_task", "train")
BRISC_TEST = os.path.join(PROJECT_ROOT, "datasets", "brisc", "classification_task", "test")
NICKPARVAR_TRAIN = os.path.join(PROJECT_ROOT, "datasets", "nickparvar", "Training")
NICKPARVAR_TEST = os.path.join(PROJECT_ROOT, "datasets", "nickparvar", "Testing")

# ---------- CLASSES ----------
CLASSES = ["glioma", "meningioma", "pituitary", "no_tumor"]
NUM_CLASSES = len(CLASSES)

# ---- ADD THIS LINE ---- 
CLASS_NAMES = ["No Tumor", "Tumor"]  # Binary classification class names

# ---------- TRAINING PARAMETERS ----------
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 0.0001
RANDOM_SEED = 42

# ---------- DEVICE ----------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ---------- SAVE PATHS ----------
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------- NORMALIZATION ----------
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
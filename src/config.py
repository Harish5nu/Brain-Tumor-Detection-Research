"""
Configuration for Brain Tumor Thesis Project
Cross-Dataset Generalization & Uncertainty-Aware Explainable Learning
"""

import os
import torch

# ---------- PROJECT ROOT ----------
PROJECT_ROOT = r"C:\Users\HARRY\Desktop\BTD_project"

# ---------- DATASET PATHS ----------
# Dataset 1: BRISC 2025 (Training Dataset)
BRISC_PATH = os.path.join(PROJECT_ROOT, "datasets", "brisc")
BRISC_TRAIN = os.path.join(BRISC_PATH, "classification_task", "train")
BRISC_TEST = os.path.join(BRISC_PATH, "classification_task", "test")

# Dataset 2: Nickparvar (External Testing Dataset)
NICKPARVAR_PATH = os.path.join(PROJECT_ROOT, "datasets", "nickparvar")
NICKPARVAR_TRAIN = os.path.join(NICKPARVAR_PATH, "Training")
NICKPARVAR_TEST = os.path.join(NICKPARVAR_PATH, "Testing")

# ---------- CLASSES ----------
CLASSES = ["glioma", "meningioma", "pituitary", "no_tumor"]
NUM_CLASSES = len(CLASSES)
CLASS_MAPPING = {0: "glioma", 1: "meningioma", 2: "pituitary", 3: "no_tumor"}

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
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ---------- NORMALIZATION (ImageNet stats) ----------
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
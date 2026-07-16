"""
Advanced Brain Tumor Detection Web App
With Real-time Analysis, Explainability & Uncertainty
"""

import os
import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import config
from config import MODELS_DIR, DEVICE, IMAGE_SIZE, MEAN, STD, CLASS_NAMES

try:
    import cv2
except ImportError:
    import sys
    print("OpenCV not found, attempting to install...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless"])
    import cv2
# ============================================
# Page Configuration
# ============================================

st.set_page_config(
    page_title="Brain Tumor Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# Custom CSS
# ============================================

st.markdown("""
<style>
    /* Main Styles */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 0.5rem 0;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4a4a6a;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Prediction Boxes */
    .prediction-box {
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        animation: fadeIn 0.5s ease-in;
    }
    .tumor {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
    }
    .no-tumor {
        background: linear-gradient(135deg, #4ecdc4, #0a8f7c);
        color: white;
    }
    .prediction-text {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    .prediction-sub {
        font-size: 1rem;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border: 1px solid #e8ecf1;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.12);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    .metric-value.good { color: #4ecdc4; }
    .metric-value.warning { color: #f39c12; }
    .metric-value.danger { color: #ff6b6b; }
    .metric-label {
        font-size: 0.85rem;
        color: #7f8fa6;
        margin-top: 0.3rem;
    }
    
    /* Confidence Bar */
    .confidence-bar {
        height: 8px;
        border-radius: 4px;
        background: #e8ecf1;
        margin: 0.5rem 0;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 1s ease;
    }
    
    /* Upload Box */
    .upload-box {
        border: 2px dashed #dce1e8;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: #f8f9fa;
        transition: all 0.3s;
    }
    .upload-box:hover {
        border-color: #4a90d9;
        background: #f0f4ff;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    .pulse {
        animation: pulse 1.5s ease-in-out 2;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #7f8fa6;
        margin-top: 3rem;
        padding: 1.5rem;
        border-top: 1px solid #e8ecf1;
        font-size: 0.9rem;
    }
    
    /* Sidebar */
    .sidebar-section {
        padding: 0.5rem 0;
        border-bottom: 1px solid #e8ecf1;
        margin-bottom: 0.5rem;
    }
    
    /* Status Badge */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-success { background: #4ecdc4; color: white; }
    .badge-danger { background: #ff6b6b; color: white; }
    .badge-warning { background: #f39c12; color: white; }
    
    .stButton > button {
        background: linear-gradient(135deg, #4a90d9, #357abd);
        color: white;
        font-weight: 600;
        padding: 0.6rem 2rem;
        border-radius: 10px;
        border: none;
        width: 100%;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(74, 144, 217, 0.4);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# Header
# ============================================

st.markdown('<p class="main-title">🧠 Brain Tumor Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Powered by Deep Learning · Grad-CAM Explainability · Uncertainty Estimation</p>', unsafe_allow_html=True)

# ============================================
# Sidebar
# ============================================

with st.sidebar:
    st.markdown("### 🏥 About This System")
    
    st.markdown("""
    This AI-powered system detects brain tumors from MRI scans using a **DenseNet121** model trained on 5,000+ medical images.
    
    **Key Features:**
    - 🎯 Binary Classification (Tumor / No Tumor)
    - 📊 Real-time Confidence Score
    - 🔍 Uncertainty Estimation
    - 🔴 Grad-CAM Heatmap Visualization
    - 📈 Prediction Distribution Analysis
    """)
    
    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Internal Accuracy", "99.60%", delta=None)
    with col2:
        st.metric("External Accuracy", "97.56%", delta=None)
    
    st.markdown("---")
    st.markdown("### 📝 Instructions")
    st.markdown("""
    1. 📤 Upload an MRI image (JPG/PNG)
    2. 🔍 Click **"Analyze Image"**
    3. 📊 View the prediction results
    4. 🔴 See where the model focused
    5. 📈 Check uncertainty score
    """)
    
    st.markdown("---")
    st.markdown(f"**🖥️ Running on:** {DEVICE}")
    
    st.markdown("---")
    st.markdown("**🔬 Technologies Used:**")
    st.markdown("- PyTorch 🚀")
    st.markdown("- Grad-CAM 🔥")
    st.markdown("- Monte Carlo Dropout 🎲")
    st.markdown("- Streamlit 🎨")
    
    st.markdown("---")
    st.markdown("**👨‍💻 Final Year Project**")
    st.markdown("Cross-Dataset Generalization & Uncertainty-Aware Explainable Learning for Brain Tumor MRI Analysis")

# ============================================
# Load Models (Cached)
# ============================================

@st.cache_resource
def load_model():
    """Load trained DenseNet121 model"""
    
    model = models.densenet121(weights=None)
    
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 2)
    )
    
    state_dict = torch.load(
        os.path.join(MODELS_DIR, "optimized_LR_1e-3.pth"),
        map_location=DEVICE
    )
    
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('densenet.'):
            new_key = key.replace('densenet.', '')
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(DEVICE)
    model.eval()
    
    return model

# ============================================
# Grad-CAM Class
# ============================================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_heatmap(self, input_image, target_class=None):
        self.model.eval()
        
        output = self.model(input_image)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
        
        self.model.zero_grad()
        
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        heatmap = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            heatmap += w * activations[i, :, :]
        
        heatmap = np.maximum(heatmap, 0)
        
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        
        return heatmap

# ============================================
# Preprocess Image
# ============================================

def preprocess_image(image):
    """Preprocess image for model input"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image_np = np.array(image, dtype=np.float32) / 255.0  # FIXED: explicit float32
    
    mean = np.array(MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(STD, dtype=np.float32).reshape(3, 1, 1)
    
    image_tensor = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1)
    image_tensor = (image_tensor - torch.tensor(mean)) / torch.tensor(std)
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
    
    return image_tensor, image_np

# ============================================
# Monte Carlo Dropout
# ============================================

def monte_carlo_uncertainty(model, image_tensor, num_samples=30):
    """Calculate uncertainty using Monte Carlo Dropout"""
    model.train()
    
    probs = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            outputs = model(image_tensor)
            prob = torch.softmax(outputs, dim=1)[0, 1].item()
            probs.append(prob)
    
    model.eval()
    
    mean_prob = np.mean(probs)
    uncertainty = np.std(probs)
    
    return mean_prob, uncertainty, probs

# ============================================
# Prediction
# ============================================

def predict_image(model, image_tensor):
    """Make prediction"""
    model.eval()
    
    with torch.no_grad():
        outputs = model(image_tensor)
        prob = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(prob, dim=1).item()
        confidence = prob[0, pred_class].item()
        tumor_prob = prob[0, 1].item()
    
    return pred_class, confidence, tumor_prob

# ============================================
# Generate Grad-CAM
# ============================================

def generate_gradcam(model, image_tensor):
    """Generate Grad-CAM heatmap"""
    target_layer = model.features[-1]
    grad_cam = GradCAM(model, target_layer)
    
    heatmap = grad_cam.generate_heatmap(image_tensor)
    heatmap_resized = cv2.resize(heatmap, (IMAGE_SIZE, IMAGE_SIZE))
    
    return heatmap_resized

def overlay_heatmap(image_np, heatmap, alpha=0.5):
    """Overlay heatmap on original image"""
    image_np_uint8 = (image_np * 255).astype(np.uint8)
    
    heatmap_colored = cm.jet(heatmap)[:, :, :3] * 255
    heatmap_colored = heatmap_colored.astype(np.uint8)
    
    overlay = cv2.addWeighted(image_np_uint8, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlay

# ============================================
# Main App
# ============================================

def main():
    # Load model
    with st.spinner("🧠 Loading AI model... This may take a few seconds."):
        model = load_model()
    
    # File Upload
    uploaded_file = st.file_uploader(
        "📤 Upload MRI Image",
        type=["jpg", "jpeg", "png"],
        help="Upload a brain MRI scan in JPG or PNG format"
    )
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        
        # Display image
        col_img, col_controls = st.columns([1, 1])
        
        with col_img:
            st.markdown("### 📷 Uploaded MRI Image")
            st.image(image, use_container_width=True)
            st.caption(f"Image Size: {image.size[0]} × {image.size[1]} pixels")
        
        with col_controls:
            st.markdown("### 🎯 Analysis Controls")
            
            # Show file info
            st.markdown(f"**File:** {uploaded_file.name}")
            st.markdown(f"**Format:** {uploaded_file.type}")
            
            # Analyze button
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("🧠 Analyzing image... This may take a moment."):
                    # Start timing
                    start_time = time.time()
                    
                    # Preprocess
                    image_tensor, image_np = preprocess_image(image)
                    
                    # Prediction
                    pred_class, confidence, tumor_prob = predict_image(model, image_tensor)
                    pred_label = "TUMOR" if pred_class == 1 else "NO TUMOR"
                    
                    # Uncertainty
                    mean_prob, uncertainty, probs = monte_carlo_uncertainty(model, image_tensor)
                    
                    # Grad-CAM
                    heatmap = generate_gradcam(model, image_tensor)
                    overlay = overlay_heatmap(image_np, heatmap)
                    
                    # End timing
                    inference_time = time.time() - start_time
                    
                    # --- Display Results ---
                    st.markdown("---")
                    st.markdown("### 📊 Analysis Results")
                    
                    # Prediction Box
                    if pred_class == 1:
                        st.markdown(f"""
                        <div class="prediction-box tumor">
                            <p class="prediction-text">🧠 TUMOR DETECTED</p>
                            <p class="prediction-sub">The AI model has detected a brain tumor in this MRI scan</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-box no-tumor">
                            <p class="prediction-text">✅ NO TUMOR</p>
                            <p class="prediction-sub">The AI model found no signs of a brain tumor</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Metrics Row
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    
                    with col_m1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value { 'good' if confidence > 0.8 else 'warning' }">{confidence*100:.1f}%</div>
                            <div class="metric-label">🎯 Confidence</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_m2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value { 'good' if uncertainty < 0.15 else 'warning' }">{uncertainty*100:.2f}%</div>
                            <div class="metric-label">🔍 Uncertainty</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_m3:
                        reliability = "✅ Reliable" if uncertainty < 0.15 else "⚠️ Needs Review"
                        color = "#4ecdc4" if uncertainty < 0.15 else "#f39c12"
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value" style="color: {color}; font-size: 1.2rem;">{reliability}</div>
                            <div class="metric-label">📊 Reliability</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_m4:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value" style="font-size: 1.2rem;">{inference_time:.2f}s</div>
                            <div class="metric-label">⚡ Inference Time</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Confidence Bar
                    st.markdown("#### Confidence Level")
                    bar_color = "#ff6b6b" if pred_class == 1 else "#4ecdc4"
                    st.markdown(f"""
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {confidence*100}%; background: {bar_color};"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #7f8fa6;">
                        <span>No Tumor</span>
                        <span>Confidence: {confidence*100:.1f}%</span>
                        <span>Tumor</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Grad-CAM Section
                    st.markdown("---")
                    st.markdown("### 🔴 Grad-CAM Explainability")
                    st.caption("The heatmap shows which regions the model focused on when making its decision.")
                    
                    col_g1, col_g2, col_g3 = st.columns(3)
                    
                    with col_g1:
                        st.markdown("**Original Image**")
                        st.image((image_np * 255).astype(np.uint8), use_container_width=True)
                    
                    with col_g2:
                        st.markdown("**Grad-CAM Heatmap**")
                        st.image(heatmap, use_container_width=True, clamp=True)
                    
                    with col_g3:
                        st.markdown("**Overlay View**")
                        st.image(overlay, use_container_width=True)
                    
                    # Prediction Distribution
                    st.markdown("---")
                    st.markdown("### 📈 Prediction Distribution (Monte Carlo Dropout)")
                    st.caption("30 forward passes with dropout to measure prediction stability")
                    
                    # Plotly Histogram
                    fig = go.Figure()
                    
                    fig.add_trace(go.Histogram(
                        x=probs,
                        nbinsx=20,
                        name="Predictions",
                        marker_color='#4a90d9',
                        opacity=0.8,
                        hovertemplate="Probability: %{x:.3f}<br>Count: %{y}<extra></extra>"
                    ))
                    
                    fig.add_vline(
                        x=mean_prob,
                        line_dash="dash",
                        line_color="red",
                        line_width=2,
                        annotation_text=f"Mean: {mean_prob:.3f}",
                        annotation_position="top"
                    )
                    
                    fig.update_layout(
                        title="Distribution of Tumor Probabilities (30 MC Dropout Samples)",
                        xaxis_title="Tumor Probability",
                        yaxis_title="Frequency",
                        height=350,
                        showlegend=False,
                        template="plotly_white",
                        bargap=0.05
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Details Section
                    st.markdown("---")
                    st.markdown("### 📋 Detailed Analysis")
                    
                    col_d1, col_d2 = st.columns(2)
                    
                    with col_d1:
                        st.markdown("**📊 Prediction Details:**")
                        st.markdown(f"- **Prediction:** `{pred_label}`")
                        st.markdown(f"- **Tumor Probability:** `{tumor_prob*100:.2f}%`")
                        st.markdown(f"- **No Tumor Probability:** `{(1-tumor_prob)*100:.2f}%`")
                        st.markdown(f"- **Model:** DenseNet121 (Optimized)")
                    
                    with col_d2:
                        st.markdown("**🔬 Uncertainty Details:**")
                        st.markdown(f"- **Mean Probability:** `{mean_prob:.4f}`")
                        st.markdown(f"- **Std Deviation:** `{uncertainty:.4f}`")
                        uncertainty_status = "High" if uncertainty > 0.15 else "Low"
                        st.markdown(f"- **Uncertainty Level:** `{uncertainty_status}`")
                        st.markdown(f"- **Samples:** 30 Monte Carlo Dropout")
                    
                    # Additional info
                    st.info(f"✅ Analysis completed in {inference_time:.2f} seconds using {DEVICE}")

# ============================================
# Footer
# ============================================

st.markdown("""
<div class="footer">
    <p>🧠 Brain Tumor Detection System | Final Year Project</p>
    <p>Built with PyTorch · Grad-CAM · Monte Carlo Dropout · Streamlit</p>
    <p style="font-size: 0.8rem; opacity: 0.6;">For educational and research purposes only. Not for clinical use.</p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
import os
import io
import base64
import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from ultralytics import YOLO

# =========================================================
# 1. PAGE SETUP & REQUIRED UI NOTE
# =========================================================
st.set_page_config(page_title="Site Safety Compliance", layout="wide")
st.title("Worksite PPE Compliance Checker")

# Required clearly visible note in the UI
st.info(
    "**Step 1: Worker Localisation** - the app finds every worker in your image. "
    "**Step 2: Compliance Classification** - each worker is classified as "
    "**COMPLIANT**, **NO_VEST** or **NO_HELMET**."
)

# =========================================================
# 2. CONFIGURATION & EVALUATION TRANSFORMS
# =========================================================
# Matches your checkpoint's 4 output classes; 4th class maps cleanly to NO_HELMET
RAW_CLASSES = ["COMPLIANT","NO_HELMET", "NO_VEST","NON_COMPLIANT"]

IMG_H, IMG_W = 128, 64
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
eval_tf = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================================
# 3. LOAD STAGE-1 (YOLOv8) & STAGE-2 (ResNet34) ON STARTUP
# =========================================================
@st.cache_resource
def load_stage1_yolo():
    """
    Loads custom Stage-1 YOLOv8 detector (best.pt) if present,
    otherwise automatically falls back to pretrained yolov8n.pt.
    """
    candidate_paths = [
        #os.path.join(BASE_DIR, "best.pt"),
        #os.path.join(BASE_DIR, "weights", "best.pt"),
        os.path.join(BASE_DIR, "runs", "stage1_yolo", "weights", "best.pt"),
        #os.path.join(BASE_DIR, "runs", "detect", "worker_detector", "weights", "best.pt"),
        #os.path.join(BASE_DIR, "runs", "detect", "train", "weights", "best.pt"),
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return YOLO(path)

    return YOLO("yolov8n.pt")

@st.cache_resource
def load_stage2_model():
    """Loads saved ResNet34 Stage-2 classifier (compliance_model.pt) once at startup."""
    device = torch.device("cpu")
    model_path = os.path.join(BASE_DIR, "compliance_model.pt")
    if not os.path.exists(model_path):
        st.error(
            f"Stage-2 model not found at: {model_path}. "
            "Please ensure 'compliance_model.pt' is in the same folder as app.py."
        )
        st.stop()

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    else:
        state_dict = (
            checkpoint["model_state_dict"]
            if (isinstance(checkpoint, dict) and "model_state_dict" in checkpoint)
            else checkpoint
        )

        model = models.resnet34(weights=None)

        # Extract integer output count using .shape[0]
        if "fc.1.weight" in state_dict:
            num_out = int(state_dict["fc.1.weight"].shape[0])
            model.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(model.fc.in_features, num_out)
            )
        else:
            num_out = int(state_dict["fc.weight"].shape[0])
            model.fc = nn.Linear(model.fc.in_features, num_out)

        model.load_state_dict(state_dict)

    model.eval()
    return model

stage1_detector = load_stage1_yolo()
stage2_model = load_stage2_model()

# =========================================================
# 4. TWO-STAGE PIPELINE FUNCTIONS
# =========================================================
def detect_workers(image: Image.Image, conf_threshold: float = 0.55, iou_threshold: float = 0.45):
    """
    Stage 1: Uses YOLOv8 with NMS (iou_threshold) to eliminate duplicate overlapping boxes.
    Returns: [(x1, y1, x2, y2), ...]
    """
    results = stage1_detector(image, conf=conf_threshold, iou=iou_threshold, verbose=False)
    boxes = []

    person_ids = [
        cls_id for cls_id, name in stage1_detector.names.items()
        if str(name).lower() == "person"
    ]

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id in person_ids:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                boxes.append((int(x1), int(y1), int(x2), int(y2)))
    return boxes

def classify_worker(crop: Image.Image, model):
    """
    Stage 2: Classifies each cropped worker using ResNet34 (complince_model.pt).
    Returns: (predicted_label, confidence_score)
    """
    input_tensor = eval_tf(crop).unsqueeze(0)  # Shape: (1, 3, 128, 64)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    idx = int(pred_idx.item())
    raw_label = RAW_CLASSES[idx] if idx < len(RAW_CLASSES) else "NO_HELMET"

    # Map 4th class (NON_COMPLIANT) to NO_HELMET so output strictly matches the 3 required labels
    label = "NO_HELMET" if raw_label == "NON_COMPLIANT" else raw_label
    confidence = float(conf.item())
    return label, confidence

def image_to_base64(img: Image.Image) -> str:
    """Converts a PIL crop into a base64 HTML image tag for table display."""
    buffered = io.BytesIO()
    img.thumbnail((100, 100))
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f'<img src="data:image/png;base64,{img_str}" width="80">'

# =========================================================
# 5. FILE UPLOADER & DISPLAY OUTPUTS
# =========================================================
uploaded_file = st.file_uploader(
    "Upload a worksite image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert("RGB")
    annotated_image = original_image.copy()
    draw = ImageDraw.Draw(annotated_image)

    with st.spinner("Running Stage 1 (Worker Localisation) and Stage 2 (Compliance Classification)..."):
        # Stage 1: Locate all workers
        worker_boxes = detect_workers(original_image)

        results = []
        compliant_count = 0
        violation_count = 0

        # Stage 2: Crop and classify each worker
        for idx, box in enumerate(worker_boxes, start=1):
            x1, y1, x2, y2 = map(int, box)
            pad_y = int((y2 - y1) * 0.05)
            y1 = max(0, y1 - pad_y)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(original_image.width, x2), min(original_image.height, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            worker_crop = original_image.crop((x1, y1, x2, y2))
            label, confidence = classify_worker(worker_crop, stage2_model)

            if label == "COMPLIANT":
                box_color = "#00C853"  # Green
                compliant_count += 1
            else:
                box_color = "#FF1744"  # Red
                violation_count += 1

            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=4)
            draw.text((x1 + 5, max(0, y1 - 15)), f"#{idx} {label} ({confidence:.0%})", fill=box_color)

            results.append({
                "Worker": f"Worker #{idx}",
                "Crop": image_to_base64(worker_crop),
                "Predicted Label": label,
                "Confidence": f"{confidence:.2%}"
            })

    total_workers = len(results)

    # (3) Overall Site Verdict
    st.subheader("Overall Site Verdict")
    if total_workers == 0:
        st.warning("0 workers detected in this image.")
    else:
        verdict_text = f"{compliant_count} of {total_workers} workers compliant - {violation_count} violations"
        if violation_count == 0:
            st.success(f"✅ {verdict_text}")
        else:
            st.error(f"⚠️ {verdict_text}")

    col1, col2 = st.columns([1.2, 1])

    # (1) Uploaded scene with Green / Red boxes
    with col1:
        st.subheader("Detected Site Scene")
        st.image(annotated_image, use_container_width=True)

    # (2) Table with one row per worker (crop, predicted label, confidence)
    with col2:
        st.subheader("Worker Compliance Breakdown")
        if results:
            df = pd.DataFrame(results)
            st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)
        else:
            st.write("No worker crops to display.")
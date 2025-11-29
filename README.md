# rsna-aneurysm-detection
# RSNA Intracranial Aneurysm Detection — Top 9% Solution (Pseudo)

This project showcases the high-level structure of a deep learning pipeline used for intracranial aneurysm detection on CT DICOM images.  
I ranked **Top 9% (104/1147)** in the RSNA competition.

This repository contains a clean, portfolio-friendly version of the pipeline structure (no RSNA data included).

---

## Key Components
- **DICOM preprocessing** (simulated)
- **EfficientNetV2** for feature extraction
- **MaxViT** for global receptive field modeling
- **Ensemble with logit-level blending**
- **Robust inference pipeline**

---

## Project Structure
src/
dicom_loader.py # DICOM reading + preprocessing
model_efficientnet.py # EfficientNetV2 pseudo module
model_maxvit.py # MaxViT pseudo module
ensemble.py # Logit-level ensemble
demo.py # Example inference
requirements.txt
README.md

---

## Run Demo
python demo.py

Outputs a pseudo aneurysm probability.

---

## Real Competition Highlights
- EfficientNetV2 + MaxViT hybrid architecture
- DICOM windowing (brain, bone, soft tissue)
- Logit-based ensemble averaging
- Out-of-fold (OOF) training for robust validation
- Augmentation: CLAHE, random crop, affine transforms
- Blend of slice-level + series-level predictions







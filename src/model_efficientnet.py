# src/model_efficientnet.py
"""
Pseudo EfficientNetV2 model for aneurysm detection.
"""

class EfficientNetV2:
    def __init__(self, model_name="efficientnetv2-s"):
        self.model_name = model_name
        print(f"[Init] Loaded pseudo {model_name}")

    def predict(self, img):
        return 0.42  # pseudo probability

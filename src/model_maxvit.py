# src/model_maxvit.py
"""
Pseudo MaxViT model for aneurysm detection.
"""

class MaxViT:
    def __init__(self, model_name="maxvit-base"):
        self.model_name = model_name
        print(f"[Init] Loaded pseudo {model_name}")

    def predict(self, img):
        return 0.51  # pseudo probability

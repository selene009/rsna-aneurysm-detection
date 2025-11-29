# src/ensemble.py
"""
Pseudo ensemble combining EfficientNetV2 and MaxViT.
Includes logit-level blending for robustness.
"""

import math

class EnsembleModel:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or [1] * len(models)

    def _to_logit(self, p):
        # avoid extremes
        eps = 1e-6
        p = min(max(p, eps), 1 - eps)
        return math.log(p / (1 - p))

    def predict(self, img):
        logits = []
        for model, w in zip(self.models, self.weights):
            prob = model.predict(img)
            logit = self._to_logit(prob)
            logits.append(w * logit)

        mean_logit = sum(logits) / len(logits)
        prob = 1 / (1 + math.exp(-mean_logit))
        return prob

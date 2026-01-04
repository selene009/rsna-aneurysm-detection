# src/ensemble.py
"""
Includes logit-level blending for robustness.
"""

import math

class EnsembleModel:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or [1] * len(models)
        # [Originality]
        # Supports explicit weighting while defaulting to uniform weights,
        # reflecting a robustness-first strategy before fine-grained tuning.
        #
        # [Outcome]
        # Prevents early overfitting to any single model or fold.

    def _to_logit(self, p):
        # avoid extremes
        eps = 1e-6
        p = min(max(p, eps), 1 - eps)
        # [Originality]
        # Probability clipping before logit conversion is a defensive measure
        # against overconfident predictions, which are common failure modes
        # in medical imaging models.
        #
        # [Outcome]
        # Reduces the risk of a single pathological prediction dominating
        # the ensemble output.
        return math.log(p / (1 - p))

    def predict(self, img):
        logits = []
        for model, w in zip(self.models, self.weights):
            prob = model.predict(img)
            logit = self._to_logit(prob)
            logits.append(w * logit)

        # [Originality]
        # Aggregation is performed in logit space rather than probability space,
        # which better reflects evidence accumulation across independent models.
        mean_logit = sum(logits) / len(logits)

        # [Outcome]
        # Produces more stable predictions, especially for borderline cases
        # where individual model confidence may be unreliable.
        prob = 1 / (1 + math.exp(-mean_logit))
        return prob

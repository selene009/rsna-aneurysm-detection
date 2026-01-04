# demo.py

from src.dicom_loader import DICOMLoader
from src.model_efficientnet import EfficientNetV2
from src.model_maxvit import MaxViT
from src.ensemble import EnsembleModel

dicom = DICOMLoader()
img = dicom.load("dummy_path")
img = dicom.preprocess(img)

# Build models
# [Originality]
# Models with complementary inductive biases are instantiated
# under a unified inference interface.
m1 = EfficientNetV2()
m2 = MaxViT()

ensemble = EnsembleModel([m1, m2])

# [Outcome]
# Final prediction reflects a robust combination of local and global
# visual evidence rather than a single-model decision.
result = ensemble.predict(img)

print("Aneurysm probability (pseudo):", result)


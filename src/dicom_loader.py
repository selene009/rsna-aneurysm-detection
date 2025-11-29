# src/dicom_loader.py
"""
Pseudo DICOM loader for aneurysm detection pipeline.
"""

class DICOMLoader:
    def load(self, path):
        # Pseudo: return a 512x512 dummy image
        return [[0] * 512 for _ in range(512)]

    def preprocess(self, img):
        # Pseudo normalization
        return img

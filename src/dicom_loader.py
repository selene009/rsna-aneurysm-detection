# src/dicom_loader.py
"""
Pseudo DICOM loader for aneurysm detection pipeline.
"""

class DICOMLoader:
    def load(self, path):
        # Pseudo: return a 512x512 dummy image
        # [Originality]
        # This class provides a minimal abstraction over the real DICOM loading pipeline.
        # In the actual competition solution, this layer encapsulates:
        # - Recursive DICOM file loading
        # - Slice ordering using ImagePositionPatient / InstanceNumber
        # - Construction of 3D volumes from ordered slices
        #
        # [Outcome]
        # By decoupling data loading from model logic, the pipeline becomes easier to
        # maintain, extend, and adapt to different modalities (CT / MR).
        return [[0] * 512 for _ in range(512)]

    def preprocess(self, img):
        # Pseudo normalization
        # [Originality]
        # Preprocessing is explicitly separated from the model to ensure that
        # inference-time transformations exactly match training-time normalization,
        # such as CT windowing or percentile-based MR normalization.
        #
        # [Outcome]
        # This design reduces domain shift and improves inference stability
        # in heterogeneous medical imaging data.
        return img

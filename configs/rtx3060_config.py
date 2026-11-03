"""Configuration profile for the verified RTX 3060 Laptop GPU."""

from .rtx3050_config import RTX3050Config


class RTX3060Config(RTX3050Config):
    """Phase 3 profile for the 6 GB RTX 3060 Laptop GPU.

    It intentionally inherits the validated 4--8 GB memory settings. The
    historical RTX3050 profile remains available for old run reproducibility.
    """

    PROFILE_NAME = "rtx3060"
    EXPECTED_GPU = "NVIDIA GeForce RTX 3060 Laptop GPU"

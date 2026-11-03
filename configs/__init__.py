from .base_config import BaseConfig
from .rtx3050_config import RTX3050Config  # Historical compatibility profile
from .rtx3060_config import RTX3060Config
from .high_res_config import HighResolutionConfig

def get_config(config_name="auto"):
    """
    Factory function to get configuration based on hardware
    """
    if config_name == "auto":
        # Auto-detect based on GPU
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0).lower()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            if gpu_memory >= 12:
                return HighResolutionConfig()
            elif gpu_memory >= 4:
                if "3060" in gpu_name:
                    return RTX3060Config()
                return RTX3050Config()
    
    # Manual selection
    config_map = {
        "rtx3050": RTX3050Config,
        "rtx3060": RTX3060Config,
        "highres": HighResolutionConfig,
        "base": BaseConfig
    }
    
    return config_map.get(config_name, BaseConfig)()
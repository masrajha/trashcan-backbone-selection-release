"""
Configuration file for TrashCAN Pipeline
"""
import os
from pathlib import Path

class Config:
    # Paths
    ROOT_DIR = Path(".")
    ORIGINAL_DATA_DIR = ROOT_DIR / "original_data"
    ANNOTATIONS_DIR = ORIGINAL_DATA_DIR / "annotations"
    IMAGES_DIR = ORIGINAL_DATA_DIR / "images"
    
    # COCO Conversion
    DATASET_VERSION = "material"  # "material" or "instance"
    DATA_FRACTION = 1.0
    SPLIT_RATIO = 0.2
    RANDOM_SEED = 42
    
    # Training
    NUM_CLASSES = 16  # material categories, excluding background
    BATCH_SIZE = 2      # 4
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.005
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005
    STEP_SIZE = 10
    GAMMA = 0.1
    SKIP_VALIDATION = False
    
    # Model
    MODEL_NAME = "maskrcnn_resnet50_fpn"
    PRETRAINED = True
    MIN_SIZE = 600              # 800
    MAX_SIZE = 1000             # 1333
    
    # Paths for processed data
    @property
    def COCO_DIR(self):
        return self.ROOT_DIR / f"{self.DATASET_VERSION}_version"
    
    @property
    def TRAIN_DIR(self):
        return self.COCO_DIR / "train"
    
    @property
    def VAL_DIR(self):
        return self.COCO_DIR / "val"
    
    @property
    def TRAIN_JSON(self):
        return self.COCO_DIR / "instances_train_trashcan.json"
    
    @property
    def VAL_JSON(self):
        return self.COCO_DIR / "instances_val_trashcan.json"
    
    # Create directories
    def setup_directories(self):
        self.COCO_DIR.mkdir(exist_ok=True, parents=True)
        self.TRAIN_DIR.mkdir(exist_ok=True, parents=True)
        self.VAL_DIR.mkdir(exist_ok=True, parents=True)
        
config = Config()

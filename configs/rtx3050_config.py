# NEW: configs/rtx3050_config.py - Config spesifik untuk RTX 3050
import torch
from .base_config import BaseConfig

class RTX3050Config(BaseConfig):
    """Memory-conscious CUDA configuration for laptop GPUs with 4-8 GB VRAM.

    The historical class name is retained for CLI compatibility."""
    
    def __init__(self):
        super().__init__()  # Panggil parent constructor jika ada
        # GPU-specific settings
        self.GPU_MEMORY_GB = 6
        self.VRAM_SAFETY_MARGIN = 0.15  # 15% safety margin
        
        # Image dimensions
        self.ORIGINAL_SIZE = (480, 270)  # width, height
        self.ASPECT_RATIO = self.ORIGINAL_SIZE[0] / self.ORIGINAL_SIZE[1]
        
        self.DATASET_VERSION = "material"
        self.DATA_FRACTION = 1.0
        self.RANDOM_SEED = 42
        self.DETERMINISTIC = True
        self.ALLOW_TF32 = False

        # Tambahkan atribut yang dibutuhkan
        self.PRETRAINED = True  # Default menggunakan pretrained weights
        self.NUM_CLASSES = 16   # TrashCAN material categories, tanpa background
        
        # Optimized input size calculation
        self._calculate_optimal_input_size()
    
    def _calculate_optimal_input_size(self):
            """Calculate optimal input size based on GPU memory"""
            # Estimate memory per pixel (RGB + float16)
            MEMORY_PER_PIXEL = 3 * 2  # bytes (3 channels * 2 bytes per float16)
            
            # Available VRAM for input data
            available_vram_bytes = self.GPU_MEMORY_GB * (1024**3) * (1 - self.VRAM_SAFETY_MARGIN)
            
            # Estimate memory needed per sample
            # Mask R-CNN dengan ResNet50-FPN: ~3-4GB untuk model
            model_memory = 3.5 * (1024**3)  # 3.5GB untuk model Mask R-CNN
            
            # Memory for batch
            available_for_batch = available_vram_bytes - model_memory
            
            if available_for_batch <= 0:
                # Fallback ke ukuran yang lebih kecil
                print("⚠️  Not enough VRAM for model, using fallback size")
                self.INPUT_HEIGHT = 512
                self.INPUT_WIDTH = 512
                return
            
            # Target batch size untuk RTX 3050 6GB
            target_batch_size = 2  # Untuk Mask R-CNN, batch size kecil
            
            memory_per_sample = available_for_batch / target_batch_size
            
            # Calculate max pixels per sample
            max_pixels = memory_per_sample / MEMORY_PER_PIXEL
            
            # Batasi maksimal pixel (untuk mencegah ukuran terlalu besar)
            MAX_ALLOWED_PIXELS = 1024 * 1024  # 1MP maksimum
            
            if max_pixels > MAX_ALLOWED_PIXELS:
                max_pixels = MAX_ALLOWED_PIXELS
            
            # Maintain aspect ratio
            # height = sqrt(max_pixels / aspect_ratio)
            target_height = int((max_pixels / self.ASPECT_RATIO) ** 0.5)
            target_width = int(target_height * self.ASPECT_RATIO)
            
            # Round to multiples of 32 (better for CNNs)
            self.INPUT_HEIGHT = ((target_height + 31) // 32) * 32
            self.INPUT_WIDTH = ((target_width + 31) // 32) * 32
            
            # Batasi maksimal ukuran untuk Mask R-CNN
            MAX_DIMENSION = 1333  # Ukuran maksimal yang umum untuk Mask R-CNN
            
            if self.INPUT_WIDTH > MAX_DIMENSION or self.INPUT_HEIGHT > MAX_DIMENSION:
                scale = MAX_DIMENSION / max(self.INPUT_WIDTH, self.INPUT_HEIGHT)
                self.INPUT_WIDTH = int(self.INPUT_WIDTH * scale / 32) * 32
                self.INPUT_HEIGHT = int(self.INPUT_HEIGHT * scale / 32) * 32
            
            print(f"GPU: {self.GPU_NAME}")
            print(f"Original: {self.ORIGINAL_SIZE[0]}x{self.ORIGINAL_SIZE[1]}")
            print(f"Optimized: {self.INPUT_WIDTH}x{self.INPUT_HEIGHT}")
    
    @property
    def BATCH_SIZE(self):
        """Dynamic batch size based on available VRAM"""
        if hasattr(self, "_batch_size_override"):
            return self._batch_size_override

        # Conservative default for memory-heavy Mask R-CNN training.
        # Override explicitly only after a successful smoke test.
        return 2
    
    @property
    def MIN_SIZE(self):
        if hasattr(self, "_min_size_override"):
            return self._min_size_override
        return min(self.INPUT_HEIGHT, 800)  # Batasi minimal size

    @MIN_SIZE.setter
    def MIN_SIZE(self, value):
        self._min_size_override = int(value)
    
    @property
    def MAX_SIZE(self):
        if hasattr(self, "_max_size_override"):
            return self._max_size_override
        return min(self.INPUT_WIDTH, 1333)  # Batasi maksimal size untuk Mask R-CNN

    @MAX_SIZE.setter
    def MAX_SIZE(self, value):
        self._max_size_override = int(value)
    
    # Training parameters optimized for RTX 3050
    NUM_EPOCHS = 30  # Reduced for faster experimentation
    LEARNING_RATE = 0.001  # Lower for stable training with higher batch size
    
    # Mixed precision settings
    USE_AMP = True
    AMP_DTYPE = torch.float16
    
    # Data loading optimized for laptop
    NUM_WORKERS = 3  # Laptop CPUs have fewer cores
    PIN_MEMORY = True
    PERSISTENT_WORKERS = True
    
    # Validation frequency
    VALIDATE_EVERY = 3  # Validate every 3 epochs
    SKIP_VALIDATION = False
    
    # Early stopping
    PATIENCE = 10  # Stop if no improvement for 10 epochs
    
    # Optimizer settings
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005
    STEP_SIZE = 10
    GAMMA = 0.1
    
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

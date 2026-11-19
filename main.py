#!/usr/bin/env python3
"""
TrashCAN Dataset Pipeline - Main Entry Point
Author: Didik K
AI assistance: OpenAI Codex
Version: 2.0
Description: Complete pipeline for TrashCAN instance segmentation
"""

import argparse
import json
import sys
import os
import warnings
import traceback
from pathlib import Path
from datetime import datetime

# Keep CLI output stable on Windows consoles that default to cp1252.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import custom modules
try:
    from configs import get_config
    from models.model import TrashCANModel
    from data.dataset import get_dataloaders
    from training.fast_trainer_rtx3050 import RTX3050Trainer
    from training.trainer import Trainer
    from inference.inference import TrashCANInference
    from utils.monitor import monitor_system
    from utils.gpu_utils import get_device, clear_gpu_cache
    from utils.experiment import resolve_experiment_config
    from utils.reproducibility import seed_everything
    from evaluation.detection_evaluator import DetectionEvaluator
    from scripts.trash_can_coco import main as convert_dataset
except ImportError as e:
    print(f"❌ Import error: {e}")
    # Tambahkan debugging detail
    print(f"Current directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    import traceback
    traceback.print_exc()
    # print("📦 Installing missing packages...")
    # os.system("pip install -r requirements.txt")
    sys.exit(1)

# ASCII Art Banner
BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║                     TrashCAN Segmentation                    ║
║                  Underwater Instance Detection               ║
╚══════════════════════════════════════════════════════════════╝
"""

def print_banner():
    """Print ASCII banner"""
    print("\033[1;36m" + BANNER + "\033[0m")
    print(f"\033[1;33m{'='*70}\033[0m")
    print(f"\033[1;32m{'TRASHCAN DATASET PIPELINE':^70}\033[0m")
    print(f"\033[1;33m{'='*70}\033[0m")

def setup_environment():
    """Setup environment for optimal performance"""
    # Set environment variables
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    # Set CUDA visible devices if specified
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    print("🔧 Environment setup complete")
    print(f"   CUDA Visible Devices: {os.environ['CUDA_VISIBLE_DEVICES']}")

def check_dependencies():
    """Check if all dependencies are available"""
    required_packages = [
        'torch',
        'torchvision',
        'numpy',
        'opencv-python',
        'matplotlib',
        'pycocotools',
        'Pillow',
        'tqdm',
        'shapely',
        'scikit-image'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("📦 Installing...")
        os.system(f"pip install {' '.join(missing_packages)}")
        return False

    return True

def normalize_data_fraction(value):
    """Accept either 0.5 or 50 for a 50 percent dataset subset."""
    if value is None:
        return 1.0
    value = float(value)
    if value > 1.0:
        value = value / 100.0
    if not 0 < value <= 1:
        raise ValueError("--data-fraction must be in the range (0, 1] or (0, 100]")
    return value

def count_dataset_categories(config):
    """Count object categories from the active COCO train JSON."""
    train_json = Path(config.TRAIN_JSON)
    if not train_json.exists():
        return getattr(config, "NUM_CLASSES", 16)

    with train_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return len(data.get("categories", []))

def apply_runtime_overrides(config, args):
    """Apply CLI runtime choices to the config object used across the pipeline."""
    config.DATASET_VERSION = args.version
    config.DATA_FRACTION = normalize_data_fraction(args.data_fraction)

    if args.data_dir:
        config.ORIGINAL_DATA_DIR = Path(args.data_dir)
    if args.batch_size:
        if hasattr(type(config), "BATCH_SIZE") and isinstance(getattr(type(config), "BATCH_SIZE"), property):
            config._batch_size_override = args.batch_size
        else:
            config.BATCH_SIZE = args.batch_size
    if args.lr:
        config.LEARNING_RATE = args.lr
    if args.epochs:
        config.NUM_EPOCHS = args.epochs
    if args.input_height:
        config.INPUT_HEIGHT = args.input_height
    if args.input_width:
        config.INPUT_WIDTH = args.input_width
    if args.min_size:
        if hasattr(type(config), "MIN_SIZE") and isinstance(getattr(type(config), "MIN_SIZE"), property):
            config._min_size_override = args.min_size
        else:
            config.MIN_SIZE = args.min_size
    elif args.input_height:
        inferred_min_size = min(args.input_height, getattr(config, "MIN_SIZE", args.input_height))
        if hasattr(type(config), "MIN_SIZE") and isinstance(getattr(type(config), "MIN_SIZE"), property):
            config._min_size_override = inferred_min_size
        else:
            config.MIN_SIZE = inferred_min_size
    if args.max_size:
        if hasattr(type(config), "MAX_SIZE") and isinstance(getattr(type(config), "MAX_SIZE"), property):
            config._max_size_override = args.max_size
        else:
            config.MAX_SIZE = args.max_size
    elif args.input_width:
        inferred_max_size = max(args.input_width, getattr(config, "MAX_SIZE", args.input_width))
        if hasattr(type(config), "MAX_SIZE") and isinstance(getattr(type(config), "MAX_SIZE"), property):
            config._max_size_override = inferred_max_size
        else:
            config.MAX_SIZE = inferred_max_size
    if args.skip_val:
        config.SKIP_VALIDATION = True
    if args.no_pretrained:
        config.PRETRAINED = False
    config.BACKBONE = args.backbone

    from models.backbone_registry import backbone_provenance
    config.BACKBONE_PROVENANCE = backbone_provenance(args.backbone)

    config.NUM_CLASSES = count_dataset_categories(config)
    return config

def convert_mode(config):
    """Convert dataset to COCO format"""
    print("\n" + "="*60)
    print("🔄 CONVERTING DATASET TO COCO FORMAT")
    print("="*60)

    try:
        # Import the conversion script
        from scripts.trash_can_coco import main as convert_main

        # Set command line arguments for conversion
        sys.argv = ['trash_can_coco.py', config.DATASET_VERSION]

        # Run conversion
        print(f"📁 Input: {config.ORIGINAL_DATA_DIR}")
        print(f"📁 Output: {config.COCO_DIR}")
        print(f"📊 Version: {config.DATASET_VERSION}")

        convert_main()

        # Verify conversion
        train_json = config.COCO_DIR / "instances_train_trashcan.json"
        val_json = config.COCO_DIR / "instances_val_trashcan.json"

        if train_json.exists() and val_json.exists():
            print("✅ Conversion successful!")
            print(f"   Train annotations: {train_json}")
            print(f"   Val annotations: {val_json}")
        else:
            raise RuntimeError("Konversi selesai tanpa menghasilkan anotasi train dan val")

    except Exception as e:
        raise RuntimeError(f"Konversi dataset gagal: {e}") from e

    return True

def train_mode(config, args):
    """Train the model"""
    print("\n" + "="*60)
    print("🚀 TRAINING MODE")
    print("="*60)

    # Monitor system
    monitor_system()

    # Setup device
    device = get_device()
    print(f"🎯 Using device: {device}")

    # Check if dataset exists
    if not config.COCO_DIR.exists():
        raise FileNotFoundError(
            f"COCO dataset tidak ditemukan: {config.COCO_DIR}. Jalankan mode convert terlebih dahulu."
        )

    # Create dataloaders
    print("\n📂 Loading dataset...")
    try:
        train_loader, val_loader, train_dataset, val_dataset = get_dataloaders(config)

        print(f"📊 Dataset statistics:")
        print(f"   Training samples: {len(train_dataset)}")
        print(f"   Validation samples: {len(val_dataset)}")
        print(f"   Dataset version: {config.DATASET_VERSION}")
        print(f"   Data fraction: {config.DATA_FRACTION:.0%}")
        print(f"   Batch size: {config.BATCH_SIZE}")
        print(f"   Input size: {config.INPUT_WIDTH}x{config.INPUT_HEIGHT}")

        # Show class distribution
        if hasattr(train_dataset, 'coco'):
            categories = train_dataset.coco.dataset['categories']
            print(f"   Number of classes: {len(categories)}")
            print("   Classes:", ', '.join([cat['name'] for cat in categories[:5]]))
            if len(categories) > 5:
                print(f"   ... and {len(categories)-5} more")

    except Exception as e:
        raise RuntimeError(f"Gagal memuat dataset: {e}") from e

    # Create model
    print("\n🧠 Creating model...")
    try:
        model = TrashCANModel(num_classes=config.NUM_CLASSES, cfg=config)
        print(f"✅ Model created: {model.__class__.__name__}")
        print(f"   Backbone: {config.BACKBONE}")
        print(f"   Pretrained: {getattr(config, 'PRETRAINED', True)}")
        print(f"   Num classes: {getattr(config, 'NUM_CLASSES', 22)}")
    except Exception as e:
        raise RuntimeError(f"Gagal membuat model: {e}") from e

    # Create trainer based on config
    print("\n⚙️  Setting up trainer...")
    try:
        use_optimized_cuda_trainer = (
            device.type == "cuda" and bool(getattr(config, "USE_AMP", False))
        )
        if use_optimized_cuda_trainer:
            # Use AMP-aware trainer for supported CUDA configurations.
            trainer = RTX3050Trainer(model, train_loader, val_loader, device, config)
            print("✅ Using memory-optimized CUDA trainer")
        else:
            # Pastikan Trainer juga menerima argument yang benar
            trainer = Trainer(model, train_loader, val_loader, device, config)
            print("✅ Using standard trainer")

        print(f"   Optimizer: SGD (lr={config.LEARNING_RATE})")
        print(f"   Epochs: {args.epochs or config.NUM_EPOCHS}")
        print(f"   Mixed Precision: {getattr(config, 'USE_AMP', False)}")

    except Exception as e:
        raise RuntimeError(f"Gagal membuat trainer: {e}") from e

    # Train
    print("\n" + "="*60)
    print("🎬 STARTING TRAINING")
    print("="*60)

    try:
        train_kwargs = {'num_epochs': args.epochs or config.NUM_EPOCHS}
        if args.resume_from:
            if not isinstance(trainer, RTX3050Trainer):
                raise ValueError('--resume-from saat ini memerlukan CUDA AMP trainer')
            train_kwargs['resume_from'] = args.resume_from
            train_kwargs['resume_reason'] = args.resume_reason
        history = trainer.train(**train_kwargs)

        # Plot results (jika ada method plot_history)
        if hasattr(trainer, 'plot_history'):
            trainer.plot_history()

        print("\n✅ Training completed successfully!")
        print(f"📁 Output saved to: {trainer.output_dir}")

        # Show final metrics
        if history.get('coco_metrics'):
            latest = history['coco_metrics'][-1]
            print("\n🏆 Final Metrics:")
            target = latest.get('target_class', {})
            if target:
                print(f"   AP {target.get('class_name', config.TARGET_CLASS)}: {target.get('AP', 0):.4f}")
            print(f"   mAP@[0.5:0.95]: {latest.get('AP', 0):.4f}")
            print(f"   mAP@0.5: {latest.get('AP50', 0):.4f}")
            print(f"   mAP@0.75: {latest.get('AP75', 0):.4f}")

    except Exception as e:
        raise RuntimeError(f"Training gagal: {e}") from e

    # Clear GPU cache
    clear_gpu_cache()
    return history

def evaluate_mode(config, args):
    """Evaluate a trained model"""
    print("\n" + "="*60)
    print("📊 EVALUATION MODE")
    print("="*60)

    if not args.model_path:
        raise ValueError("Mode eval membutuhkan --model-path")

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")

    # Setup device
    device = get_device()

    # Create inference engine
    print(f"📂 Loading model from: {model_path}")
    try:
        inference = TrashCANInference(str(model_path), device, config)
        print(f"✅ Model loaded successfully")
        print(f"   Device: {device}")
        print(f"   Categories: {len(inference.category_map)}")

    except Exception as e:
        raise RuntimeError(f"Gagal memuat model: {e}") from e

    # Evaluate on validation set
    print("\n📈 Running evaluation...")
    try:
        _, val_loader, _, _ = get_dataloaders(config)
        evaluator = DetectionEvaluator(
            model=inference.model,
            dataloader=val_loader,
            device=device,
            annotation_file=config.EVAL_ANNOTATION_JSON,
            category_map=config.CATEGORY_MAP,
            output_dir=config.RUN_DIR / "metrics" / "evaluation",
            target_class=config.TARGET_CLASS,
            score_threshold=config.EVAL_SCORE_THRESHOLD,
        )
        metrics = evaluator.evaluate()

        print("\n🏆 Evaluation Results:")
        print("="*40)
        for key, value in metrics["overall"].items():
            print(f"   {key:15}: {value:.4f}")
        target = metrics["target_class"]
        print(f"\n   Target class: {target['class_name']}")
        print(f"   AP/AP50/AP75: {target['AP']:.4f} / {target['AP50']:.4f} / {target['AP75']:.4f}")
        print(f"   Artifacts: {config.RUN_DIR / 'metrics' / 'evaluation'}")

    except Exception as e:
        raise RuntimeError(
            "Evaluasi gagal; tidak ada fallback ke test resmi. "
            f"Perbaiki konfigurasi atau manifest internal: {e}"
        ) from e
    return metrics

def inference_mode(config, args):
    """Run inference on images"""
    print("\n" + "="*60)
    print("🔍 INFERENCE MODE")
    print("="*60)

    if not args.model_path:
        raise ValueError("Mode infer membutuhkan --model-path")

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")

    if not args.image_path and not args.image_dir:
        raise ValueError("Mode infer membutuhkan --image-path atau --image-dir")

    # Setup device
    device = get_device()

    # Create inference engine
    print(f"📂 Loading model from: {model_path}")
    try:
        inference = TrashCANInference(str(model_path), device, config)
        print(f"✅ Model loaded successfully")

    except Exception as e:
        raise RuntimeError(f"Gagal memuat model: {e}") from e

    # Single image inference
    if args.image_path:
        image_path = Path(args.image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Citra tidak ditemukan: {image_path}")

        print(f"🖼️  Processing: {image_path.name}")

        # Create output directory
        output_dir = Path("inference_results") / datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"pred_{image_path.stem}.png"

        # Run inference
        try:
            inference.visualize(str(image_path), str(output_path),
                              confidence_threshold=args.confidence)
            print(f"✅ Results saved to: {output_path}")

            # Also save raw predictions
            predictions = inference.predict(str(image_path), args.confidence)
            pred_file = output_dir / f"pred_{image_path.stem}.json"
            import json
            with open(pred_file, 'w') as f:
                json.dump(predictions, f, indent=2)
            print(f"📄 Predictions saved to: {pred_file}")

        except Exception as e:
            print(f"❌ Inference failed: {e}")

    # Batch inference
    elif args.image_dir:
        image_dir = Path(args.image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"Direktori citra tidak ditemukan: {image_dir}")

        print(f"📁 Processing directory: {image_dir}")

        # Create output directory
        output_dir = Path("inference_results") / datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run batch inference
        try:
            results = inference.batch_predict(
                str(image_dir),
                str(output_dir),
                confidence_threshold=args.confidence
            )

            print(f"\n✅ Batch processing complete!")
            print(f"📁 Results saved to: {output_dir}")
            print(f"📊 Processed {len(results)} images")

        except Exception as e:
            print(f"❌ Batch inference failed: {e}")

def export_mode(config, args):
    """Export model to different formats"""
    print("\n" + "="*60)
    print("📦 EXPORT MODE")
    print("="*60)

    if not args.model_path:
        raise ValueError("Mode export membutuhkan --model-path")

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")

    print(f"📂 Loading model from: {model_path}")

    try:
        import torch
        from models.model import TrashCANModel

        # Load model
        device = get_device()
        model = TrashCANModel(num_classes=config.NUM_CLASSES, cfg=config).to(device)
        model.load(str(model_path))
        model.eval()

        # Create export directory
        export_dir = Path("exported_models") / datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir.mkdir(parents=True, exist_ok=True)

        # Export formats
        formats = args.export_format or ['torchscript', 'onnx']

        for fmt in formats:
            if fmt == 'torchscript':
                print("\n📦 Exporting to TorchScript...")
                try:
                    # Create example input
                    example_input = torch.randn(1, 3, config.INPUT_HEIGHT, config.INPUT_WIDTH).to(device)

                    # Trace the model
                    traced_model = torch.jit.trace(model.model, example_input)

                    # Save
                    output_path = export_dir / "model_traced.pt"
                    traced_model.save(str(output_path))
                    print(f"✅ TorchScript model saved to: {output_path}")

                except Exception as e:
                    print(f"❌ TorchScript export failed: {e}")

            elif fmt == 'onnx':
                print("\n📦 Exporting to ONNX...")
                try:
                    import torch.onnx

                    # Create example input
                    dummy_input = torch.randn(1, 3, config.INPUT_HEIGHT, config.INPUT_WIDTH).to(device)

                    # Export
                    output_path = export_dir / "model.onnx"
                    torch.onnx.export(
                        model.model,
                        dummy_input,
                        str(output_path),
                        export_params=True,
                        opset_version=11,
                        do_constant_folding=True,
                        input_names=['input'],
                        output_names=['output'],
                        dynamic_axes={
                            'input': {0: 'batch_size'},
                            'output': {0: 'batch_size'}
                        }
                    )
                    print(f"✅ ONNX model saved to: {output_path}")

                except Exception as e:
                    print(f"❌ ONNX export failed: {e}")

            elif fmt == 'tensorrt':
                print("\n⚠️  TensorRT export requires additional setup")
                print("   Please install tensorrt: pip install tensorrt")

            else:
                print(f"❌ Unknown format: {fmt}")

        print(f"\n📁 All exports saved to: {export_dir}")

    except Exception as e:
        raise RuntimeError(f"Export gagal: {e}") from e

def benchmark_mode(config, args):
    """Benchmark model performance"""
    print("\n" + "="*60)
    print("⚡ BENCHMARK MODE")
    print("="*60)

    from utils.benchmark import (
        benchmark_training,
        benchmark_inference_pipeline,
        benchmark_batch_inference,
        save_benchmark_artifacts,
    )

    # Monitor system
    monitor_system()

    # Setup device
    device = get_device()

    if args.benchmark_type == 'training':
        print("\n🏋️  Benchmarking training performance...")

        # Create dataloaders
        train_loader, _, _, _ = get_dataloaders(config)

        # Create model
        model = TrashCANModel(num_classes=config.NUM_CLASSES, cfg=config).to(device)

        # Run benchmark
        results = benchmark_training(
            model,
            train_loader,
            num_batches=args.num_batches,
            use_amp=bool(getattr(config, "USE_AMP", False)),
        )
        save_benchmark_artifacts(results, config.RUN_DIR / "benchmarks", "training_benchmark")

        print(f"\n📊 Training Benchmark Results:")
        print(f"   Average batch time: {results['timing']['mean_seconds']:.3f}s")
        print(f"   Batches per second: {results['timing']['batches_per_second']:.1f}")
        print(f"   Estimated epoch time: {results['timing']['estimated_epoch_minutes']:.1f} minutes")

    elif args.benchmark_type == 'inference':
        print("\n⚡ Benchmarking inference performance...")

        if not args.model_path:
            raise ValueError("Benchmark inference membutuhkan --model-path")

        # Load inference pipeline
        inference = TrashCANInference(args.model_path, device, config)

        # Gunakan image default jika tidak ada
        if args.image_path:
            image_path = args.image_path
        else:
            _, _, _, val_dataset = get_dataloaders(config)
            image_id = int(val_dataset.ids[0])
            file_name = val_dataset.coco.imgs[image_id]["file_name"]
            image_path = str(Path(config.TRAIN_DIR) / file_name)
            print(f"📁 Using val_internal image: {image_path}")

        # Run benchmark
        results = benchmark_inference_pipeline(
            inference=inference,
            image_path=image_path,
            num_iterations=args.num_iterations,
            confidence_threshold=args.confidence if hasattr(args, 'confidence') else 0.5
        )

        if results:
            save_benchmark_artifacts(results, config.RUN_DIR / "benchmarks", "inference_benchmark")
            print(f"\n✅ Benchmark completed successfully!")
            print(f"   Average inference time: {results['timing']['avg_ms']:.1f}ms")
            print(f"   FPS: {results['timing']['fps']:.1f}")
            print(f"   Throughput: {results['performance']['throughput_fps']:.1f} images/sec")
        else:
            raise RuntimeError("Benchmark inference tidak menghasilkan data")

    elif args.benchmark_type == 'batch':
        print("\n📦 Benchmarking batch inference performance...")

        if not args.model_path:
            print("❌ Please specify model path with --model-path")
            return

        # Load model
        inference = TrashCANInference(args.model_path, device, config)

        # Gunakan image directory
        if args.image_path:
            image_dir = args.image_path
        else:
            image_dir = "instance_version/val"

        # Run benchmark
        results = benchmark_batch_inference(
            model=inference.model,
            image_dir=image_dir,
            batch_sizes=[1, 2, 4, 8],
            num_iterations=args.num_iterations
        )

    elif args.benchmark_type == 'memory':
        print("\n💾 Benchmarking memory usage...")

        if not args.model_path:
            print("❌ Please specify model path with --model-path")
            return

        # Load model
        inference = TrashCANInference(args.model_path, device, config)

        # Run memory benchmark
        results = benchmark_memory(
            model=inference.model,
            input_size=(3, config.INPUT_HEIGHT, config.INPUT_WIDTH)
        )

    else:
        raise ValueError(f"Benchmark type tidak valid: {args.benchmark_type}")
    return results

def main():
    """Main entry point"""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='TrashCAN Dataset Pipeline - Instance Segmentation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode convert --version instance
  python main.py --mode train --config rtx3060 --epochs 30
  python main.py --mode train --config rtx3060 --version material --data-fraction 5 --epochs 1 --input-height 320 --input-width 569 --max-size 640 --skip-val
  python main.py --mode infer --model-path best_model.pth --image-path test.jpg
  python main.py --mode eval --model-path best_model.pth
  python main.py --mode benchmark --benchmark-type training
        """
    )

    # Main arguments
    parser.add_argument('--mode', required=True,
                       choices=['convert', 'train', 'eval', 'infer',
                                'export', 'benchmark', 'monitor'],
                       help='Pipeline mode')
    parser.add_argument('--config', default='auto',
                       choices=['auto', 'rtx3060', 'rtx3050', 'highres', 'base', 'custom'],
                       help='Configuration to use')

    # Dataset arguments
    parser.add_argument('--version', default='material',
                       choices=['instance', 'material'],
                       help='Dataset version (instance or material)')
    parser.add_argument('--data-dir', help='Path to dataset directory')
    parser.add_argument('--data-fraction', type=float, default=1.0,
                       help='Fraction of train/val dataset to load. Accepts 0.5 or 50 for 50%%.')
    parser.add_argument('--manifest-path',
                       help='Internal split manifest; defaults to Riset/output/research_phase1/internal_split_manifest.json')
    parser.add_argument('--target-class', default='trash_plastic',
                       help='Primary class for per-class evaluation')
    parser.add_argument('--output-root', default='Riset/output/runs',
                       help='Root directory for resolved experiment runs')
    parser.add_argument('--run-id', help='Stable run identifier; timestamp is used when omitted')
    parser.add_argument('--eval-score-threshold', type=float, default=0.0,
                       help='Prediction threshold for COCO evaluation; use 0 for formal AP')

    # Training arguments
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--resume-from',
                       help='Full-state checkpoint untuk melanjutkan training')
    parser.add_argument('--resume-reason', default='infrastructure_interruption',
                       help='Alasan audit untuk resume training')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--input-height', type=int,
                       help='Override model input height/min resize for smoke tests')
    parser.add_argument('--input-width', type=int,
                       help='Override model input width/max resize for smoke tests')
    parser.add_argument('--min-size', type=int,
                       help='Override torchvision detection min_size')
    parser.add_argument('--max-size', type=int,
                       help='Override torchvision detection max_size')
    parser.add_argument('--skip-val', action='store_true',
                       help='Skip validation/evaluation during training smoke tests')
    parser.add_argument('--no-pretrained', action='store_true',
                       help='Disable pretrained weights (intended for offline smoke tests)')
    parser.add_argument('--backbone', default='resnet50',
                       choices=['resnet50', 'efficientnet_b0', 'convnext_tiny',
                                'swin_t', 'pvtv2_b1', 'swin_v2_t'],
                       help='Registered Phase 3 backbone')
    parser.add_argument('--phase3-stage', choices=['smoke', 'pilot', 'formal'],
                       help='Enforce the controlled Phase 3 training protocol for this stage')
    parser.add_argument('--phase3-protocol',
                       default='Riset/configs/phase3_training_protocol.json',
                       help='Machine-readable Phase 3 detector/training protocol')
    parser.add_argument('--phase3-source-attestation',
                       help='Verified source-bundle attestation for a Git-less Kaggle runtime')
    parser.add_argument('--phase4-fixed-fusion', choices=['smoke', 'full'],
                       help='Use the registered EfficientNet-B0/PVTv2-B1 fixed-fusion baseline')
    parser.add_argument('--phase4-protocol',
                       default='Riset/configs/phase4_fixed_fusion_protocol.json',
                       help='Recorded Phase 4 fixed-fusion transition')
    parser.add_argument('--seed', type=int,
                       help='Formal seed; must be registered by the Phase 3 protocol')

    # Model arguments
    parser.add_argument('--model-path', help='Path to model for inference/evaluation')
    parser.add_argument('--model-type', default='maskrcnn',
                       choices=['maskrcnn', 'fasterrcnn', 'yolact', 'solov2'],
                       help='Model architecture')

    # Inference arguments
    parser.add_argument('--image-path', help='Path to image for inference')
    parser.add_argument('--image-dir', help='Path to directory of images')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold for detection')

    # Export arguments
    parser.add_argument('--export-format', nargs='+',
                       choices=['torchscript', 'onnx', 'tensorrt'],
                       help='Export format(s)')

    # Benchmark arguments
    parser.add_argument('--benchmark-type',
                       choices=['training', 'inference'],
                       help='Type of benchmark to run')
    parser.add_argument('--num-batches', type=int, default=50,
                       help='Number of batches for training benchmark')
    parser.add_argument('--num-iterations', type=int, default=100,
                       help='Number of iterations for inference benchmark')

    # Other arguments
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode')
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run without actual execution')

    args = parser.parse_args()

    # Setup
    if not args.dry_run:
        print_banner()
        setup_environment()

    # Get configuration
    config = get_config(args.config)

    config = apply_runtime_overrides(config, args)

    if args.phase3_stage and args.phase4_fixed_fusion:
        parser.error('--phase3-stage and --phase4-fixed-fusion cannot be combined')
    if args.phase4_fixed_fusion:
        if args.no_pretrained:
            parser.error('--no-pretrained conflicts with the locked Phase 4 branch checkpoints')
        from utils.phase4_fixed_fusion import apply_phase4_fixed_fusion
        config = apply_phase4_fixed_fusion(
            config, stage=args.phase4_fixed_fusion, seed=args.seed,
            path=args.phase4_protocol,
        )
        config.PHASE4_INITIALIZE_BRANCHES = (
            (args.mode == 'train' or
             (args.mode == 'benchmark' and args.benchmark_type == 'training'))
            and not args.resume_from
        )
        frozen_manifest = config.PHASE4_FIXED_FUSION_PROTOCOL['split_manifest']
        if args.manifest_path and Path(args.manifest_path).resolve() != Path(frozen_manifest).resolve():
            parser.error('Phase 4 baseline requires the recorded frozen split manifest')
        args.manifest_path = frozen_manifest

    if args.phase3_stage:
        from utils.phase3_protocol import apply_phase3_protocol, load_phase3_protocol

        protocol, protocol_path, protocol_sha256 = load_phase3_protocol(
            args.phase3_protocol
        )
        config = apply_phase3_protocol(
            config, protocol, stage=args.phase3_stage, seed=args.seed
        )
        config.PHASE3_TRAINING_PROTOCOL = protocol
        config.PHASE3_TRAINING_PROTOCOL_PATH = protocol_path
        config.PHASE3_TRAINING_PROTOCOL_SHA256 = protocol_sha256
        if args.phase3_stage == "formal":
            from utils.phase3_formal_preflight import (
                validate_phase3_formal_preflight,
            )

            config.PHASE3_FORMAL_PREFLIGHT = validate_phase3_formal_preflight(
                backbones=[args.backbone],
                seeds=[getattr(config, "RANDOM_SEED")],
                require_clean=True,
                source_attestation=args.phase3_source_attestation,
            )

    reproducibility = seed_everything(
        getattr(config, "RANDOM_SEED", 42),
        deterministic=getattr(config, "DETERMINISTIC", True),
    )
    config.REPRODUCIBILITY = reproducibility

    research_modes = {'train', 'eval', 'benchmark'}
    if args.mode in research_modes | {'infer', 'export'}:
        config = resolve_experiment_config(
            config,
            manifest_path=args.manifest_path,
            output_root=args.output_root,
            run_id=args.run_id,
            target_class=args.target_class,
            score_threshold=args.eval_score_threshold,
            mode=args.mode,
            require_manifest=args.mode in research_modes,
            model_path=args.model_path,
            resume_from=args.resume_from,
        )

    # Debug info
    if args.verbose:
        print("\n📋 Configuration:")
        print(f"   Mode: {args.mode}")
        print(f"   Config: {args.config}")
        print(f"   Dataset Version: {config.DATASET_VERSION}")
        print(f"   Data Fraction: {config.DATA_FRACTION:.0%}")
        print(f"   GPU: {config.GPU_NAME}")
        print(f"   Input Size: {config.INPUT_WIDTH}x{config.INPUT_HEIGHT}")
        print(f"   Min/Max Size: {config.MIN_SIZE}/{config.MAX_SIZE}")
        print(f"   Batch Size: {config.BATCH_SIZE}")
        print(f"   Skip Validation: {getattr(config, 'SKIP_VALIDATION', False)}")

    if args.dry_run:
        print("\nDry run complete; no model, dataloader, or official test data was opened.")
        if hasattr(config, 'RUN_DIR'):
            print(f"Resolved run directory: {config.RUN_DIR}")
            print(f"Split manifest hash: {config.SPLIT_MANIFEST_SHA256}")
        return

    # Run selected mode
    try:
        if args.mode == 'convert':
            convert_mode(config)

        elif args.mode == 'train':
            train_mode(config, args)

        elif args.mode == 'eval':
            evaluate_mode(config, args)

        elif args.mode == 'infer':
            inference_mode(config, args)

        elif args.mode == 'export':
            export_mode(config, args)

        elif args.mode == 'benchmark':
            benchmark_mode(config, args)

        elif args.mode == 'monitor':
            monitor_system()

        else:
            print(f"❌ Unknown mode: {args.mode}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        if hasattr(config, "RUN_DIR"):
            (config.RUN_DIR / "run_status.json").write_text(
                json.dumps({"status": "interrupted", "exit_code": 130}, indent=2),
                encoding="utf-8",
            )
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        if hasattr(config, "RUN_DIR"):
            (config.RUN_DIR / "run_status.json").write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "error_type": type(e).__name__,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        if args.debug:
            traceback.print_exc()
        sys.exit(1)

    if hasattr(config, "RUN_DIR"):
        (config.RUN_DIR / "run_status.json").write_text(
            json.dumps({"status": "completed", "exit_code": 0}, indent=2),
            encoding="utf-8",
        )

    print("\n" + "="*60)
    print("🎉 Pipeline completed successfully!")
    print("="*60)

if __name__ == "__main__":
    # Check dependencies
    # if not check_dependencies():
    #     print("❌ Please install missing dependencies and try again")
    #     sys.exit(1)

    # Run main
    main()

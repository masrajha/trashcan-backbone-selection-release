# utils/benchmark.py
import time
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import json
import csv
from datetime import datetime

def benchmark_training(model, train_loader, num_batches=100, use_amp=False):
    """Benchmark forward/backward speed using the configured precision."""
    model.train()
    device = next(model.parameters()).device
    amp_enabled = bool(use_amp and device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)

    times = []

    # Warmup
    for i, (images, targets) in enumerate(train_loader):
        if i >= min(3, num_batches):
            break
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        model.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=amp_enabled, dtype=torch.float16):
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
        scaler.scale(loss).backward()

    # Benchmark
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.time()

    for i, (images, targets) in enumerate(train_loader):
        if i >= num_batches:
            break

        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        batch_start = time.time()

        model.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=amp_enabled, dtype=torch.float16):
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
        scaler.scale(loss).backward()

        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        batch_time = time.time() - batch_start
        times.append(batch_time)

        if i % 10 == 0:
            print(f"Batch {i}: {batch_time:.3f}s")

    if not times:
        raise ValueError("Training benchmark tidak menghasilkan data timing")
    avg_time = sum(times) / len(times)
    print(f"\nAverage batch time: {avg_time:.3f}s")
    print(f"Estimated epoch time: {avg_time * len(train_loader) / 60:.1f} minutes")
    return {
        'benchmark_type': 'training',
        'device': str(device),
        'iterations': len(times),
        'batch_size': getattr(train_loader, 'batch_size', None),
        'precision': 'amp_fp16' if amp_enabled else 'fp32',
        'dataset_batches': len(train_loader),
        'timing': {
            'mean_seconds': float(np.mean(times)),
            'median_seconds': float(np.percentile(times, 50)),
            'p90_seconds': float(np.percentile(times, 90)),
            'p95_seconds': float(np.percentile(times, 95)),
            'batches_per_second': float(1.0 / avg_time),
            'estimated_epoch_minutes': float(avg_time * len(train_loader) / 60),
        },
        'memory': {
            'peak_allocated_mb': (
                float(torch.cuda.max_memory_allocated(device) / 1024**2)
                if device.type == 'cuda'
                else None
            )
        },
    }

def benchmark_inference(model, image_path, num_iterations=100, image_size=(800, 1333), confidence_threshold=0.5):
    """Benchmark inference speed on single image"""
    model.eval()
    device = next(model.parameters()).device

    print(f"\n{'='*60}")
    print("SINGLE IMAGE INFERENCE BENCHMARK")
    print(f"{'='*60}")

    # Load and preprocess image
    if isinstance(image_path, (str, Path)):
        if not Path(image_path).exists():
            print(f"❌ Image not found: {image_path}")
            return None

        image = Image.open(image_path).convert('RGB')
        print(f"📄 Image loaded: {Path(image_path).name}")
    else:
        image = image_path

    # Get original image size
    orig_width, orig_height = image.size
    print(f"📏 Original size: {orig_width}x{orig_height}")
    print(f"📏 Target size: {image_size[0]}x{image_size[1]}")

    # Preprocessing
    transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image).unsqueeze(0).to(device)

    # Warmup
    print("\n🔥 Warming up...")
    with torch.no_grad():
        for i in range(10):
            predictions = model.predict([image_tensor])
            if i == 0 and predictions and len(predictions) > 0:
                pred = predictions[0]
                num_detections = len(pred['boxes']) if 'boxes' in pred else 0
                print(f"  Warmup {i+1}: {num_detections} detections")

    # Benchmark
    print(f"\n⏱️  Benchmarking inference for {num_iterations} iterations...")
    times = []
    detection_counts = []

    with torch.no_grad():
        for i in range(num_iterations):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.time()

            predictions = model.predict([image_tensor])

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            inference_time = time.time() - start
            times.append(inference_time)

            # Count detections
            if predictions and len(predictions) > 0:
                pred = predictions[0]
                if 'boxes' in pred:
                    boxes = pred['boxes']
                    scores = pred['scores'] if 'scores' in pred else []

                    # Filter by confidence
                    if len(scores) > 0:
                        keep = scores > confidence_threshold
                        num_detections = keep.sum().item() if hasattr(keep, 'sum') else sum(keep)
                    else:
                        num_detections = len(boxes)
                    detection_counts.append(num_detections)

            if (i + 1) % 20 == 0:
                print(f"  Iteration {i+1}: {inference_time*1000:.1f}ms")

    # Calculate statistics
    if not times:
        print("❌ No timing data collected")
        return None

    times_np = np.array(times)
    avg_time = np.mean(times_np)
    std_time = np.std(times_np)
    min_time = np.min(times_np)
    max_time = np.max(times_np)
    fps = 1.0 / avg_time

    # Detection statistics
    avg_detections = np.mean(detection_counts) if detection_counts else 0

    print(f"\n{'='*60}")
    print("📊 INFERENCE BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"📱 Device: {device}")
    print(f"🖼️  Image size: {orig_width}x{orig_height} → {image_size[0]}x{image_size[1]}")
    print(f"🔄 Iterations: {num_iterations}")
    print(f"🎯 Confidence threshold: {confidence_threshold}")
    print(f"\n⏱️  Timing Statistics:")
    print(f"  • Average: {avg_time*1000:.2f} ms")
    print(f"  • Std Dev: {std_time*1000:.2f} ms")
    print(f"  • Minimum: {min_time*1000:.2f} ms")
    print(f"  • Maximum: {max_time*1000:.2f} ms")
    print(f"  • FPS: {fps:.2f}")
    print(f"\n🎯 Performance Metrics:")
    print(f"  • Throughput: {fps:.1f} images/second")
    print(f"  • Latency (p95): {np.percentile(times_np, 95)*1000:.1f} ms")
    print(f"  • Latency (p99): {np.percentile(times_np, 99)*1000:.1f} ms")
    print(f"  • Average detections: {avg_detections:.1f}")

    # Check memory usage
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.max_memory_allocated() / 1024**3
        memory_reserved = torch.cuda.max_memory_reserved() / 1024**3
        print(f"\n💾 GPU Memory Usage:")
        print(f"  • Allocated: {memory_allocated:.2f} GB")
        print(f"  • Reserved: {memory_reserved:.2f} GB")

    # Save results
    results = {
        'benchmark_type': 'inference',
        'device': str(device),
        'image_size': f"{orig_width}x{orig_height}",
        'target_size': f"{image_size[0]}x{image_size[1]}",
        'iterations': num_iterations,
        'confidence_threshold': confidence_threshold,
        'timing': {
            'avg_ms': float(avg_time * 1000),
            'std_ms': float(std_time * 1000),
            'min_ms': float(min_time * 1000),
            'max_ms': float(max_time * 1000),
            'fps': float(fps)
        },
        'performance': {
            'throughput_fps': float(fps),
            'latency_p95_ms': float(np.percentile(times_np, 95) * 1000),
            'latency_p99_ms': float(np.percentile(times_np, 99) * 1000),
            'avg_detections': float(avg_detections)
        },
        'memory': {
            'allocated_gb': float(memory_allocated) if torch.cuda.is_available() else 0,
            'reserved_gb': float(memory_reserved) if torch.cuda.is_available() else 0
        }
    }

    return results

def benchmark_batch_inference(model, image_dir, batch_sizes=[1, 2, 4, 8], num_iterations=50):
    """Benchmark inference with different batch sizes"""
    model.eval()
    device = next(model.parameters()).device

    print(f"\n{'='*60}")
    print("BATCH INFERENCE BENCHMARK")
    print(f"{'='*60}")

    # Load images
    image_dir = Path(image_dir)
    image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))

    if not image_files:
        print(f"❌ No images found in {image_dir}")
        return None

    print(f"📂 Found {len(image_files)} images")

    results = {}

    for batch_size in batch_sizes:
        print(f"\n📦 Testing batch size: {batch_size}")

        # Create batch
        batch_images = []
        for i in range(min(batch_size, len(image_files))):
            image = Image.open(image_files[i]).convert('RGB')
            transform = T.Compose([
                T.Resize((800, 1333)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image_tensor = transform(image).to(device)
            batch_images.append(image_tensor)

        if not batch_images:
            continue

        # Warmup
        print(f"  Warming up...")
        with torch.no_grad():
            for _ in range(5):
                _ = model.predict(batch_images)

        # Benchmark
        times = []

        with torch.no_grad():
            for i in range(num_iterations):
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start = time.time()

                predictions = model.predict(batch_images)

                torch.cuda.synchronize() if torch.cuda.is_available() else None
                inference_time = time.time() - start
                times.append(inference_time)

        times_np = np.array(times)
        avg_time = np.mean(times_np)
        fps = batch_size / avg_time

        print(f"  • Average batch time: {avg_time*1000:.1f} ms")
        print(f"  • Throughput: {fps:.1f} images/second")
        print(f"  • Batch latency: {avg_time:.3f} seconds")

        results[batch_size] = {
            'avg_time_ms': float(avg_time * 1000),
            'fps': float(fps),
            'throughput': float(fps),
            'batch_size': batch_size
        }

    # Print summary
    print(f"\n{'='*60}")
    print("📊 BATCH INFERENCE SUMMARY")
    print(f"{'='*60}")

    best_throughput = 0
    best_batch_size = 1

    for batch_size, result in results.items():
        print(f"Batch size {batch_size}: {result['fps']:.1f} FPS")
        if result['fps'] > best_throughput:
            best_throughput = result['fps']
            best_batch_size = batch_size

    print(f"\n✅ Optimal batch size: {best_batch_size} ({best_throughput:.1f} FPS)")

    return results

def benchmark_memory(model, input_size=(3, 800, 1333)):
    """Benchmark model memory usage"""
    device = next(model.parameters()).device

    print(f"\n{'='*60}")
    print("MODEL MEMORY BENCHMARK")
    print(f"{'='*60}")

    # Create dummy input
    dummy_input = torch.randn(1, *input_size).to(device)

    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Forward pass
    with torch.no_grad():
        _ = model.predict([dummy_input])

    results = {}

    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**2
        memory_reserved = torch.cuda.memory_reserved() / 1024**2

        print(f"💾 Memory Usage:")
        print(f"  • Allocated: {memory_allocated:.1f} MB")
        print(f"  • Reserved: {memory_reserved:.1f} MB")

        # Model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\n🔢 Model Parameters:")
        print(f"  • Total: {total_params:,}")
        print(f"  • Trainable: {trainable_params:,}")
        print(f"  • Size (FP32): {total_params * 4 / 1024**2:.1f} MB")

        results = {
            'memory_allocated_mb': float(memory_allocated),
            'memory_reserved_mb': float(memory_reserved),
            'total_params': int(total_params),
            'trainable_params': int(trainable_params),
            'model_size_mb': float(total_params * 4 / 1024**2)
        }

    return results

def save_benchmark_results(results, output_dir="benchmark_results"):
    """Save benchmark results to JSON file"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"benchmark_{timestamp}.json"

    # Convert numpy arrays to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.device):
            return str(obj)
        elif isinstance(obj, Path):
            return str(obj)
        return obj

    # Convert results
    serializable_results = json.loads(json.dumps(results, default=convert_for_json))

    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Benchmark results saved to: {results_file}")
    return results_file


def save_benchmark_artifacts(results, output_dir, stem="benchmark"):
    """Save one benchmark result in paired JSON and flattened CSV formats."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    flattened = {}

    def visit(prefix, value):
        if isinstance(value, dict):
            for key, child in value.items():
                visit(f"{prefix}.{key}" if prefix else str(key), child)
        elif isinstance(value, (list, tuple)):
            flattened[prefix] = json.dumps(value)
        else:
            flattened[prefix] = value

    visit("", results)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in flattened.items():
            writer.writerow({"metric": key, "value": value})
    print(f"Benchmark artifacts saved to: {json_path} and {csv_path}")
    return json_path, csv_path

# Di utils/benchmark.py, tambahkan fungsi ini:
def benchmark_inference_pipeline(inference, image_path, num_iterations=100, image_size=(800, 1333), confidence_threshold=0.5):
    """Benchmark inference speed on single image using inference pipeline"""
    model = inference.model
    device = inference.device

    print(f"\n{'='*60}")
    print("INFERENCE PIPELINE BENCHMARK")
    print(f"{'='*60}")

    # Load and preprocess image
    if isinstance(image_path, (str, Path)):
        if not Path(image_path).exists():
            print(f"❌ Image not found: {image_path}")
            return None

        image = Image.open(image_path).convert('RGB')
        print(f"📄 Image loaded: {Path(image_path).name}")
    else:
        image = image_path

    # Get original image size
    orig_width, orig_height = image.size
    print(f"📏 Original size: {orig_width}x{orig_height}")

    # Warmup
    print("\n🔥 Warming up inference pipeline...")
    with torch.no_grad():
        for i in range(10):
            results = inference.predict(image_path, confidence_threshold=confidence_threshold)
            if i == 0 and results:
                num_detections = len(results.get('boxes', []))
                print(f"  Warmup {i+1}: {num_detections} detections")

    # Benchmark
    print(f"\n⏱️  Benchmarking inference for {num_iterations} iterations...")
    times = []
    detection_counts = []

    with torch.no_grad():
        for i in range(num_iterations):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.time()

            results = inference.predict(image_path, confidence_threshold=confidence_threshold)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            inference_time = time.time() - start
            times.append(inference_time)

            # Count detections
            if results:
                num_detections = len(results.get('boxes', []))
                detection_counts.append(num_detections)

            if (i + 1) % 20 == 0:
                print(f"  Iteration {i+1}: {inference_time*1000:.1f}ms")

    # Calculate statistics
    if not times:
        print("❌ No timing data collected")
        return None

    times_np = np.array(times)
    avg_time = np.mean(times_np)
    std_time = np.std(times_np)
    min_time = np.min(times_np)
    max_time = np.max(times_np)
    fps = 1.0 / avg_time

    # Detection statistics
    avg_detections = np.mean(detection_counts) if detection_counts else 0

    print(f"\n{'='*60}")
    print("📊 INFERENCE PIPELINE RESULTS")
    print(f"{'='*60}")
    print(f"📱 Device: {device}")
    print(f"🖼️  Image size: {orig_width}x{orig_height}")
    print(f"🔄 Iterations: {num_iterations}")
    print(f"🎯 Confidence threshold: {confidence_threshold}")
    print(f"\n⏱️  Timing Statistics:")
    print(f"  • Average: {avg_time*1000:.2f} ms")
    print(f"  • Std Dev: {std_time*1000:.2f} ms")
    print(f"  • Minimum: {min_time*1000:.2f} ms")
    print(f"  • Maximum: {max_time*1000:.2f} ms")
    print(f"  • FPS: {fps:.2f}")
    print(f"\n🎯 Performance Metrics:")
    print(f"  • Throughput: {fps:.1f} images/second")
    print(f"  • Latency (p95): {np.percentile(times_np, 95)*1000:.1f} ms")
    print(f"  • Latency (p99): {np.percentile(times_np, 99)*1000:.1f} ms")
    print(f"  • Average detections: {avg_detections:.1f}")

    # Check memory usage
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.max_memory_allocated() / 1024**3
        memory_reserved = torch.cuda.max_memory_reserved() / 1024**3
        print(f"\n💾 GPU Memory Usage:")
        print(f"  • Allocated: {memory_allocated:.2f} GB")
        print(f"  • Reserved: {memory_reserved:.2f} GB")

    # Save results
    results = {
        'benchmark_type': 'inference_pipeline',
        'device': str(device),
        'image_size': f"{orig_width}x{orig_height}",
        'iterations': num_iterations,
        'confidence_threshold': confidence_threshold,
        'timing': {
            'avg_ms': float(avg_time * 1000),
            'std_ms': float(std_time * 1000),
            'min_ms': float(min_time * 1000),
            'max_ms': float(max_time * 1000),
            'fps': float(fps)
        },
        'performance': {
            'throughput_fps': float(fps),
            'latency_p95_ms': float(np.percentile(times_np, 95) * 1000),
            'latency_p99_ms': float(np.percentile(times_np, 99) * 1000),
            'avg_detections': float(avg_detections)
        },
        'memory': {
            'allocated_gb': float(memory_allocated) if torch.cuda.is_available() else 0,
            'reserved_gb': float(memory_reserved) if torch.cuda.is_available() else 0
        }
    }

    return results

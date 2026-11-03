import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import patches
import json
from pathlib import Path
from PIL import Image

# PERBAIKAN: Import dengan path yang benar
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.checkpoint import load_checkpoint

try:
    from models.model import TrashCANModel
except ImportError as e:
    print(f"❌ Import error in inference.py: {e}")
    raise

class TrashCANInference:
    """Inference pipeline for TrashCAN model"""
    
    def __init__(self, model_path, device=None, cfg=None):
        """Initialize inference pipeline - Simplified version"""
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        if cfg is None:
            raise ValueError("TrashCANInference membutuhkan resolved experiment config")
        config = cfg
        
        print(f"📂 Loading model from: {model_path}")
        checkpoint = load_checkpoint(model_path, map_location='cpu')
        
        if isinstance(checkpoint, dict):
            num_classes = checkpoint.get('num_classes', getattr(config, 'NUM_CLASSES', 16))
            checkpoint_config = checkpoint.get('config', {})
            if checkpoint_config.get('dataset_version'):
                config.DATASET_VERSION = checkpoint_config['dataset_version']
            checkpoint_manifest_hash = checkpoint_config.get('split_manifest_sha256')
            active_manifest_hash = getattr(config, 'SPLIT_MANIFEST_SHA256', None)
            if (
                checkpoint_manifest_hash
                and active_manifest_hash
                and checkpoint_manifest_hash != active_manifest_hash
            ):
                raise ValueError(
                    "Checkpoint berasal dari manifest split yang berbeda dari konfigurasi aktif"
                )
            checkpoint_categories = {
                int(key): value for key, value in checkpoint_config.get('category_map', {}).items()
            }
            active_categories = {
                int(key): value for key, value in getattr(config, 'CATEGORY_MAP', {}).items()
            }
            if checkpoint_categories and active_categories and checkpoint_categories != active_categories:
                raise ValueError("Category mapping checkpoint berbeda dari konfigurasi aktif")
            checkpoint_target = checkpoint_config.get('target_class')
            active_target = getattr(config, 'TARGET_CLASS', None)
            if checkpoint_target and active_target and checkpoint_target != active_target:
                raise ValueError("Kelas target checkpoint berbeda dari konfigurasi aktif")
        else:
            num_classes = getattr(config, 'NUM_CLASSES', 16)
        
        print(f"Creating model with {num_classes} object classes")
        self.config = config
        self.config.PRETRAINED = False
        self.model = TrashCANModel(num_classes=num_classes, cfg=config).to(self.device)
        
        # Handle checkpoint format
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"📊 Checkpoint has {len(state_dict)} parameters")
        else:
            state_dict = checkpoint
        
        # Kontrak kelas dan arsitektur harus sama dengan checkpoint.
        try:
            self.model.model.load_state_dict(state_dict, strict=True)
            print(f"✅ Model loaded successfully (strict=True)")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
        
        self.model.eval()
        
        # Load category mapping
        self.category_map = self.load_category_map(config)
        
        # Color palette
        num_colors = len(self.category_map) if self.category_map else 22
        self.colors = plt.cm.tab20(np.linspace(0, 1, num_colors))
        np.random.shuffle(self.colors)
    
    def load_category_map(self, config=None):
        """Load category mapping from COCO JSON"""
        # PERBAIKAN: Gunakan path dari config atau tentukan secara eksplisit
        try:
            train_json = Path(config.TRAIN_JSON)
            
            if train_json.exists():
                with open(train_json, 'r') as f:
                    data = json.load(f)
                
                category_map = {cat['id']: cat['name'] for cat in data['categories']}
                return category_map
            else:
                raise FileNotFoundError(f"COCO JSON not found: {train_json}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load category map: {e}") from e
    
    def preprocess_image(self, image):
        """Preprocess image for model input"""
        if isinstance(image, str) or isinstance(image, Path):
            # Load image from file
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Convert to tensor
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        return image_tensor, image
    
    # inference.py - perbaiki di method predict() atau bagian save JSON
    def predict(self, image, confidence_threshold=0.5, include_visualization_data=False):
        """Run detection inference; masks and source pixels are opt-in for visualization."""
        # Preprocess
        image_tensor, original_image = self.preprocess_image(image)
        
        # Move to device
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            predictions = self.model.predict(image_tensor)
        
        # Post-process
        prediction = predictions[0]
        
        # Filter by confidence
        keep = prediction['scores'] > confidence_threshold
        boxes = prediction['boxes'][keep].cpu().numpy()
        labels = prediction['labels'][keep].cpu().numpy()
        scores = prediction['scores'][keep].cpu().numpy()
        results = {
            'boxes': boxes.tolist(),
            'labels': labels.tolist(),
            'scores': scores.tolist(),
        }
        if include_visualization_data:
            masks = prediction['masks'][keep].cpu().numpy() > 0.5
            results['masks'] = masks.tolist()
            results['original_image'] = np.array(original_image).tolist()
        return results
    
    def visualize(self, image_path, output_path=None, confidence_threshold=0.5):
        """Fixed visualization method - menggantikan implementasi lama"""
        # Run prediction
        results = self.predict(
            image_path,
            confidence_threshold,
            include_visualization_data=True,
        )
        
        # Convert to numpy dengan error handling
        try:
            import numpy as np
            
            # Safely convert
            boxes = np.array(results.get('boxes', []))
            labels = np.array(results.get('labels', []))
            scores = np.array(results.get('scores', []))
            masks = np.array(results.get('masks', [])) if results.get('masks') else np.array([])
            
            # PERBAIKAN: Normalize original_image dengan benar
            if 'original_image' in results:
                if isinstance(results['original_image'], list):
                    original_image = np.array(results['original_image'], dtype=np.float32)
                else:
                    original_image = results['original_image'].astype(np.float32)
                
                # Normalize ke range [0, 1]
                if original_image.max() > 1.0:
                    original_image = original_image / 255.0
            else:
                # Load image manually
                from PIL import Image
                original_image = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32) / 255.0
            
            # PERBAIKAN: Pastikan tidak ada nilai di luar range [0, 1]
            original_image = np.clip(original_image, 0.0, 1.0)
            
            print(f"📊 Visualization data:")
            print(f"  Boxes: {boxes.shape if boxes.size > 0 else 'empty'}")
            print(f"  Labels: {labels.shape if labels.size > 0 else 'empty'}")
            print(f"  Scores: {scores.shape if scores.size > 0 else 'empty'}")
            print(f"  Masks: {masks.shape if masks.size > 0 else 'empty'}")
            print(f"  Image shape: {original_image.shape}")
            print(f"  Image range: [{original_image.min():.3f}, {original_image.max():.3f}]")
            print(f"  Image dtype: {original_image.dtype}")
            
        except Exception as e:
            print(f"❌ Error processing data: {e}")
            return
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
        # PERBAIKAN: Gunakan imshow dengan parameter vmin dan vmin
        # Original image
        axes[0].imshow(original_image, vmin=0.0, vmax=1.0)
        axes[0].set_title(f'Original\n{Path(image_path).name}')
        axes[0].axis('off')
        
        # Bounding boxes only
        axes[1].imshow(original_image, vmin=0.0, vmax=1.0)
        if boxes.size > 0:
            for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
                color = self.colors[int(label) % len(self.colors)]
                x1, y1, x2, y2 = box
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor=color[:3], facecolor='none'
                )
                axes[1].add_patch(rect)
                
                # Label
                category_name = self.category_map.get(int(label), f'Class_{label}')
                label_text = f'{category_name}: {score:.2f}'
                axes[1].text(
                    x1, y1 - 5, label_text,
                    bbox=dict(facecolor=color[:3], alpha=0.8),
                    fontsize=8, color='white'
                )
        axes[1].set_title(f'BBox Detections: {len(boxes)}')
        axes[1].axis('off')
        
        # Masks visualization
        axes[2].imshow(original_image, vmin=0.0, vmax=1.0)
        if masks.size > 0:
            # Create composite mask
            composite_mask = np.zeros((*original_image.shape[:2], 4))
            
            for i, (mask, label) in enumerate(zip(masks, labels)):
                color = self.colors[int(label) % len(self.colors)]
                
                # Process mask
                if mask.ndim == 3:
                    mask_binary = mask[0] > 0.5
                else:
                    mask_binary = mask > 0.5
                
                # Apply color
                for c in range(3):
                    composite_mask[:, :, c] += mask_binary * color[c]
                composite_mask[:, :, 3] += mask_binary * 0.5  # Alpha
            
            composite_mask[:, :, 3] = np.clip(composite_mask[:, :, 3], 0, 0.7)
            axes[2].imshow(composite_mask, alpha=0.7)
        
        axes[2].set_title(f'Segmentation Masks')
        axes[2].axis('off')
        
        plt.suptitle(f'Inference Results (Confidence > {confidence_threshold})', fontsize=16)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✅ Visualization saved to: {output_path}")
        else:
            plt.show()
        
        # Print detection summary
        if boxes.size > 0:
            print(f"\n📋 Detection Summary:")
            print(f"  Total detections: {len(boxes)}")
            print(f"  Confidence range: {scores.min():.3f} - {scores.max():.3f}")
            print(f"  Average confidence: {scores.mean():.3f}")
            
            # Count by class
            unique_labels, counts = np.unique(labels, return_counts=True)
            print(f"  Detections by class:")
            for label, count in zip(unique_labels, counts):
                class_name = self.category_map.get(int(label), f'Class_{label}')
                class_scores = scores[labels == label]
                avg_score = class_scores.mean() if len(class_scores) > 0 else 0
                print(f"    {class_name}: {count} (avg conf: {avg_score:.3f})")
                
    def batch_predict(self, image_dir, output_dir=None, confidence_threshold=0.5):
        """Run inference on batch of images"""
        image_dir = Path(image_dir)
        image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
        
        all_results = {}
        
        for image_file in image_files:
            print(f"Processing: {image_file.name}")
            results = self.predict(image_file, confidence_threshold)
            
            if output_dir:
                # Save visualization
                output_path = Path(output_dir) / f'pred_{image_file.stem}.png'
                self.visualize(str(image_file), str(output_path), confidence_threshold)
            
            all_results[image_file.name] = {
                'num_detections': len(results['boxes']),
                'categories': [self.category_map.get(l, f'Unknown_{l}') 
                              for l in results['labels']],
                'scores': results['scores']
            }
        
        # Save results summary
        if output_dir:
            summary_file = Path(output_dir) / 'results_summary.json'
            with open(summary_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\nResults summary saved to: {summary_file}")
        
        return all_results
    
    # inference.py - tambahkan method evaluate
    def evaluate(self, val_loader, confidence_threshold=0.5):
        """Evaluate model on validation dataset"""
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        import json
        import tempfile
        import os
        
        self.model.eval()
        
        results = []
        image_ids = []
        
        print("Running COCO evaluation...")
        for batch_idx, (images, targets) in enumerate(val_loader):
            # Move to device
            images = [img.to(self.device) for img in images]
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model.predict(images)
            
            # Convert predictions to COCO format
            for i, output in enumerate(outputs):
                image_id = targets[i]['image_id'].item()
                image_ids.append(image_id)
                
                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                masks = output['masks'].cpu().numpy()
                
                for j in range(len(boxes)):
                    if scores[j] < confidence_threshold:
                        continue
                    
                    # Convert mask to RLE
                    mask = masks[j] > 0.5
                    if mask.ndim == 3:
                        mask = mask[0]
                    
                    # Convert to RLE
                    from pycocotools import mask as mask_utils
                    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
                    rle['counts'] = rle['counts'].decode('ascii')
                    
                    # Convert bbox from [x1, y1, x2, y2] to [x, y, width, height]
                    x1, y1, x2, y2 = boxes[j]
                    bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                    
                    result = {
                        'image_id': image_id,
                        'category_id': int(labels[j]),
                        'bbox': bbox,
                        'score': float(scores[j]),
                        'segmentation': rle
                    }
                    results.append(result)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(val_loader)} batches")
        
        if not results:
            print("⚠️  No predictions above confidence threshold")
            return {}
        
        # Save results to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(results, f)
            temp_pred_file = f.name
        
        try:
            # Load ground truth
            # PERBAIKAN: Cari file COCO validation JSON
            import glob
            coco_json_files = glob.glob("*/instances_val*.json") + glob.glob("*_version/instances_val*.json")
            
            if not coco_json_files:
                print("❌ COCO validation JSON not found")
                return {}
            
            val_json_path = coco_json_files[0]
            print(f"📄 Using COCO annotations: {val_json_path}")
            
            coco_gt = COCO(val_json_path)
            
            # Load predictions
            coco_dt = coco_gt.loadRes(temp_pred_file)
            
            # Run COCO evaluation
            coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
            coco_eval.params.imgIds = list(set(image_ids))
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            # Get metrics
            stats = coco_eval.stats
            
            metrics = {
                'AP': float(stats[0]),           # mAP @ [0.5:0.95]
                'AP50': float(stats[1]),         # mAP @ 0.5
                'AP75': float(stats[2]),         # mAP @ 0.75
                'AP_small': float(stats[3]),     # AP for small objects
                'AP_medium': float(stats[4]),    # AP for medium objects
                'AP_large': float(stats[5]),     # AP for large objects
                'AR1': float(stats[6]),          # AR @ 1 detection per image
                'AR10': float(stats[7]),         # AR @ 10 detections per image
                'AR100': float(stats[8]),        # AR @ 100 detections per image
                'AR_small': float(stats[9]),     # AR for small objects
                'AR_medium': float(stats[10]),   # AR for medium objects
                'AR_large': float(stats[11]),    # AR for large objects
                'num_images': len(set(image_ids)),
                'num_predictions': len(results)
            }
            
            return metrics
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_pred_file):
                os.remove(temp_pred_file)

def main():
    """Example usage"""
    # Initialize inference pipeline
    model_path = "output/best_model.pth"  # Path to trained model
    inference = TrashCANInference(model_path)
    
    # Single image inference
    image_path = "test_image.jpg"
    inference.visualize(image_path, confidence_threshold=0.7)
    
    # Batch inference
    # inference.batch_predict("test_images/", "output/predictions/")

if __name__ == "__main__":
    main()

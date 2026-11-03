"""
TrashCAN Dataset Module for COCO format
"""
import torch
import torch.utils.data
from torchvision.datasets import CocoDetection
import torchvision.transforms as T
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import random
import json
import re
from pathlib import Path
from pycocotools.coco import COCO
from config import config as default_config
from utils.reproducibility import make_generator, seed_worker

class TrashCANDataset(CocoDetection):
    """Custom TrashCAN Dataset with augmentation"""
    
    def __init__(self, root, annFile, transforms=None, training=True):
        super().__init__(root, annFile)
        self._transforms = transforms
        self.training = training
        self.coco = COCO(annFile)
        
    def __getitem__(self, idx):
        # Get image and annotations
        img, target = super().__getitem__(idx)
        
        # Convert PIL image to tensor
        img = F.to_tensor(img)
        
        # Extract annotations
        image_id = self.ids[idx]
        annotations = target
        
        # Parse annotations
        boxes = []
        labels = []
        masks = []
        areas = []
        iscrowd = []
        
        for ann in annotations:
            # Bounding box
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            
            # Label
            labels.append(ann['category_id'])
            
            # Mask
            mask = self.coco.annToMask(ann)
            masks.append(mask)
            
            # Area
            areas.append(ann['area'])
            
            # Is crowd
            iscrowd.append(ann['iscrowd'])
        
        # Convert to tensors
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        else:
            # Handle empty annotations
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
            masks = torch.zeros((0, img.shape[1], img.shape[2]), dtype=torch.uint8)
            areas = torch.zeros(0, dtype=torch.float32)
            iscrowd = torch.zeros(0, dtype=torch.int64)
        
        # Create target dictionary
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([image_id]),
            "area": areas,
            "iscrowd": iscrowd
        }
        
        # Apply transforms
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        
        return img, target

class Compose:
    """Compose transforms for both image and target"""
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class RandomHorizontalFlip:
    """Random horizontal flip"""
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            
            if "boxes" in target:
                boxes = target["boxes"]
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                target["boxes"] = boxes
            
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
        
        return image, target

class ToTensor:
    """Convert to tensor"""
    def __call__(self, image, target):
        return image, target

def get_transform(train=True):
    """Get data transformation pipeline"""
    transforms = [ToTensor()]
    
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    
    return Compose(transforms)

def collate_fn(batch):
    """Custom collate function for batch processing"""
    return tuple(zip(*batch))

def _normalize_fraction(data_fraction):
    if data_fraction is None:
        return 1.0
    data_fraction = float(data_fraction)
    if data_fraction > 1.0:
        data_fraction = data_fraction / 100.0
    return max(0.0, min(1.0, data_fraction))

def _apply_dataset_fraction(dataset, data_fraction, seed=42):
    data_fraction = _normalize_fraction(data_fraction)
    if data_fraction >= 1.0:
        return dataset

    if data_fraction <= 0.0:
        raise ValueError("data_fraction must be greater than 0")

    original_size = len(dataset.ids)
    subset_size = max(1, int(original_size * data_fraction))
    rng = random.Random(seed)
    ids = list(dataset.ids)
    rng.shuffle(ids)
    dataset.ids = ids[:subset_size]
    dataset.data_fraction = data_fraction
    dataset.original_size = original_size
    return dataset


class ManifestSubset(torch.utils.data.Dataset):
    """Subset that preserves COCO metadata and the manifest partition identity."""

    def __init__(self, dataset, image_ids, partition):
        self.dataset = dataset
        index_by_id = {int(image_id): index for index, image_id in enumerate(dataset.ids)}
        missing = sorted(set(int(image_id) for image_id in image_ids) - set(index_by_id))
        if missing:
            raise ValueError(f"Image ID manifest tidak ditemukan pada COCO: {missing[:10]}")
        self.indices = [index_by_id[int(image_id)] for image_id in image_ids]
        self.ids = [int(image_id) for image_id in image_ids]
        self.partition = partition
        self.coco = dataset.coco
        self.annotation_file = getattr(dataset.coco, "annotation_file", None)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


def _group_id(file_name, image_id):
    match = re.match(r"^(vid_\d+)_frame\d+", Path(file_name).name, flags=re.IGNORECASE)
    return match.group(1).lower() if match else f"image_{image_id}"


def _manifest_partition_ids(
    dataset,
    partition_data,
    data_fraction,
    seed,
    required_category_ids=None,
    group_aliases=None,
):
    allowed_ids = [int(image_id) for image_id in partition_data["image_ids"]]
    dataset_ids = set(int(image_id) for image_id in dataset.ids)
    if not set(allowed_ids).issubset(dataset_ids):
        raise ValueError("Manifest memuat image ID di luar anotasi train resmi")
    if data_fraction >= 1.0:
        return sorted(allowed_ids)

    group_aliases = group_aliases or {}
    ids_by_group = {}
    for image_id in allowed_ids:
        image = dataset.coco.imgs[image_id]
        raw_group_id = _group_id(image["file_name"], image_id)
        group_id = group_aliases.get(raw_group_id, raw_group_id)
        ids_by_group.setdefault(group_id, []).append(image_id)

    group_ids = sorted(ids_by_group)
    random.Random(seed).shuffle(group_ids)
    target_size = max(1, round(len(allowed_ids) * data_fraction))
    selected_ids = []
    selected_groups = set()
    required_category_ids = [int(value) for value in (required_category_ids or [])]
    for category_id in required_category_ids:
        eligible_groups = []
        for group_id in group_ids:
            annotation_ids = dataset.coco.getAnnIds(
                imgIds=ids_by_group[group_id], catIds=[category_id]
            )
            if annotation_ids:
                eligible_groups.append(group_id)
        if not eligible_groups:
            raise ValueError(
                f"Partisi manifest tidak memiliki kategori wajib {category_id}"
            )
        chosen_group = next(
            (group_id for group_id in eligible_groups if group_id not in selected_groups),
            eligible_groups[0],
        )
        if chosen_group not in selected_groups:
            selected_groups.add(chosen_group)
            selected_ids.extend(ids_by_group[chosen_group])

    for group_id in group_ids:
        if len(selected_ids) >= target_size and selected_ids:
            break
        if group_id in selected_groups:
            continue
        selected_ids.extend(ids_by_group[group_id])
        if len(selected_ids) >= target_size:
            break
    return sorted(selected_ids)


def get_dataloaders(cfg=None, data_fraction=None, manifest_path=None):
    """Create train and validation dataloaders"""
    cfg = cfg or default_config
    if data_fraction is None:
        data_fraction = getattr(cfg, "DATA_FRACTION", 1.0)
    data_fraction = _normalize_fraction(data_fraction)
    seed = getattr(cfg, "RANDOM_SEED", 42)

    resolved_manifest_path = manifest_path or getattr(cfg, "SPLIT_MANIFEST_PATH", None)
    if resolved_manifest_path:
        resolved_manifest_path = Path(resolved_manifest_path)
        with resolved_manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("official_test_role") != "locked_final_test":
            raise ValueError("Manifest tidak mengunci test resmi")

        train_base = TrashCANDataset(
            root=str(cfg.TRAIN_DIR),
            annFile=str(cfg.TRAIN_JSON),
            transforms=get_transform(train=True),
            training=True,
        )
        val_base = TrashCANDataset(
            root=str(cfg.TRAIN_DIR),
            annFile=str(cfg.TRAIN_JSON),
            transforms=get_transform(train=False),
            training=False,
        )
        target_category_id = getattr(cfg, "TARGET_CATEGORY_ID", None)
        required_categories = [target_category_id] if target_category_id is not None else []
        group_aliases = manifest.get("group_rule", {}).get("merged_group_aliases", {})
        train_ids = _manifest_partition_ids(
            train_base,
            manifest["train_internal"],
            data_fraction,
            seed,
            required_category_ids=required_categories,
            group_aliases=group_aliases,
        )
        val_ids = _manifest_partition_ids(
            val_base,
            manifest["val_internal"],
            data_fraction,
            seed + 1,
            required_category_ids=required_categories,
            group_aliases=group_aliases,
        )
        if set(train_ids) & set(val_ids):
            raise ValueError("Terdapat overlap image ID train dan validasi internal")
        train_groups = set(manifest["train_internal"].get("group_ids", []))
        val_groups = set(manifest["val_internal"].get("group_ids", []))
        if train_groups & val_groups:
            raise ValueError("Terdapat overlap group ID train dan validasi internal")
        train_dataset = ManifestSubset(train_base, train_ids, "train_internal")
        val_dataset = ManifestSubset(val_base, val_ids, "val_internal")
        cfg.EVAL_ANNOTATION_JSON = Path(cfg.TRAIN_JSON)
        cfg.EVAL_IMAGE_IDS = val_ids
    else:
        train_dataset = TrashCANDataset(
            root=str(cfg.TRAIN_DIR),
            annFile=str(cfg.TRAIN_JSON),
            transforms=get_transform(train=True),
            training=True,
        )
        val_dataset = TrashCANDataset(
            root=str(cfg.VAL_DIR),
            annFile=str(cfg.VAL_JSON),
            transforms=get_transform(train=False),
            training=False,
        )
        train_dataset = _apply_dataset_fraction(train_dataset, data_fraction, seed=seed)
        val_dataset = _apply_dataset_fraction(val_dataset, data_fraction, seed=seed + 1)

    num_workers = getattr(cfg, "NUM_WORKERS", 4)
    pin_memory = getattr(cfg, "PIN_MEMORY", torch.cuda.is_available()) and torch.cuda.is_available()
    persistent_workers = getattr(cfg, "PERSISTENT_WORKERS", False) and num_workers > 0
    train_generator = make_generator(seed)
    val_generator = make_generator(seed + 1)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=seed_worker,
        generator=train_generator,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,  # Use batch size 1 for validation
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=seed_worker,
        generator=val_generator,
    )
    
    return train_loader, val_loader, train_dataset, val_dataset

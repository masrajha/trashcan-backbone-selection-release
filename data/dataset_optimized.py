# dataset_optimized.py
import torchvision.transforms as T

class SmallResTransform:
    """Transforms for 480x270 images on RTX 3050"""
    
    def __init__(self, training=True):
        self.training = training
        
        # Minimal transforms for small images
        if training:
            self.transforms = T.Compose([
                T.Resize((320, 569)),  # Maintain aspect ~1.78
                T.ColorJitter(0.1, 0.1, 0.1, 0.05),
                T.RandomHorizontalFlip(0.3),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transforms = T.Compose([
                T.Resize((320, 569)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
            ])
    
    def __call__(self, image, target):
        image = self.transforms(image)
        
        # Scale bounding boxes
        if 'boxes' in target:
            # Scale factor
            orig_h, orig_w = 270, 480
            new_h, new_w = 320, 569
            
            scale_x = new_w / orig_w
            scale_y = new_h / orig_h
            
            boxes = target['boxes']
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            target['boxes'] = boxes
            
            # Scale masks if present
            if 'masks' in target:
                # Resize masks
                masks = target['masks']
                # Assuming masks are [N, H, W]
                resized_masks = F.interpolate(
                    masks.unsqueeze(1).float(),
                    size=(new_h, new_w),
                    mode='nearest'
                ).squeeze(1).byte()
                target['masks'] = resized_masks
        
        return image, target
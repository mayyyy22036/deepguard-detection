"""
Techniques d'augmentation avancées pour la Semaine 3
"""

import numpy as np
import torch


class MixUp:
    """
    MixUp augmentation: mélange deux images et leurs labels
    
    Reference: https://arxiv.org/abs/1710.09412
    """
    
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, images, labels):
        """
        Args:
            images: batch de images (B, C, H, W)
            labels: batch de labels (B,)
        
        Returns:
            mixed_images, labels_a, labels_b, lam
        """
        batch_size = images.size(0)
        
        # Sample lambda from Beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        # Random permutation
        index = torch.randperm(batch_size).to(images.device)
        
        # Mix images
        mixed_images = lam * images + (1 - lam) * images[index]
        
        # Mix labels (soft labels)
        labels_a = labels
        labels_b = labels[index]
        
        return mixed_images, labels_a, labels_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Loss function for MixUp/CutMix"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
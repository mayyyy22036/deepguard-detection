import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, List
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image


class DeepfakeDataset(Dataset):
    """
    Dataset PyTorch pour la classification deepfake (Real vs Fake)
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: Optional[A.Compose] = None,
        image_size: int = 224
    ):
        """
        Args:
            data_dir: Chemin vers le dossier data/processed/
            split: 'train', 'val', ou 'test'
            transform: Transformations albumentations
            image_size: Taille des images (défaut: 224x224)
        """
        self.data_dir = Path(data_dir) / split
        self.split = split
        self.image_size = image_size
        
        # Transformations
        if transform is None:
            self.transform = self._get_default_transforms()
        else:
            self.transform = transform
        
       # Mapping des labels (DOIT etre avant _load_samples)
        self.class_to_idx = {'real': 0, 'fake': 1}
        self.idx_to_class = {0: 'real', 1: 'fake'}

        # Charger les chemins des images
        self.samples = self._load_samples()
        
        print(f"Dataset {split}: {len(self.samples)} images")
        print(f"  - Real: {sum(1 for _, label in self.samples if label == 0)}")
        print(f"  - Fake: {sum(1 for _, label in self.samples if label == 1)}")
    
    def _load_samples(self) -> List[Tuple[Path, int]]:
        """Charge tous les chemins d'images avec leurs labels"""
        samples = []
        
        # Parcourir real/ et fake/
        for class_name in ['real', 'fake']:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                print(f"Warning: {class_dir} n'existe pas")
                continue
            
            label = self.class_to_idx[class_name]
            
            # Lister toutes les images
            for img_path in class_dir.glob('*.jpg'):
                samples.append((img_path, label))
        
        return samples
    
    def _get_default_transforms(self) -> A.Compose:
        """Transformations par défaut selon le split"""
        if self.split == 'train':
            # Augmentation pour l'entraînement
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:
            # Val/Test : juste resize et normalize
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Retourne une image et son label
        
        Returns:
            image: Tensor (3, H, W)
            label: 0 (real) ou 1 (fake)
        """
        img_path, label = self.samples[idx]
        
        # Charger l'image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Appliquer les transformations
        transformed = self.transform(image=image)
        image = transformed['image']
        
        return image, label


def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
    image_size: int = 224
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Crée les DataLoaders pour train, val, test
    
    Args:
        data_dir: Chemin vers data/processed/
        batch_size: Taille des batchs
        num_workers: Nombre de workers pour le chargement
        image_size: Taille des images
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Créer les datasets
    train_dataset = DeepfakeDataset(
        data_dir=data_dir,
        split='train',
        image_size=image_size
    )
    
    val_dataset = DeepfakeDataset(
        data_dir=data_dir,
        split='val',
        image_size=image_size
    )
    
    test_dataset = DeepfakeDataset(
        data_dir=data_dir,
        split='test',
        image_size=image_size
    )
    
    # Créer les dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test du dataset
    print("Test du dataset...")
    
    # Créer un dataset
    dataset = DeepfakeDataset(
        data_dir="data/processed",
        split="train"
    )
    
    # Tester un échantillon
    image, label = dataset[0]
    print(f"\nShape de l'image: {image.shape}")
    print(f"Label: {label} ({dataset.idx_to_class[label]})")
    print(f"Type: {type(image)}")
    print(f"Min/Max: {image.min():.3f} / {image.max():.3f}")
    
    # Tester le dataloader
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir="data/processed",
        batch_size=4
    )
    
    # Itérer sur un batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Labels: {labels}")
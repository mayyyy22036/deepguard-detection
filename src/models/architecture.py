"""
Architectures de modèles pour la détection de deepfakes
"""

import torch
import torch.nn as nn
import timm
from typing import Optional


class DeepfakeClassifier(nn.Module):
    """
    Modèle de classification deepfake basé sur EfficientNet
    """
    
    def __init__(
        self,
        model_name: str = 'efficientnet_b0',
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        """
        Args:
            model_name: Nom du modèle timm (ex: 'efficientnet_b0', 'resnet50')
            num_classes: Nombre de classes (2 pour binaire: real/fake)
            pretrained: Utiliser les poids pré-entraînés ImageNet
            dropout: Taux de dropout avant la couche finale
        """
        super(DeepfakeClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Charger le modèle pré-entraîné
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Retirer la tête de classification
            global_pool=''
        )
        
        # Obtenir la dimension des features
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            self.feature_dim = features.shape[1]
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes)
        )
        
        print(f"Model: {model_name}")
        print(f"Feature dim: {self.feature_dim}")
        print(f"Num classes: {num_classes}")
        print(f"Pretrained: {pretrained}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Tensor (batch_size, 3, H, W)
        
        Returns:
            logits: Tensor (batch_size, num_classes)
        """
        # Extract features
        features = self.backbone(x)
        
        # Global pooling
        pooled = self.global_pool(features)
        pooled = pooled.flatten(1)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extraire les features (pour visualisation, Grad-CAM, etc.)"""
        features = self.backbone(x)
        return features


def count_parameters(model: nn.Module) -> int:
    """Compte le nombre de paramètres entraînables"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test du modèle
    print("Test du modèle...")
    
    # Créer le modèle
    model = DeepfakeClassifier(
        model_name='efficientnet_b0',
        num_classes=2,
        pretrained=True
    )
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Test avec CUDA si disponible
    if torch.cuda.is_available():
        print("\nTest avec GPU...")
        model = model.cuda()
        dummy_input = dummy_input.cuda()
        output = model(dummy_input)
        print(f"GPU Output shape: {output.shape}")

def create_efficientnet_b4(num_classes=2, pretrained=True, dropout=0.3):
    """EfficientNet-B4 - Plus grand, meilleur"""
    return DeepfakeClassifier(
        model_name='efficientnet_b4',
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    )


def create_xception(num_classes=2, pretrained=True, dropout=0.3):
    """
    XceptionNet - Architecture spécialisée pour deepfakes
    Performances excellentes dans la littérature
    """
    return DeepfakeClassifier(
        model_name='xception',
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    )


def create_vision_transformer(num_classes=2, pretrained=True, dropout=0.3):
    """Vision Transformer - État de l'art"""
    return DeepfakeClassifier(
        model_name='vit_base_patch16_224',
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    )


# Factory function
def get_model(model_name='efficientnet_b0', **kwargs):
    """
    Crée un modèle par son nom
    
    Args:
        model_name: 'efficientnet_b0', 'efficientnet_b4', 'xception', 'vit'
    
    Returns:
        model instance
    """
    models = {
        'efficientnet_b0': lambda: DeepfakeClassifier('efficientnet_b0', **kwargs),
        'efficientnet_b4': lambda: create_efficientnet_b4(**kwargs),
        'xception': lambda: create_xception(**kwargs),
        'vit': lambda: create_vision_transformer(**kwargs)
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    
    return models[model_name]()
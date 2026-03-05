"""
Boucle d'entraînement avec MLflow tracking
"""

import sys

from numpy.compat import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import mlflow
import mlflow.pytorch
from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


class Trainer:
    """Classe pour entraîner le modèle avec MLflow tracking"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        experiment_name: str = 'deepguard_detection'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        # MLflow
        mlflow.set_experiment(experiment_name)
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
    
    def train_epoch(self) -> Dict[str, float]:
        """Entraîne une epoch"""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            # Métriques
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        # Calculer les métriques
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return {
            'train_loss': epoch_loss,
            'train_acc': epoch_acc
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Valide le modèle"""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        for images, labels in tqdm(self.val_loader, desc='Validation'):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Métriques
            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Proba classe 1 (fake)
        
        # Calculer les métriques
        val_loss = running_loss / len(self.val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        
        # Précision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary'
        )
        
        # AUC-ROC
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.0
        
        return {
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_precision': precision,
            'val_recall': recall,
            'val_f1': f1,
            'val_auc': auc
        }
    
    def train(
        self,
        num_epochs: int,
        early_stopping_patience: int = 5,
        save_path: str = 'models/best_model.pth'
    ):
        """
        Entraîne le modèle pour num_epochs
        
        Args:
            num_epochs: Nombre d'epochs
            early_stopping_patience: Patience pour early stopping
            save_path: Chemin pour sauvegarder le meilleur modèle
        """
        with mlflow.start_run():
            # Log hyperparamètres
            mlflow.log_param('model', self.model.model_name)
            mlflow.log_param('num_epochs', num_epochs)
            mlflow.log_param('batch_size', self.train_loader.batch_size)
            mlflow.log_param('lr', self.optimizer.param_groups[0]['lr'])
            mlflow.log_param('device', self.device)
            
            print(f"\n{'='*60}")
            print(f"Starting training on {self.device}")
            print(f"{'='*60}\n")
            
            for epoch in range(num_epochs):
                print(f"\nEpoch {epoch+1}/{num_epochs}")
                print("-" * 40)
                
                # Train
                train_metrics = self.train_epoch()
                
                # Validate
                val_metrics = self.validate()
                
                # Scheduler step
                if self.scheduler:
                    self.scheduler.step(val_metrics['val_loss'])
                
                # Log métriques
                metrics = {**train_metrics, **val_metrics}
                for key, value in metrics.items():
                    mlflow.log_metric(key, value, step=epoch)
                
                # Print
                print(f"Train Loss: {train_metrics['train_loss']:.4f} | "
                      f"Train Acc: {train_metrics['train_acc']:.4f}")
                print(f"Val Loss: {val_metrics['val_loss']:.4f} | "
                      f"Val Acc: {val_metrics['val_acc']:.4f} | "
                      f"Val AUC: {val_metrics['val_auc']:.4f}")
                
                # Save best model
                if val_metrics['val_acc'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['val_acc']
                    self.best_val_loss = val_metrics['val_loss']
                    self.patience_counter = 0
                    
                    # Save model
                    torch.save(self.model.state_dict(), save_path)
                    mlflow.pytorch.log_model(self.model, "model")
                    print(f"✅ Best model saved! (Acc: {self.best_val_acc:.4f})")
                else:
                    self.patience_counter += 1
                
                # Early stopping
                if self.patience_counter >= early_stopping_patience:
                    print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
                    break
            
            print(f"\n{'='*60}")
            print(f"Training complete!")
            print(f"Best Val Acc: {self.best_val_acc:.4f}")
            print(f"Best Val Loss: {self.best_val_loss:.4f}")
            print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test

    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.append(str(PROJECT_ROOT))
     
    data_path = PROJECT_ROOT / "data" / "processed"
    from src.data.dataset import get_dataloaders
    from src.models.architecture import DeepfakeClassifier
    
    # Dataloaders
    train_loader, val_loader, _ = get_dataloaders(
        "data/processed",
        batch_size=32
    )
    
    # Model
    model = DeepfakeClassifier(
        model_name='efficientnet_b0',
        num_classes=2,
        pretrained=True
    )
    
    # Optimizer & Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler
    )
    
    # Train
    trainer.train(num_epochs=10, early_stopping_patience=5)
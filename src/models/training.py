"""
Boucle d'entraînement avec MLflow tracking
"""

import sys
from pathlib import Path   # ✅ FIX CLAUDE

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
        experiment_name: str = 'deepguard_detection',
        use_mixup: bool = False
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.use_mixup = use_mixup
        self.mixup = None
        if use_mixup:
            from src.data.augmentation import MixUp
            self.mixup = MixUp(alpha=0.2)

        mlflow.set_experiment(experiment_name)

        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({'loss': loss.item()})

        return {
            'train_loss': running_loss / len(self.train_loader),
            'train_acc': accuracy_score(all_labels, all_preds)
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()

        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        for images, labels in tqdm(self.val_loader, desc='Validation'):
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

        val_loss = running_loss / len(self.val_loader)
        val_acc = accuracy_score(all_labels, all_preds)

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary'
        )

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

    def train(self, num_epochs, early_stopping_patience=5, save_path='models/best_model.pth'):

        with mlflow.start_run():

            mlflow.log_param('num_epochs', num_epochs)
            mlflow.log_param('lr', self.optimizer.param_groups[0]['lr'])
            mlflow.log_param('device', self.device)

            for epoch in range(num_epochs):

                print(f"\nEpoch {epoch+1}/{num_epochs}")

                train_metrics = self.train_epoch()
                val_metrics = self.validate()

                if self.scheduler:
                    self.scheduler.step(val_metrics['val_loss'])

                metrics = {**train_metrics, **val_metrics}

                for k, v in metrics.items():
                    mlflow.log_metric(k, v, step=epoch)

                print(f"Train Acc: {train_metrics['train_acc']:.4f}")
                print(f"Val Acc: {val_metrics['val_acc']:.4f}")

                if val_metrics['val_acc'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['val_acc']
                    self.patience_counter = 0

                    torch.save(self.model.state_dict(), save_path)
                    mlflow.pytorch.log_model(self.model, "model")

                    print("Saved best model")
                else:
                    self.patience_counter += 1

                if self.patience_counter >= early_stopping_patience:
                    print("Early stopping")
                    break


if __name__ == "__main__":

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.append(str(PROJECT_ROOT))

    from src.data.dataset import get_dataloaders
    from src.models.architecture import DeepfakeClassifier

    train_loader, val_loader, _ = get_dataloaders(
        str(PROJECT_ROOT / "data/processed"),
        batch_size=32
    )

    model = DeepfakeClassifier(
        model_name='efficientnet_b0',
        num_classes=2,
        pretrained=True
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # ✅ FIX CLAUDE

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler
    )

    trainer.train(num_epochs=20, early_stopping_patience=7).
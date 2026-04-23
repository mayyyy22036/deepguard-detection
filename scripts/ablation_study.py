"""
Étude d'ablation pour comprendre l'impact de chaque composant
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from itertools import product

from src.data.dataset import get_dataloaders
from src.models.architecture import get_model
from src.models.training import Trainer


def run_ablation_experiment(config):
    """
    Lance une expérience avec une configuration donnée
    
    Args:
        config: dict avec les hyperparamètres
    
    Returns:
        dict avec les résultats
    """
    print(f"\nRunning: {config}")
    
    # Data
    train_loader, val_loader, _ = get_dataloaders(
        "data/processed",
        batch_size=config['batch_size']
    )
    
    # Model
    model = get_model(
        model_name='efficientnet_b0',
        num_classes=2,
        pretrained=True,
        dropout=config['dropout']
    )
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        use_mixup=config['use_mixup']
    )
    
    # Train
    trainer.train(num_epochs=10, early_stopping_patience=5)
    
    return {
        **config,
        'val_acc': trainer.best_val_acc,
        'val_loss': trainer.best_val_loss
    }


def main():
    """Étude d'ablation complète"""
    
    # Configurations à tester
    ablation_configs = []
    
    # 1. Baseline
    ablation_configs.append({
        'name': 'baseline',
        'dropout': 0.3,
        'lr': 0.0001,
        'weight_decay': 0.0001,
        'batch_size': 32,
        'use_mixup': False
    })
    
    # 2. Sans dropout
    ablation_configs.append({
        'name': 'no_dropout',
        'dropout': 0.0,
        'lr': 0.0001,
        'weight_decay': 0.0001,
        'batch_size': 32,
        'use_mixup': False
    })
    
    # 3. Dropout élevé
    ablation_configs.append({
        'name': 'high_dropout',
        'dropout': 0.5,
        'lr': 0.0001,
        'weight_decay': 0.0001,
        'batch_size': 32,
        'use_mixup': False
    })
    
    # 4. Learning rate élevé
    ablation_configs.append({
        'name': 'high_lr',
        'dropout': 0.3,
        'lr': 0.001,
        'weight_decay': 0.0001,
        'batch_size': 32,
        'use_mixup': False
    })
    
    # 5. Avec MixUp
    ablation_configs.append({
        'name': 'with_mixup',
        'dropout': 0.3,
        'lr': 0.0001,
        'weight_decay': 0.0001,
        'batch_size': 32,
        'use_mixup': True
    })
    
    # 6. Batch size plus grand
    ablation_configs.append({
        'name': 'large_batch',
        'dropout': 0.3,
        'lr': 0.0001,
        'weight_decay': 0.0001,
        'batch_size': 64,
        'use_mixup': False
    })
    
    # Lancer toutes les expériences
    results = []
    for config in ablation_configs:
        result = run_ablation_experiment(config)
        results.append(result)
    
    # Créer un DataFrame
    df = pd.DataFrame(results)
    
    print(f"\n{'='*60}")
    print("ABLATION STUDY RESULTS")
    print(f"{'='*60}\n")
    print(df.to_string(index=False))
    
    # Sauvegarder
    df.to_csv('models/ablation_results.csv', index=False)
    
    # Analyse
    baseline = df[df['name'] == 'baseline']['val_acc'].values[0]
    
    print(f"\n📊 IMPACT ANALYSIS (vs baseline: {baseline:.4f}):")
    for _, row in df.iterrows():
        if row['name'] != 'baseline':
            diff = row['val_acc'] - baseline
            symbol = "📈" if diff > 0 else "📉"
            print(f"{symbol} {row['name']:15s}: {row['val_acc']:.4f} ({diff:+.4f})")


if __name__ == "__main__":
    main()
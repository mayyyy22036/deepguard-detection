"""
Optimisation des hyperparamètres avec Optuna
"""

import optuna
from optuna.integration.mlflow import MLflowCallback
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import mlflow

from src.data.dataset import get_dataloaders
from src.models.architecture import DeepfakeClassifier
from src.models.training import Trainer


def objective(trial):
    """
    Fonction objective pour Optuna
    
    Args:
        trial: Optuna trial object
    
    Returns:
        validation accuracy (à maximiser)
    """
    # Hyperparamètres à optimiser
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    model_name = trial.suggest_categorical('model_name', 
                                           ['efficientnet_b0', 'efficientnet_b2', 'efficientnet_b4'])
    
    # Data loaders
    train_loader, val_loader, _ = get_dataloaders(
        "data/processed",
        batch_size=batch_size,
        num_workers=2
    )
    
    # Model
    model = DeepfakeClassifier(
        model_name=model_name,
        num_classes=2,
        pretrained=True,
        dropout=dropout
    )
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=2,
        factor=0.5
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        experiment_name='deepguard_optuna'
    )
    
    # Train for limited epochs (faster tuning)
    trainer.train(
        num_epochs=5,  # Court pour tuning
        early_stopping_patience=3,
        save_path=f'models/optuna_trial_{trial.number}.pth'
    )
    
    return trainer.best_val_acc


def run_hyperparameter_search(n_trials=20):
    """
    Lance la recherche d'hyperparamètres
    
    Args:
        n_trials: Nombre d'essais Optuna
    """
    # Créer une study
    study = optuna.create_study(
        study_name='deepguard_tuning',
        direction='maximize',  # Maximiser val_acc
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # MLflow callback
    mlflow_callback = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        metric_name='val_acc'
    )
    
    # Lancer l'optimisation
    print(f"\n{'='*60}")
    print(f"HYPERPARAMETER SEARCH - {n_trials} trials")
    print(f"{'='*60}\n")
    
    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[mlflow_callback],
        show_progress_bar=True
    )
    
    # Meilleurs paramètres
    print(f"\n{'='*60}")
    print("BEST HYPERPARAMETERS")
    print(f"{'='*60}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best val_acc: {study.best_value:.4f}")
    print(f"\nBest params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Sauvegarder les résultats
    import json
    results = {
        'best_trial': study.best_trial.number,
        'best_value': study.best_value,
        'best_params': study.best_params
    }
    
    with open('models/optuna_best_params.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Best params saved to models/optuna_best_params.json")
    
    # Visualization
    try:
        import optuna.visualization as vis
        
        # Importance des hyperparamètres
        fig = vis.plot_param_importances(study)
        fig.write_html('models/optuna_param_importance.html')
        
        # Historique d'optimisation
        fig = vis.plot_optimization_history(study)
        fig.write_html('models/optuna_history.html')
        
        print(f"✅ Visualizations saved to models/")
    except:
        print("⚠️  Visualization skipped (install plotly if needed)")
    
    return study


if __name__ == "__main__":
    # Lancer la recherche
    study = run_hyperparameter_search(n_trials=15)


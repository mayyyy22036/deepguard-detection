import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn

# Démarrer une expérience
mlflow.set_experiment("deepguard_detection")

with mlflow.start_run(run_name="test_mlflow"):
    # Logger des paramètres
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("model", "efficientnet_b4")
    
    # Logger des métriques
    for epoch in range(5):
        mlflow.log_metric("accuracy", 0.7 + epoch * 0.05, step=epoch)
        mlflow.log_metric("loss", 0.5 - epoch * 0.08, step=epoch)
    
    # Créer un modèle dummy et le logger
    model = nn.Linear(10, 2)
    mlflow.pytorch.log_model(model, "model")
    
    print("✅ MLflow test réussi!")

print("\n📊 Vérifier les résultats sur: http://localhost:5000")

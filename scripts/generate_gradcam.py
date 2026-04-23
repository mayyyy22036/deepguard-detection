"""
Génère les visualisations Grad-CAM pour la présentation
Usage: python scripts/generate_gradcam.py
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import timm
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

from src.explainability.gradcam import GradCAM, apply_heatmap


# ---- Modèle ----
class XceptionDeepfake(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('xception', pretrained=False,
                                          num_classes=0, global_pool='avg')
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(2048, 2))
    def forward(self, x):
        return self.classifier(self.backbone(x))

model = XceptionDeepfake()
model.load_state_dict(torch.load('models/deepguard_xception.pth', map_location='cpu'))
model.eval()
print('✅ Modèle Xception chargé')

gradcam = GradCAM(model)

transform = A.Compose([
    A.Resize(299, 299),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    A.pytorch.ToTensorV2()
])

# ---- Générer visualisations ----
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle('DeepGuard — Grad-CAM Visualization\nXception trained on FF++ + Celeb-DF v2',
             fontsize=14, fontweight='bold')

examples = [
    ('REAL', 'data/processed/test/real', 0),
    ('FAKE', 'data/processed/test/fake', 1),
]

for row, (label, folder, class_idx) in enumerate(examples):
    files = os.listdir(folder)[:2]
    for col, fname in enumerate(files):
        img_path = os.path.join(folder, fname)
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_display = cv2.resize(img_rgb, (299, 299))

        # Prédiction + Grad-CAM
        t = transform(image=img_rgb)['image'].unsqueeze(0)
        t.requires_grad_(True)
        cam, pred_idx = gradcam.generate(t, class_idx)
        pred_label = 'FAKE' if pred_idx == 1 else 'REAL'

        # Probas
        with torch.no_grad():
            probs = torch.softmax(model(t), dim=1)[0]
        fake_prob = probs[1].item() * 100

        overlay = apply_heatmap(img_display, cam)
        color = 'green' if pred_label == label else 'red'

        # Original
        axes[row, col*2].imshow(img_display)
        axes[row, col*2].set_title(f'Ground Truth: {label}', fontsize=9, fontweight='bold')
        axes[row, col*2].axis('off')

        # Grad-CAM
        axes[row, col*2+1].imshow(overlay)
        status = '✓' if pred_label == label else '✗'
        axes[row, col*2+1].set_title(
            f'Pred: {pred_label} {status}\nP(fake)={fake_prob:.1f}%',
            fontsize=9, color=color, fontweight='bold'
        )
        axes[row, col*2+1].axis('off')

plt.tight_layout()
os.makedirs('results', exist_ok=True)
plt.savefig('results/gradcam_visualization.png', dpi=150, bbox_inches='tight')
print('✅ Sauvegardé : results/gradcam_visualization.png')
plt.close()
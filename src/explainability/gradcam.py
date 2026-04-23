"""
Grad-CAM pour DeepGuard — Semaine 4
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            # Clone pour éviter le conflit inplace ReLU
            self.activations = output.clone().detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].clone().detach()

        # Hook sur le bloc conv avant le pooling global
        target = self.model.backbone.conv4
        target.register_forward_hook(forward_hook)
        target.register_full_backward_hook(backward_hook)

    def generate(self, image_tensor, class_idx=None):
        self.model.eval()

        # Désactiver inplace pour éviter le conflit
        for module in self.model.modules():
            if isinstance(module, torch.nn.ReLU):
                module.inplace = False

        output = self.model(image_tensor)

        if class_idx is None:
            class_idx = output.argmax().item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        # Grad-CAM
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(299, 299), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, class_idx


def apply_heatmap(image_np, cam, alpha=0.4):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = alpha * heatmap + (1 - alpha) * image_np
    return np.clip(overlay, 0, 255).astype(np.uint8)
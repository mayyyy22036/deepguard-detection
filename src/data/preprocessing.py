import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm

class VideoFrameExtractor:
    """Extrait des frames uniformément espacées d'une vidéo"""
    
    def __init__(self, num_frames: int = 32):
        self.num_frames = num_frames
    
    def extract_frames(self, video_path: str) -> List[np.ndarray]:
        """
        Extrait num_frames frames uniformément espacées
        
        Args:
            video_path: Chemin vers la vidéo
            
        Returns:
            Liste de frames (numpy arrays)
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < self.num_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        return frames

class FaceDetector:
    """Détecte et extrait les visages dans les images"""
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.mtcnn = MTCNN(
            keep_all=False,  # Garder seulement le visage le plus grand
            device=self.device,
            post_process=False
        )
    
    def detect_face(self, frame: np.ndarray, margin: float = 0.2) -> Optional[np.ndarray]:
        """
        Détecte et recadre un visage dans une frame
        
        Args:
            frame: Image (numpy array)
            margin: Marge autour du visage (0.2 = 20%)
            
        Returns:
            Visage recadré ou None si pas détecté
        """
        boxes, probs = self.mtcnn.detect(frame)
        
        if boxes is None:
            return None
        
        # Prendre le visage avec la plus haute probabilité
        box = boxes[0]
        x1, y1, x2, y2 = [int(b) for b in box]
        
        # Ajouter une marge
        h, w = frame.shape[:2]
        margin_x = int((x2 - x1) * margin)
        margin_y = int((y2 - y1) * margin)
        
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(w, x2 + margin_x)
        y2 = min(h, y2 + margin_y)
        
        face = frame[y1:y2, x1:x2]
        return face

def process_video(
    video_path: str,
    output_dir: str,
    num_frames: int = 32,
    face_size: Tuple[int, int] = (224, 224)
) -> bool:
    """
    Pipeline complet: extraction frames + détection visages
    
    Args:
        video_path: Chemin vers la vidéo
        output_dir: Dossier de sortie
        num_frames: Nombre de frames à extraire
        face_size: Taille des visages recadrés
        
    Returns:
        True si succès, False sinon
    """
    extractor = VideoFrameExtractor(num_frames)
    detector = FaceDetector()
    
    # Extraire les frames
    frames = extractor.extract_frames(video_path)
    
    if not frames:
        return False
    
    # Créer le dossier de sortie
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Traiter chaque frame
    video_name = Path(video_path).stem
    saved_count = 0
    
    for i, frame in enumerate(frames):
        face = detector.detect_face(frame)
        
        if face is not None:
            # Redimensionner
            face_resized = cv2.resize(face, face_size)
            
            # Sauvegarder
            output_file = output_path / f"{video_name}_frame_{i:03d}.jpg"
            cv2.imwrite(str(output_file), cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR))
            saved_count += 1
    
    return saved_count > 0

if __name__ == "__main__":
    # Test
    test_video = "path/to/test/video.mp4"
    test_output = "data/processed/test"
    
    success = process_video(test_video, test_output)
    print(f"✅ Processing {'successful' if success else 'failed'}")
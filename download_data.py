"""
Script de téléchargement des datasets de deepfakes
FaceForensics++, Celeb-DF, DFDC
"""

import os
import argparse
import subprocess
import json
from pathlib import Path
from tqdm import tqdm

def download_faceforensics(output_dir, compression="c23", dataset_type="all"):
    """
    Télécharge FaceForensics++
    
    Args:
        output_dir: Dossier de destination
        compression: 'c23' (légère) ou 'c40' (forte)
        dataset_type: 'all', 'original', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures'
    
    Note:
        Nécessite d'avoir reçu l'accès au dataset via le formulaire:
        https://github.com/ondyari/FaceForensics/blob/master/dataset/README.md
    """
    print(f"📥 Téléchargement de FaceForensics++ ({compression})...")
    
    # Créer le dossier de destination
    output_path = Path(output_dir) / "faceforensics" / compression
    output_path.mkdir(parents=True, exist_ok=True)
    
    # URL du script de téléchargement
    download_script_url = "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/download-FaceForensics.py"
    
    print("\n⚠️  IMPORTANT:")
    print("1. Vous devez d'abord demander l'accès au dataset")
    print("2. Remplir le formulaire: https://github.com/ondyari/FaceForensics")
    print("3. Vous recevrez un email avec vos identifiants")
    print("4. Ensuite, téléchargez le script officiel et exécutez-le:")
    print(f"\n   wget {download_script_url}")
    print(f"   python download-FaceForensics.py \\")
    print(f"       --output_path {output_path} \\")
    print(f"       --compression {compression} \\")
    print(f"       --dataset {dataset_type}")
    print("\n" + "="*70)
    
    return output_path

def download_celebdf(output_dir):
    """
    Télécharge Celeb-DF v2
    
    Args:
        output_dir: Dossier de destination
    
    Note:
        Accès via: https://github.com/yuezunli/celeb-deepfakeforensics
    """
    print("📥 Téléchargement de Celeb-DF v2...")
    
    output_path = Path(output_dir) / "celeb_df"
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n⚠️  Pour télécharger Celeb-DF:")
    print("1. Accéder à: https://github.com/yuezunli/celeb-deepfakeforensics")
    print("2. Suivre les instructions pour obtenir l'accès")
    print("3. Utiliser le script de téléchargement fourni")
    print("\n" + "="*70)
    
    return output_path

def download_dfdc(output_dir):
    """
    Télécharge DFDC Preview Dataset
    
    Args:
        output_dir: Dossier de destination
    
    Note:
        Disponible sur Kaggle: https://www.kaggle.com/c/deepfake-detection-challenge
    """
    print("📥 Téléchargement de DFDC Preview...")
    
    output_path = Path(output_dir) / "dfdc"
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n⚠️  Pour télécharger DFDC:")
    print("1. Créer un compte Kaggle: https://www.kaggle.com")
    print("2. Installer kaggle CLI: pip install kaggle")
    print("3. Configurer API token: https://www.kaggle.com/docs/api")
    print("4. Télécharger avec:")
    print(f"   kaggle competitions download -c deepfake-detection-challenge -p {output_path}")
    print("\n" + "="*70)
    
    return output_path

def create_dataset_structure(data_dir):
    """
    Crée la structure de dossiers pour les datasets
    """
    structure = {
        "faceforensics": {
            "c23": ["original_sequences", "manipulated_sequences", "splits"],
            "c40": ["original_sequences", "manipulated_sequences", "splits"]
        },
        "celeb_df": ["real", "fake"],
        "dfdc": ["train", "test"]
    }
    
    data_path = Path(data_dir)
    
    for dataset, subdirs in structure.items():
        if isinstance(subdirs, dict):
            for compression, folders in subdirs.items():
                for folder in folders:
                    folder_path = data_path / dataset / compression / folder
                    folder_path.mkdir(parents=True, exist_ok=True)
        else:
            for folder in subdirs:
                folder_path = data_path / dataset / folder
                folder_path.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Structure de dossiers créée dans: {data_path}")

def verify_dataset(dataset_path, dataset_name="faceforensics"):
    """
    Vérifie l'intégrité du dataset téléchargé
    """
    path = Path(dataset_path)
    
    if not path.exists():
        print(f"❌ Dataset non trouvé: {path}")
        return False
    
    if dataset_name == "faceforensics":
        required_dirs = ["original_sequences", "manipulated_sequences"]
        for dir_name in required_dirs:
            dir_path = path / dir_name
            if not dir_path.exists():
                print(f"❌ Dossier manquant: {dir_path}")
                return False
        
        # Compter les vidéos
        video_count = len(list(path.rglob("*.mp4")))
        print(f"✅ Dataset valide - {video_count} vidéos trouvées")
        return True
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Téléchargement des datasets de deepfakes")
    
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["faceforensics", "celebdf", "dfdc", "all"],
        default="faceforensics",
        help="Dataset à télécharger"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Dossier de destination"
    )
    
    parser.add_argument(
        "--compression",
        type=str,
        choices=["c23", "c40"],
        default="c23",
        help="Niveau de compression pour FaceForensics++ (c23=léger, c40=fort)"
    )
    
    parser.add_argument(
        "--create_structure",
        action="store_true",
        help="Créer seulement la structure de dossiers"
    )
    
    args = parser.parse_args()
    
    if args.create_structure:
        create_dataset_structure(args.output_dir)
        return
    
    if args.dataset in ["faceforensics", "all"]:
        download_faceforensics(args.output_dir, args.compression)
    
    if args.dataset in ["celebdf", "all"]:
        download_celebdf(args.output_dir)
    
    if args.dataset in ["dfdc", "all"]:
        download_dfdc(args.output_dir)
    
    print("\n✅ Instructions de téléchargement affichées!")
    print("📝 N'oubliez pas de demander l'accès aux datasets avant de télécharger.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Configure le meilleur modèle depuis un checkpoint spécifique.
Utile après un entraînement où le modèle a divergé (NaN).

Usage:
    python scripts/setup_best_model.py --checkpoint 2
"""

import os
import sys
import argparse
import shutil
import json
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_best_model(checkpoint_iter: int, checkpoint_dir: str, model_dir: str):
    """
    Copie un checkpoint comme meilleur modèle.

    Args:
        checkpoint_iter: Numéro de l'itération du checkpoint
        checkpoint_dir: Répertoire des checkpoints
        model_dir: Répertoire des modèles
    """
    checkpoint_dir = Path(checkpoint_dir)
    model_dir = Path(model_dir)

    # Fichiers source
    weights_src = checkpoint_dir / f"checkpoint_iter_{checkpoint_iter}.weights.h5"
    meta_src = checkpoint_dir / f"checkpoint_iter_{checkpoint_iter}_meta.json"

    # Fichiers destination
    weights_dst = model_dir / "best_model.weights.h5"
    meta_dst = model_dir / "best_model_meta.json"

    # Vérifier que les sources existent
    if not weights_src.exists():
        print(f"Erreur: Checkpoint non trouvé: {weights_src}")
        return False

    print(f"Configuration du meilleur modèle depuis l'itération {checkpoint_iter}")
    print(f"  Source: {weights_src}")
    print(f"  Destination: {weights_dst}")

    # Copier les poids
    try:
        # Lire le fichier source en binaire
        with open(weights_src, 'rb') as f:
            weights_data = f.read()

        # Écrire le fichier destination
        with open(weights_dst, 'wb') as f:
            f.write(weights_data)

        print(f"  Poids copiés ({len(weights_data) / 1024 / 1024:.2f} MB)")
    except Exception as e:
        print(f"Erreur lors de la copie des poids: {e}")
        return False

    # Copier/mettre à jour les métadonnées
    if meta_src.exists():
        with open(meta_src, 'r') as f:
            meta = json.load(f)
    else:
        meta = {"iteration": checkpoint_iter}

    # Ajouter une note
    meta['note'] = f"Restauré depuis checkpoint iteration {checkpoint_iter}"
    meta['network_config']['learning_rate'] = 0.0001  # Nouveau learning rate

    with open(meta_dst, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"  Métadonnées mises à jour")
    print(f"\nModèle configuré avec succès!")
    print(f"Learning rate recommandé: 0.0001 (10x plus bas qu'avant)")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Configure le meilleur modèle depuis un checkpoint'
    )
    parser.add_argument('--checkpoint', '-c', type=int, default=2,
                       help='Numéro de l\'itération du checkpoint (défaut: 2)')
    parser.add_argument('--checkpoint-dir', type=str, default='data/checkpoints',
                       help='Répertoire des checkpoints')
    parser.add_argument('--model-dir', type=str, default='data/models',
                       help='Répertoire des modèles')

    args = parser.parse_args()

    success = setup_best_model(
        checkpoint_iter=args.checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        model_dir=args.model_dir
    )

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())

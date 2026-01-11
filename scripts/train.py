#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script d'entraînement AlphaZero pour Quarto.

Deux architectures disponibles:
- LOCAL (défaut): Réseau local par worker, 3-5x plus rapide
- SERVER: InferenceServer centralisé (ancienne architecture)

Usage:
    python scripts/train.py --iterations 100 --games-per-iter 50
    python scripts/train.py --quick  # Mode rapide pour test
    python scripts/train.py --evaluate  # Évaluer le modèle actuel
"""

import argparse
import sys
import os
import warnings

# Supprimer les warnings non-critiques
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from alphaquarto.ai.network import AlphaZeroNetwork
from alphaquarto.ai.trainer import AlphaZeroTrainer, Evaluator
from alphaquarto.ai.trainer_local import AlphaZeroTrainerLocal
from alphaquarto.utils.config import (
    Config, NetworkConfig, MCTSConfig, InferenceConfig, TrainingConfig
)


def get_device_info():
    """Affiche les informations sur le device disponible."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {device_name} ({memory:.1f} GB)")
        return "cuda"
    else:
        print("  Device: CPU (GPU non disponible)")
        return "cpu"


def build_config(args) -> Config:
    """Construit la configuration à partir des arguments CLI."""
    # Configuration réseau selon la taille
    network_sizes = {
        'small': {'num_filters': 32, 'num_res_blocks': 3},
        'medium': {'num_filters': 64, 'num_res_blocks': 6},
        'large': {'num_filters': 128, 'num_res_blocks': 10}
    }
    net_params = network_sizes.get(args.network_size, network_sizes['medium'])

    # Device
    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"

    return Config(
        network=NetworkConfig(
            num_filters=net_params['num_filters'],
            num_res_blocks=net_params['num_res_blocks'],
            learning_rate=args.learning_rate,
            l2_reg=1e-4
        ),
        mcts=MCTSConfig(
            num_simulations=args.mcts_sims,
            c_puct=1.5,
            dirichlet_alpha=0.8,
            dirichlet_epsilon=0.25,
            temperature_threshold=10
        ),
        inference=InferenceConfig(
            max_batch_size=64,
            batch_timeout_ms=10.0,
            device=device
        ),
        training=TrainingConfig(
            iterations=args.iterations,
            games_per_iteration=args.games_per_iter,
            epochs_per_iteration=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.workers,
            buffer_size=args.buffer_size,
            min_buffer_size=args.batch_size * 4,
            checkpoint_dir=args.checkpoint_dir,
            model_dir=args.model_dir,
            save_frequency=args.save_frequency,
            clear_buffer_after_training=args.clear_buffer
        )
    )


def train(args):
    """Lance l'entraînement."""
    print(f"\n{'='*60}")
    print("INITIALISATION")
    print(f"{'='*60}")

    # Configuration matérielle
    print("\nConfiguration matérielle...")
    device = get_device_info()

    # Construire la configuration
    config = build_config(args)
    print(f"\nConfiguration réseau ({args.network_size})...")

    # Créer le trainer selon l'architecture choisie
    print("\nInitialisation du trainer...")
    if args.arch == 'local':
        print("  Architecture: LOCAL (réseau par worker, rapide)")
        trainer = AlphaZeroTrainerLocal(config)
    else:
        print("  Architecture: SERVER (InferenceServer centralisé)")
        trainer = AlphaZeroTrainer(config)

    print(f"  Paramètres: {trainer.network.count_parameters():,}")

    if args.workers > 1:
        print(f"  Workers: {args.workers}")

    # Charger un checkpoint existant si spécifié
    if args.resume_from:
        print(f"\nChargement du checkpoint {args.resume_from}...")
        trainer.load_checkpoint(args.resume_from)
        print(f"  Reprise à l'itération {trainer.iteration}")

    # Lancer l'entraînement
    print("\nDémarrage de l'entraînement...")
    try:
        trainer.train(
            num_iterations=args.iterations,
            verbose=True
        )
    except KeyboardInterrupt:
        print("\n\n*** Entraînement interrompu par l'utilisateur ***")
        print("Sauvegarde du modèle actuel...")
        trainer.save_best_model()
        trainer.save_training_stats()
        print(f"Modèle sauvé à l'itération {trainer.iteration}")
        print("Vous pouvez reprendre avec: --resume-from", trainer.iteration)
        return 0

    return 0


def evaluate(args):
    """Évalue le modèle actuel."""
    print(f"\n{'='*60}")
    print("ÉVALUATION DU MODÈLE")
    print(f"{'='*60}")

    # Configuration
    config = build_config(args)
    device = torch.device(config.inference.device)

    # Charger le réseau
    print(f"\nCréation du réseau ({args.network_size})...")
    network = AlphaZeroNetwork(config.network)
    network.to(device)
    network.eval()

    # Charger les poids
    model_path = os.path.join(args.model_dir, "best_model.pt")
    if os.path.exists(model_path):
        print(f"Chargement du modèle: {model_path}")
        network.load_state_dict(
            torch.load(model_path, map_location=device)
        )
    else:
        print("Aucun modèle entraîné trouvé. Utilisation du réseau aléatoire.")

    # Évaluer
    evaluator = Evaluator(network=network, config=config.mcts)

    print(f"\nÉvaluation contre joueur aléatoire ({args.eval_games} parties)...")
    results = evaluator.evaluate_vs_random(
        num_games=args.eval_games,
        verbose=args.verbose
    )

    print(f"\n{'='*60}")
    print("RÉSULTATS")
    print(f"{'='*60}")
    print(f"  Victoires réseau: {results['network_wins']} ({results['network_winrate']*100:.1f}%)")
    print(f"  Victoires random: {results['random_wins']} ({results['random_winrate']*100:.1f}%)")
    print(f"  Matchs nuls: {results['draws']}")

    return 0


def quick_test(args):
    """Test rapide du système."""
    print(f"\n{'='*60}")
    print("TEST RAPIDE")
    print(f"{'='*60}")

    # Configuration minimale
    config = Config.quick_test()

    # Créer le trainer avec architecture locale (rapide)
    print("\nCréation d'un petit réseau...")
    print(f"  Architecture: LOCAL (réseau par worker)")
    trainer = AlphaZeroTrainerLocal(config)
    print(f"  Paramètres: {trainer.network.count_parameters():,}")
    print(f"  Device: {trainer.device}")

    # Une itération rapide
    print("\nItération de test (4 parties, 2 workers)...")

    try:
        trainer.train(num_iterations=1, verbose=True)
    except Exception as e:
        print(f"\nErreur: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print(f"\n{'='*60}")
    print("TEST RÉUSSI")
    print(f"{'='*60}")
    print(f"  Parties jouées: {trainer.total_games}")
    print(f"  Buffer: {len(trainer.replay_buffer)} exemples")
    if trainer.training_stats:
        stats = trainer.training_stats[-1]
        if 'training' in stats and stats['training']:
            print(f"  Loss: {stats['training'].get('loss', 'N/A'):.4f}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Entraînement AlphaZero pour Quarto (PyTorch)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Entraînement standard
  python scripts/train.py --iterations 100 --games-per-iter 50

  # Entraînement PARALLÈLE avec GPU (recommandé)
  python scripts/train.py --iterations 100 --games-per-iter 50 --workers 4

  # Entraînement intensif (machines puissantes, A100)
  python scripts/train.py --iterations 100 --games-per-iter 100 --workers 20

  # Test rapide
  python scripts/train.py --quick

  # Évaluation du modèle
  python scripts/train.py --evaluate

  # Reprendre un entraînement
  python scripts/train.py --resume-from 7 --iterations 50 --workers 4

  # Forcer CPU uniquement (debug ou comparaison)
  python scripts/train.py --no-gpu --iterations 10 --games-per-iter 10
        """
    )

    # Mode
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--quick', action='store_true',
                           help='Test rapide du système')
    mode_group.add_argument('--evaluate', action='store_true',
                           help='Évaluer le modèle actuel')

    # Paramètres d'entraînement
    parser.add_argument('--iterations', type=int, default=100,
                       help='Nombre d\'itérations d\'entraînement (défaut: 100)')
    parser.add_argument('--games-per-iter', type=int, default=200,
                       help='Parties de self-play par itération (défaut: 200)')
    parser.add_argument('--epochs', type=int, default=2,
                       help='Époques d\'entraînement par itération (défaut: 2, évite l\'overfitting)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Taille des batches (défaut: 32)')
    parser.add_argument('--mcts-sims', type=int, default=100,
                       help='Simulations MCTS par coup (défaut: 100)')
    parser.add_argument('--workers', '-w', type=int, default=4,
                       help='Nombre de workers pour le self-play parallèle (défaut: 4)')

    # Architecture
    parser.add_argument('--arch', choices=['local', 'server'], default='local',
                       help='Architecture: local (rapide, défaut) ou server (InferenceServer)')

    # Paramètres du réseau
    parser.add_argument('--network-size', choices=['small', 'medium', 'large'],
                       default='medium', help='Taille du réseau (défaut: medium)')
    parser.add_argument('--learning-rate', type=float, default=0.0003,
                       help='Taux d\'apprentissage initial (défaut: 0.0003, avec cosine decay)')

    # Configuration GPU
    parser.add_argument('--gpu', action='store_true', default=True,
                       help='Utiliser le GPU si disponible (défaut: True)')
    parser.add_argument('--no-gpu', action='store_false', dest='gpu',
                       help='Forcer l\'utilisation du CPU uniquement')

    # Replay buffer
    parser.add_argument('--buffer-size', type=int, default=50000,
                       help='Taille du replay buffer (défaut: 50000, réduit pour éviter le distribution shift)')
    parser.add_argument('--clear-buffer', action='store_true', default=False,
                       help='Vider le buffer après chaque training (force données fraîches)')

    # Checkpoints
    parser.add_argument('--checkpoint-dir', type=str, default='data/checkpoints',
                       help='Répertoire des checkpoints')
    parser.add_argument('--model-dir', type=str, default='data/models',
                       help='Répertoire des modèles')
    parser.add_argument('--resume-from', type=int, default=None,
                       help='Reprendre depuis une itération spécifique')
    parser.add_argument('--save-frequency', type=int, default=1,
                       help='Fréquence de sauvegarde du best_model (défaut: 1)')

    # Évaluation
    parser.add_argument('--eval-games', type=int, default=50,
                       help='Nombre de parties pour l\'évaluation (défaut: 50)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Affichage détaillé')

    args = parser.parse_args()

    try:
        if args.quick:
            return quick_test(args)
        elif args.evaluate:
            return evaluate(args)
        else:
            return train(args)

    except KeyboardInterrupt:
        print("\n\nEntraînement interrompu par l'utilisateur.")
        return 1
    except Exception as e:
        print(f"\nErreur: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

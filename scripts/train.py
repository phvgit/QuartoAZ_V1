#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script d'entraînement AlphaZero pour Quarto.

Usage:
    python scripts/train.py --iterations 100 --games-per-iter 50
    python scripts/train.py --quick  # Mode rapide pour test
    python scripts/train.py --evaluate  # Évaluer le modèle actuel
"""

import argparse
import sys
import os

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alphaquarto.ai.network import AlphaZeroNetwork, create_network
from alphaquarto.ai.trainer import AlphaZeroTrainer, Evaluator


def train(args):
    """Lance l'entraînement."""
    print(f"\n{'='*60}")
    print("INITIALISATION")
    print(f"{'='*60}")

    # Créer le réseau
    print(f"\nCréation du réseau ({args.network_size})...")
    network = create_network(
        size=args.network_size,
        learning_rate=args.learning_rate
    )
    print(f"  Paramètres: {network.count_parameters():,}")

    # Créer le trainer
    print("\nInitialisation du trainer...")
    trainer = AlphaZeroTrainer(
        network=network,
        mcts_sims=args.mcts_sims,
        buffer_size=args.buffer_size,
        checkpoint_dir=args.checkpoint_dir,
        model_dir=args.model_dir
    )

    # Charger un checkpoint existant si spécifié
    if args.resume_from:
        print(f"\nChargement du checkpoint {args.resume_from}...")
        trainer.load_checkpoint(args.resume_from)
        print(f"  Reprise à l'itération {trainer.iteration}")

    # Charger le replay buffer si existant
    if args.load_buffer:
        print("\nChargement du replay buffer...")
        trainer.load_replay_buffer()
        print(f"  Buffer chargé: {len(trainer.replay_buffer)} exemples")

    # Lancer l'entraînement
    print("\nDémarrage de l'entraînement...")
    trainer.train(
        num_iterations=args.iterations,
        games_per_iteration=args.games_per_iter,
        epochs_per_iteration=args.epochs,
        batch_size=args.batch_size,
        save_frequency=args.save_frequency,
        verbose=True
    )

    # Sauvegarder le replay buffer
    if args.save_buffer:
        print("\nSauvegarde du replay buffer...")
        trainer.save_replay_buffer()

    return 0


def evaluate(args):
    """Évalue le modèle actuel."""
    print(f"\n{'='*60}")
    print("ÉVALUATION DU MODÈLE")
    print(f"{'='*60}")

    # Charger le réseau
    print(f"\nCréation du réseau ({args.network_size})...")
    network = create_network(size=args.network_size)

    # Charger les poids
    model_path = os.path.join(args.model_dir, "best_model.weights.h5")
    if os.path.exists(model_path):
        print(f"Chargement du modèle: {model_path}")
        network.load_weights(model_path)
    else:
        print("Aucun modèle entraîné trouvé. Utilisation du réseau aléatoire.")

    # Évaluer
    evaluator = Evaluator(num_simulations=args.mcts_sims)

    print(f"\nÉvaluation contre joueur aléatoire ({args.eval_games} parties)...")
    results = evaluator.evaluate_vs_random(
        network,
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

    # Créer un petit réseau
    print("\nCréation d'un petit réseau...")
    network = AlphaZeroNetwork(num_filters=32, num_res_blocks=2)
    print(f"  Paramètres: {network.count_parameters():,}")

    # Créer le trainer
    trainer = AlphaZeroTrainer(
        network=network,
        mcts_sims=20,  # Peu de simulations pour aller vite
        buffer_size=10000
    )

    # Une itération rapide
    print("\nItération de test (3 parties, 2 époques)...")
    stats = trainer.train_iteration(
        num_games=3,
        epochs=2,
        batch_size=8,
        verbose=True
    )

    print(f"\n{'='*60}")
    print("TEST RÉUSSI")
    print(f"{'='*60}")
    print(f"  Parties jouées: {stats['total_games']}")
    print(f"  Exemples collectés: {stats['buffer_size']}")
    print(f"  Loss: {stats['training'].get('loss', 'N/A')}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Entraînement AlphaZero pour Quarto',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Entraînement standard
  python scripts/train.py --iterations 100 --games-per-iter 50

  # Test rapide
  python scripts/train.py --quick

  # Évaluation du modèle
  python scripts/train.py --evaluate

  # Reprendre un entraînement
  python scripts/train.py --resume-from 50 --iterations 100
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
    parser.add_argument('--games-per-iter', type=int, default=50,
                       help='Parties de self-play par itération (défaut: 50)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Époques d\'entraînement par itération (défaut: 10)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Taille des batches (défaut: 32)')
    parser.add_argument('--mcts-sims', type=int, default=100,
                       help='Simulations MCTS par coup (défaut: 100)')

    # Paramètres du réseau
    parser.add_argument('--network-size', choices=['small', 'medium', 'large'],
                       default='medium', help='Taille du réseau (défaut: medium)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Taux d\'apprentissage (défaut: 0.001)')

    # Replay buffer
    parser.add_argument('--buffer-size', type=int, default=100000,
                       help='Taille du replay buffer (défaut: 100000)')
    parser.add_argument('--load-buffer', action='store_true',
                       help='Charger le replay buffer existant')
    parser.add_argument('--save-buffer', action='store_true',
                       help='Sauvegarder le replay buffer')

    # Checkpoints
    parser.add_argument('--checkpoint-dir', type=str, default='data/checkpoints',
                       help='Répertoire des checkpoints')
    parser.add_argument('--model-dir', type=str, default='data/models',
                       help='Répertoire des modèles')
    parser.add_argument('--resume-from', type=int, default=None,
                       help='Reprendre depuis une itération spécifique')
    parser.add_argument('--save-frequency', type=int, default=10,
                       help='Fréquence de sauvegarde (défaut: 10)')

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

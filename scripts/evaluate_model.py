# -*- coding: utf-8 -*-
"""
Script d'évaluation précise du modèle contre Random.
"""

import sys
import argparse
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from alphaquarto.game.quarto import Quarto
from alphaquarto.ai.network import AlphaZeroNetwork
from alphaquarto.ai.mcts import MCTS
from alphaquarto.utils.config import NetworkConfig, MCTSConfig


def evaluate_vs_random(network, num_games: int = 100, mcts_sims: int = 100):
    """Évalue le réseau contre un joueur aléatoire."""

    mcts_config = MCTSConfig(num_simulations=mcts_sims, c_puct=1.5)
    mcts = MCTS(config=mcts_config, network=network)

    wins, losses, draws = 0, 0, 0

    for game_idx in range(num_games):
        game = Quarto()
        network_is_player = game_idx % 2  # Alterne qui commence

        # Premier choix de pièce
        if network_is_player == 0:
            piece_probs = mcts.search_piece(game, temperature=0)
            piece = int(np.argmax(piece_probs)) + 1
        else:
            piece = np.random.choice(game.get_available_pieces())
        game.choose_piece(piece)

        while not game.game_over:
            if game.current_player == network_is_player:
                move = mcts.get_best_move(game)
            else:
                move = np.random.choice(game.get_legal_moves())
            game.play_move(move)

            if not game.game_over and game.get_available_pieces():
                if game.current_player == network_is_player:
                    piece = mcts.get_best_piece(game)
                else:
                    piece = np.random.choice(game.get_available_pieces())
                game.choose_piece(piece)

        if game.winner == network_is_player:
            wins += 1
        elif game.winner is None:
            draws += 1
        else:
            losses += 1

        # Affichage progression
        if (game_idx + 1) % 10 == 0:
            wr = wins / (game_idx + 1) * 100
            print(f"  Partie {game_idx + 1}/{num_games}: WR = {wr:.1f}% ({wins}W/{losses}L/{draws}D)")

    return wins, losses, draws


def main():
    parser = argparse.ArgumentParser(description="Évaluation du modèle")
    parser.add_argument('--model', type=str, default='data/models/best_model.pt',
                       help='Chemin vers le modèle')
    parser.add_argument('--games', type=int, default=100,
                       help='Nombre de parties')
    parser.add_argument('--mcts-sims', type=int, default=100,
                       help='Simulations MCTS')
    parser.add_argument('--network-size', choices=['small', 'medium', 'large'],
                       default='medium', help='Taille du réseau')
    args = parser.parse_args()

    # Config réseau
    if args.network_size == 'small':
        network_config = NetworkConfig.small()
    elif args.network_size == 'large':
        network_config = NetworkConfig.large()
    else:
        network_config = NetworkConfig.medium()

    # Charger le modèle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = AlphaZeroNetwork(network_config)

    model_path = Path(args.model)
    if model_path.exists():
        network.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Modèle chargé: {model_path}")
    else:
        print(f"ERREUR: Modèle non trouvé: {model_path}")
        return

    network.to(device)
    network.eval()

    print(f"\n{'='*50}")
    print(f"ÉVALUATION vs RANDOM")
    print(f"{'='*50}")
    print(f"Parties: {args.games}")
    print(f"MCTS sims: {args.mcts_sims}")
    print(f"Device: {device}")
    print(f"{'='*50}\n")

    wins, losses, draws = evaluate_vs_random(
        network,
        num_games=args.games,
        mcts_sims=args.mcts_sims
    )

    total = wins + losses + draws
    print(f"\n{'='*50}")
    print(f"RÉSULTATS FINAUX")
    print(f"{'='*50}")
    print(f"Victoires réseau: {wins} ({wins/total*100:.1f}%)")
    print(f"Victoires random: {losses} ({losses/total*100:.1f}%)")
    print(f"Nuls: {draws} ({draws/total*100:.1f}%)")
    print(f"{'='*50}")

    # Intervalle de confiance approximatif (95%)
    p = wins / total
    margin = 1.96 * np.sqrt(p * (1-p) / total)
    print(f"\nWin Rate: {p*100:.1f}% ± {margin*100:.1f}%")


if __name__ == '__main__':
    main()

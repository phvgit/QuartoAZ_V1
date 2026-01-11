# -*- coding: utf-8 -*-
"""
Workers de self-play avec réseau LOCAL (pas d'IPC pour l'inférence).

Architecture optimisée:
- Chaque worker a sa propre copie du réseau sur CPU
- Pas de communication inter-processus pour l'inférence
- Sync des poids via fichier partagé
- 3-5x plus rapide que l'architecture avec InferenceServer

Usage:
    Ce module remplace selfplay_worker.py pour de meilleures performances.
"""

import time
import os
from typing import List, Optional
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import torch

from alphaquarto.game.quarto import Quarto
from alphaquarto.game.constants import NUM_PIECES, NUM_SQUARES
from alphaquarto.ai.mcts import MCTS
from alphaquarto.ai.network import AlphaZeroNetwork, StateEncoder
from alphaquarto.utils.config import MCTSConfig, NetworkConfig


# =============================================================================
# Structures de données (réutilisées)
# =============================================================================

@dataclass
class TrainingExample:
    """Un exemple d'entraînement généré pendant le self-play."""
    state: np.ndarray         # État encodé (21, 4, 4)
    policy_target: np.ndarray # Distribution MCTS sur les cases (16,)
    piece_target: np.ndarray  # Distribution MCTS sur les pièces (16,)
    value_target: float       # Résultat du jeu (-1, 0, 1)


@dataclass
class GameResult:
    """Résultat d'une partie de self-play."""
    examples: List[TrainingExample] = field(default_factory=list)
    winner: Optional[int] = None  # 0, 1, ou None (nul)
    num_moves: int = 0
    duration: float = 0.0


# =============================================================================
# Worker avec réseau local
# =============================================================================

def worker_process_local(
    worker_id: int,
    game_output_queue,        # Queue pour envoyer les parties terminées
    network_config: NetworkConfig,
    mcts_config: MCTSConfig,
    weights_path: str,        # Chemin vers les poids partagés
    num_games: int,
    verbose: bool = False
):
    """
    Worker de self-play avec réseau local (pas d'IPC pour inférence).

    Chaque worker:
    1. Crée son propre réseau sur CPU
    2. Charge les poids depuis le fichier partagé
    3. Fait l'inférence localement (très rapide)
    4. Envoie seulement les résultats de parties

    Args:
        worker_id: Identifiant unique du worker
        game_output_queue: Queue pour envoyer les parties terminées
        network_config: Configuration du réseau
        mcts_config: Configuration MCTS
        weights_path: Chemin vers le fichier de poids (.pt)
        num_games: Nombre de parties à jouer
        verbose: Afficher des informations de debug
    """
    if verbose:
        print(f"[Worker {worker_id}] Initialisation du réseau local...")

    # Créer le réseau LOCAL sur CPU
    network = AlphaZeroNetwork(network_config)
    network.to('cpu')
    network.eval()

    # Charger les poids si disponibles
    if weights_path and os.path.exists(weights_path):
        try:
            state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
            network.load_state_dict(state_dict)
            if verbose:
                print(f"[Worker {worker_id}] Poids chargés depuis {weights_path}")
        except Exception as e:
            if verbose:
                print(f"[Worker {worker_id}] Erreur chargement poids: {e}")

    # Créer le MCTS avec le réseau local (pas d'inference_client)
    mcts = MCTS(config=mcts_config, network=network)

    if verbose:
        print(f"[Worker {worker_id}] Prêt, {num_games} parties à jouer")

    games_played = 0
    total_time = 0.0

    for game_idx in range(num_games):
        try:
            start = time.time()
            game_result = play_single_game_local(mcts, network, mcts_config)
            elapsed = time.time() - start

            total_time += elapsed
            games_played += 1

            game_output_queue.put({
                'worker_id': worker_id,
                'game_idx': game_idx,
                'result': game_result
            })

            if verbose and (game_idx + 1) % 5 == 0:
                avg_time = total_time / games_played
                print(f"[Worker {worker_id}] {game_idx + 1}/{num_games} "
                      f"({avg_time:.1f}s/partie)")

        except Exception as e:
            game_output_queue.put({
                'worker_id': worker_id,
                'game_idx': game_idx,
                'error': str(e)
            })

    if verbose:
        avg_time = total_time / games_played if games_played > 0 else 0
        print(f"[Worker {worker_id}] Terminé: {games_played} parties, "
              f"{avg_time:.1f}s/partie en moyenne")


def play_single_game_local(
    mcts: MCTS,
    network: AlphaZeroNetwork,
    config: MCTSConfig
) -> GameResult:
    """
    Joue une partie complète avec réseau local.

    Args:
        mcts: Instance MCTS avec réseau local
        network: Réseau pour l'encodage d'état
        config: Configuration MCTS

    Returns:
        GameResult avec les exemples d'entraînement
    """
    start_time = time.time()
    game = Quarto()
    trajectory = []
    move_count = 0

    # Le premier joueur choisit une pièce pour l'adversaire
    if game.get_available_pieces():
        piece_probs = mcts.search_piece(game, temperature=1.0)
        piece = _sample_from_probs(piece_probs) + 1
        game.choose_piece(piece)

    # Boucle principale du jeu
    while not game.game_over:
        move_count += 1

        # Température haute au début, basse après
        temperature = 1.0 if move_count <= config.temperature_threshold else 0.1

        # Encoder l'état actuel
        state = StateEncoder.encode(game)
        current_player = game.current_player

        # MCTS pour le placement (inférence locale, très rapide)
        move_probs, _ = mcts.search(game, temperature=temperature)
        move = _sample_from_probs(move_probs)
        game.play_move(move)

        # MCTS pour le choix de pièce
        if not game.game_over and game.get_available_pieces():
            piece_probs = mcts.search_piece(game, temperature=temperature)
            piece = _sample_from_probs(piece_probs) + 1
            game.choose_piece(piece)
        else:
            piece_probs = np.zeros(NUM_PIECES)

        # Stocker dans la trajectoire
        trajectory.append({
            'state': state,
            'policy': move_probs,
            'piece_probs': piece_probs,
            'player': current_player
        })

    # Assigner les valeurs en fonction du résultat
    examples = []
    for t in trajectory:
        if game.winner is None:
            value = 0.0
        elif game.winner == t['player']:
            value = 1.0
        else:
            value = -1.0

        examples.append(TrainingExample(
            state=t['state'],
            policy_target=t['policy'],
            piece_target=t['piece_probs'],
            value_target=value
        ))

    return GameResult(
        examples=examples,
        winner=game.winner,
        num_moves=move_count,
        duration=time.time() - start_time
    )


def _sample_from_probs(probs: np.ndarray) -> int:
    """Échantillonne une action selon une distribution de probabilités."""
    total = probs.sum()
    if total == 0:
        valid = np.where(probs >= 0)[0]
        return np.random.choice(valid) if len(valid) > 0 else 0
    probs = probs / total
    return int(np.random.choice(len(probs), p=probs))

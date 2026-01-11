# -*- coding: utf-8 -*-
"""
Workers de self-play pour AlphaZero Quarto.

Ce module implémente les workers légers qui:
- Exécutent le MCTS pour générer des parties
- Communiquent avec le serveur d'inférence pour les évaluations
- Envoient les données de jeu au processus principal

Les workers sont conçus pour être légers (pas de réseau local)
et communiquer via des queues multiprocessing.
"""

import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import numpy as np

from alphaquarto.game.quarto import Quarto
from alphaquarto.game.constants import NUM_PIECES, NUM_SQUARES
from alphaquarto.ai.mcts import MCTS
from alphaquarto.ai.network import StateEncoder
from alphaquarto.ai.inference_client import InferenceClient
from alphaquarto.utils.config import MCTSConfig


# =============================================================================
# Structures de données pour les résultats
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
# Fonction principale du worker
# =============================================================================

def worker_process(
    worker_id: int,
    request_queue,      # multiprocessing.Queue pour les requêtes d'inférence
    result_queue,       # multiprocessing.Queue pour les résultats d'inférence
    game_output_queue,  # multiprocessing.Queue pour les parties complètes
    mcts_config: MCTSConfig,
    num_games: int,
    verbose: bool = False
):
    """
    Processus worker pour le self-play.

    Génère des parties en utilisant MCTS et envoie les données
    d'entraînement au processus principal.

    Args:
        worker_id: Identifiant unique du worker
        request_queue: Queue partagée pour envoyer les requêtes d'inférence
        result_queue: Queue dédiée pour recevoir les résultats
        game_output_queue: Queue pour envoyer les parties terminées
        mcts_config: Configuration MCTS
        num_games: Nombre de parties à jouer
        verbose: Afficher des informations de debug
    """
    # Créer le client d'inférence
    client = InferenceClient(
        worker_id=worker_id,
        request_queue=request_queue,
        result_queue=result_queue,
        timeout=5.0  # Timeout plus long pour les workers
    )

    # Créer le MCTS avec le client
    mcts = MCTS(config=mcts_config, inference_client=client)

    if verbose:
        print(f"[Worker {worker_id}] Démarré, {num_games} parties à jouer")

    for game_idx in range(num_games):
        try:
            game_result = play_single_game(mcts, mcts_config, verbose=False)
            game_output_queue.put({
                'worker_id': worker_id,
                'game_idx': game_idx,
                'result': game_result
            })

            if verbose and (game_idx + 1) % 10 == 0:
                print(f"[Worker {worker_id}] {game_idx + 1}/{num_games} parties")

        except Exception as e:
            # En cas d'erreur, on signale mais on continue
            game_output_queue.put({
                'worker_id': worker_id,
                'game_idx': game_idx,
                'error': str(e)
            })

    if verbose:
        print(f"[Worker {worker_id}] Terminé")


def play_single_game(
    mcts: MCTS,
    config: MCTSConfig,
    verbose: bool = False
) -> GameResult:
    """
    Joue une partie complète de self-play.

    Args:
        mcts: Instance MCTS configurée
        config: Configuration MCTS
        verbose: Afficher les détails

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

        # MCTS pour le placement
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
    """
    Échantillonne une action selon une distribution de probabilités.

    Args:
        probs: Distribution de probabilités

    Returns:
        Index de l'action sélectionnée
    """
    total = probs.sum()
    if total == 0:
        # Fallback: uniforme sur les indices non-nuls
        valid = np.where(probs >= 0)[0]
        return np.random.choice(valid) if len(valid) > 0 else 0

    # Normaliser et échantillonner
    probs = probs / total
    return int(np.random.choice(len(probs), p=probs))


# =============================================================================
# Worker alternatif sans serveur (pour tests)
# =============================================================================

def worker_process_standalone(
    worker_id: int,
    game_output_queue,
    mcts_config: MCTSConfig,
    num_games: int,
    verbose: bool = False
):
    """
    Worker standalone pour tests sans serveur d'inférence.

    Utilise le fallback MCTS (rollouts intelligents) sans réseau.
    """
    mcts = MCTS(config=mcts_config, inference_client=None)

    if verbose:
        print(f"[Worker {worker_id}] Mode standalone (pas de réseau)")

    for game_idx in range(num_games):
        try:
            game_result = play_single_game(mcts, mcts_config, verbose=False)
            game_output_queue.put({
                'worker_id': worker_id,
                'game_idx': game_idx,
                'result': game_result
            })
        except Exception as e:
            game_output_queue.put({
                'worker_id': worker_id,
                'game_idx': game_idx,
                'error': str(e)
            })

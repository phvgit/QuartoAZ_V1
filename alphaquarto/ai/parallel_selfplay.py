# -*- coding: utf-8 -*-
"""
Self-play parallélisé pour AlphaZero.

Utilise multiprocessing pour générer des parties en parallèle,
accélérant significativement la phase de self-play.
"""

import numpy as np
import multiprocessing as mp
from multiprocessing import Queue, Process
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import time
import tempfile
import os

# Configuration multiprocessing - 'spawn' requis pour compatibilité CUDA/TensorFlow
# Sur Linux, 'fork' (défaut) cause des erreurs CUDA_ERROR_NOT_INITIALIZED
# car le contexte CUDA du processus parent ne peut pas être hérité
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Déjà configuré


def _init_worker():
    """Initialise un worker (appelé une fois par processus)."""
    # Réduire la verbosité de TensorFlow dans les workers
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def _play_single_game(args: Tuple) -> Dict[str, Any]:
    """
    Joue une seule partie de self-play (fonction worker).

    Cette fonction est exécutée dans un processus séparé.
    Elle crée son propre réseau et charge les poids depuis un fichier.

    Note: Les workers utilisent le CPU pour l'inférence pendant le self-play.
    Le GPU est réservé au processus principal pour l'entraînement (plus efficace).

    Args:
        args: Tuple (weights_path, network_config, mcts_config, game_id)

    Returns:
        Dictionnaire avec les données de la partie
    """
    weights_path, network_config, mcts_config, game_id = args

    # Configuration pour les workers: utiliser CPU uniquement
    # Le GPU est réservé au processus principal pour l'entraînement
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Désactive GPU pour ce worker

    from alphaquarto.game.quarto import Quarto
    from alphaquarto.game.constants import NUM_SQUARES, NUM_PIECES
    from alphaquarto.ai.mcts import MCTS
    from alphaquarto.ai.network import AlphaZeroNetwork, StateEncoder

    # Créer le réseau local (sur CPU)
    network = AlphaZeroNetwork(
        num_filters=network_config['num_filters'],
        num_res_blocks=network_config['num_res_blocks'],
        l2_reg=network_config.get('l2_reg', 1e-4),
        learning_rate=network_config.get('learning_rate', 0.001)
    )

    # Charger les poids
    if weights_path and Path(weights_path).exists():
        network.load_weights(weights_path)

    # Configuration MCTS
    num_simulations = mcts_config.get('num_simulations', 100)
    temperature_threshold = mcts_config.get('temperature_threshold', 10)
    dirichlet_alpha = mcts_config.get('dirichlet_alpha', 0.3)
    dirichlet_epsilon = mcts_config.get('dirichlet_epsilon', 0.25)

    encoder = StateEncoder()

    # Jouer la partie
    start_time = time.time()
    game = Quarto()
    trajectory = []

    mcts = MCTS(
        num_simulations=num_simulations,
        c_puct=1.41,
        network=network,
        use_dirichlet=True,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon
    )

    move_count = 0

    # Premier joueur choisit une pièce
    if game.get_available_pieces():
        piece_probs = mcts.search_piece(game, temperature=1.0)
        piece = _sample_action(piece_probs) + 1
        game.choose_piece(piece)

    while not game.game_over:
        move_count += 1
        temperature = 1.0 if move_count <= temperature_threshold else 0.1

        # Encoder l'état
        state = encoder.encode(game)
        current_player = game.current_player

        # MCTS pour le placement
        move_probs, _ = mcts.search(game, temperature=temperature)
        move = _sample_action(move_probs)
        game.play_move(move)

        # MCTS pour le choix de pièce
        if not game.game_over and game.get_available_pieces():
            piece_probs = mcts.search_piece(game, temperature=temperature)
            piece = _sample_action(piece_probs) + 1
            game.choose_piece(piece)
        else:
            piece_probs = np.zeros(NUM_PIECES)

        trajectory.append({
            'state': state,
            'policy': move_probs,
            'piece_probs': piece_probs,
            'player': current_player
        })

    duration = time.time() - start_time

    # Assigner les valeurs
    examples = []
    for t in trajectory:
        if game.winner is None:
            value = 0.0
        elif game.winner == t['player']:
            value = 1.0
        else:
            value = -1.0

        examples.append({
            'state': t['state'],
            'policy_target': t['policy'],
            'piece_target': t['piece_probs'],
            'value_target': value
        })

    return {
        'game_id': game_id,
        'examples': examples,
        'winner': game.winner,
        'num_moves': move_count,
        'duration': duration
    }


def _sample_action(probs: np.ndarray) -> int:
    """Échantillonne une action selon la distribution."""
    if probs.sum() == 0:
        valid = np.where(probs >= 0)[0]
        return np.random.choice(valid) if len(valid) > 0 else 0
    probs = probs / probs.sum()
    return np.random.choice(len(probs), p=probs)


class ParallelSelfPlay:
    """
    Génère des parties de self-play en parallèle.

    Utilise un pool de workers pour jouer plusieurs parties simultanément.
    Chaque worker a sa propre copie du réseau pour éviter les conflits.
    """

    def __init__(
        self,
        network,
        num_workers: int = None,
        num_simulations: int = 100,
        temperature_threshold: int = 10,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25
    ):
        """
        Initialise le générateur parallèle.

        Args:
            network: Réseau AlphaZero (pour obtenir la config et les poids)
            num_workers: Nombre de workers (défaut: nb CPUs - 1)
            num_simulations: Simulations MCTS par coup
            temperature_threshold: Coups avant mode glouton
            dirichlet_alpha: Paramètre Dirichlet
            dirichlet_epsilon: Poids du bruit
        """
        self.network = network
        self.num_workers = num_workers or max(1, mp.cpu_count() - 1)
        self.num_simulations = num_simulations
        self.temperature_threshold = temperature_threshold
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

        # Fichier temporaire pour partager les poids
        self.temp_dir = tempfile.mkdtemp(prefix='alphaquarto_')
        self.weights_path = os.path.join(self.temp_dir, 'worker_weights.weights.h5')

    def _save_weights(self):
        """Sauvegarde les poids pour les workers."""
        self.network.save_weights(self.weights_path)

    def _get_network_config(self) -> Dict[str, Any]:
        """Retourne la configuration du réseau."""
        return self.network.get_config()

    def _get_mcts_config(self) -> Dict[str, Any]:
        """Retourne la configuration MCTS."""
        return {
            'num_simulations': self.num_simulations,
            'temperature_threshold': self.temperature_threshold,
            'dirichlet_alpha': self.dirichlet_alpha,
            'dirichlet_epsilon': self.dirichlet_epsilon
        }

    def generate_games(
        self,
        num_games: int,
        verbose: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Génère plusieurs parties en parallèle.

        Args:
            num_games: Nombre de parties à générer
            verbose: Afficher la progression

        Returns:
            Liste des résultats de parties
        """
        if verbose:
            print(f"  Génération parallèle: {num_games} parties sur {self.num_workers} workers")

        # Sauvegarder les poids pour les workers
        self._save_weights()

        network_config = self._get_network_config()
        mcts_config = self._get_mcts_config()

        # Préparer les arguments pour chaque partie
        args_list = [
            (self.weights_path, network_config, mcts_config, i)
            for i in range(num_games)
        ]

        start_time = time.time()
        results = []

        # Utiliser un pool de processus
        # Note: on utilise maxtasksperchild pour éviter les fuites mémoire
        with mp.Pool(
            processes=self.num_workers,
            initializer=_init_worker,
            maxtasksperchild=10
        ) as pool:
            # imap_unordered pour obtenir les résultats au fur et à mesure
            for i, result in enumerate(pool.imap_unordered(_play_single_game, args_list)):
                results.append(result)
                if verbose and (i + 1) % max(1, num_games // 5) == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    print(f"    Parties: {i + 1}/{num_games} ({rate:.2f}/s)")

        duration = time.time() - start_time

        if verbose:
            print(f"    Terminé en {duration:.1f}s ({num_games/duration:.2f} parties/s)")

        return results

    def cleanup(self):
        """Nettoie les fichiers temporaires."""
        try:
            if os.path.exists(self.weights_path):
                os.remove(self.weights_path)
            if os.path.exists(self.temp_dir):
                os.rmdir(self.temp_dir)
        except Exception:
            pass

    def __del__(self):
        """Destructeur - nettoie les fichiers temporaires."""
        self.cleanup()


def convert_results_to_records(results: List[Dict[str, Any]]) -> List:
    """
    Convertit les résultats parallèles en GameRecords.

    Args:
        results: Liste des résultats des workers

    Returns:
        Liste de GameRecord
    """
    from alphaquarto.ai.trainer import GameRecord, TrainingExample

    records = []
    for r in results:
        record = GameRecord(
            winner=r['winner'],
            num_moves=r['num_moves'],
            duration=r['duration']
        )
        for ex in r['examples']:
            example = TrainingExample(
                state=ex['state'],
                policy_target=ex['policy_target'],
                piece_target=ex['piece_target'],
                value_target=ex['value_target']
            )
            record.examples.append(example)
        records.append(record)

    return records

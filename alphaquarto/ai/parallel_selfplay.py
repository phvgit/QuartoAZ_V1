# -*- coding: utf-8 -*-
"""
Self-play parallélisé pour AlphaZero.

Utilise multiprocessing pour générer des parties en parallèle,
accélérant significativement la phase de self-play.

Architecture optimisée:
- Chaque worker charge le réseau UNE SEULE FOIS au démarrage
- Les parties suivantes réutilisent ce réseau (pas de réinitialisation)
- Workers utilisent CPU pour l'inférence, GPU réservé au processus principal
"""

# =============================================================================
# CRITICAL: Configuration TensorFlow AVANT tout import
# Ces variables DOIVENT être définies avant que TensorFlow soit importé
# NOTE: CUDA_VISIBLE_DEVICES est défini dans _init_worker, pas ici
# =============================================================================
import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Supprime INFO, WARNING, ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
from typing import List, Dict, Any, Tuple
from pathlib import Path
import time
import tempfile

# Configuration multiprocessing - 'spawn' requis pour compatibilité CUDA/TensorFlow
# Sur Linux, 'fork' (défaut) cause des erreurs CUDA_ERROR_NOT_INITIALIZED
# car le contexte CUDA du processus parent ne peut pas être hérité
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Déjà configuré


# =============================================================================
# Variables globales pour les workers (initialisées une fois par processus)
# =============================================================================

_worker_network = None
_worker_encoder = None
_worker_mcts_config = None


def _init_worker(weights_path: str, network_config: Dict[str, Any], mcts_config: Dict[str, Any]):
    """
    Initialise un worker (appelé une fois par processus au démarrage du pool).

    Charge le réseau et configure TensorFlow pour le CPU.
    Le réseau est stocké dans une variable globale et réutilisé pour toutes les parties.

    Args:
        weights_path: Chemin vers les poids du réseau
        network_config: Configuration du réseau (num_filters, num_res_blocks, etc.)
        mcts_config: Configuration MCTS (num_simulations, etc.)
    """
    global _worker_network, _worker_encoder, _worker_mcts_config

    # Désactiver GPU pour ce worker (CPU uniquement pour l'inférence)
    # NOTE: TF_CPP_MIN_LOG_LEVEL est déjà défini au niveau du module
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    from alphaquarto.ai.network import AlphaZeroNetwork, StateEncoder

    # Créer le réseau (une seule fois!)
    _worker_network = AlphaZeroNetwork(
        num_filters=network_config['num_filters'],
        num_res_blocks=network_config['num_res_blocks'],
        l2_reg=network_config.get('l2_reg', 1e-4),
        learning_rate=network_config.get('learning_rate', 0.001)
    )

    # Charger les poids si disponibles
    if weights_path and Path(weights_path).exists():
        # Utiliser skip_mismatch=True pour éviter les warnings Adam optimizer
        _worker_network.model.load_weights(weights_path, skip_mismatch=True)

    _worker_encoder = StateEncoder()
    _worker_mcts_config = mcts_config


def _play_single_game(game_id: int) -> Dict[str, Any]:
    """
    Joue une seule partie de self-play (fonction worker).

    Utilise le réseau global initialisé par _init_worker.
    Cette approche évite de recréer le réseau pour chaque partie.

    Args:
        game_id: Identifiant de la partie

    Returns:
        Dictionnaire avec les données de la partie
    """
    global _worker_network, _worker_encoder, _worker_mcts_config

    from alphaquarto.game.quarto import Quarto
    from alphaquarto.game.constants import NUM_PIECES
    from alphaquarto.ai.mcts import MCTS

    # Configuration MCTS
    num_simulations = _worker_mcts_config.get('num_simulations', 100)
    temperature_threshold = _worker_mcts_config.get('temperature_threshold', 10)
    dirichlet_alpha = _worker_mcts_config.get('dirichlet_alpha', 0.3)
    dirichlet_epsilon = _worker_mcts_config.get('dirichlet_epsilon', 0.25)

    # Jouer la partie
    start_time = time.time()
    game = Quarto()
    trajectory = []

    mcts = MCTS(
        num_simulations=num_simulations,
        c_puct=1.41,
        network=_worker_network,
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
        state = _worker_encoder.encode(game)
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

    Architecture optimisée:
    - Chaque worker charge le réseau UNE SEULE FOIS au démarrage
    - Les parties suivantes réutilisent ce réseau persistant
    - Workers utilisent CPU, GPU réservé au processus principal pour l'entraînement
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

        start_time = time.time()
        results = []

        # Utiliser un pool de processus avec initialisation personnalisée
        # Le réseau est chargé UNE SEULE FOIS par worker au démarrage du pool
        with Pool(
            processes=self.num_workers,
            initializer=_init_worker,
            initargs=(self.weights_path, network_config, mcts_config)
        ) as pool:
            # imap_unordered pour obtenir les résultats au fur et à mesure
            for i, result in enumerate(pool.imap_unordered(_play_single_game, range(num_games))):
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

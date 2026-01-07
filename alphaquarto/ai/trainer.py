# -*- coding: utf-8 -*-
"""
Entraîneur AlphaZero pour Quarto.

Implémente le cycle complet d'entraînement:
1. Self-play: Génération de parties avec MCTS guidé par le réseau
2. Replay buffer: Stockage des données d'entraînement
3. Training: Entraînement du réseau sur les données collectées
4. Evaluation: Comparaison du nouveau réseau avec l'ancien
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import json
import time
import pickle

from alphaquarto.game.quarto import Quarto
from alphaquarto.game.constants import NUM_SQUARES, NUM_PIECES
from alphaquarto.ai.mcts import MCTS
from alphaquarto.ai.network import AlphaZeroNetwork, StateEncoder


# =============================================================================
# Structures de données
# =============================================================================

@dataclass
class TrainingExample:
    """
    Exemple d'entraînement issu du self-play.

    Attributes:
        state: État encodé du jeu (4, 4, 21)
        policy_target: Distribution MCTS sur les cases (16,)
        piece_target: Distribution sur les pièces à donner (16,)
        value_target: Résultat final du jeu (-1, 0, ou 1)
    """
    state: np.ndarray
    policy_target: np.ndarray
    piece_target: np.ndarray
    value_target: float


@dataclass
class GameRecord:
    """
    Enregistrement d'une partie de self-play.

    Attributes:
        examples: Liste des exemples d'entraînement
        winner: Gagnant de la partie (0, 1, ou None pour match nul)
        num_moves: Nombre de coups joués
        duration: Durée de la partie en secondes
    """
    examples: List[TrainingExample] = field(default_factory=list)
    winner: Optional[int] = None
    num_moves: int = 0
    duration: float = 0.0


class ReplayBuffer:
    """
    Buffer circulaire pour stocker les exemples d'entraînement.

    Maintient un historique des N derniers exemples pour l'entraînement.
    Permet un échantillonnage aléatoire pour créer des mini-batches.

    Attributes:
        max_size: Taille maximale du buffer
        buffer: Deque contenant les exemples
    """

    def __init__(self, max_size: int = 100000):
        """
        Initialise le replay buffer.

        Args:
            max_size: Nombre maximum d'exemples à stocker
        """
        self.max_size = max_size
        self.buffer: deque = deque(maxlen=max_size)

    def add(self, example: TrainingExample):
        """Ajoute un exemple au buffer."""
        self.buffer.append(example)

    def add_game(self, game_record: GameRecord):
        """Ajoute tous les exemples d'une partie."""
        for example in game_record.examples:
            self.add(example)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Échantillonne un batch aléatoire.

        Args:
            batch_size: Taille du batch

        Returns:
            Tuple (states, policy_targets, piece_targets, value_targets)
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        samples = [self.buffer[i] for i in indices]

        states = np.array([s.state for s in samples])
        policy_targets = np.array([s.policy_target for s in samples])
        piece_targets = np.array([s.piece_target for s in samples])
        value_targets = np.array([s.value_target for s in samples])

        return states, policy_targets, piece_targets, value_targets

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self):
        """Vide le buffer."""
        self.buffer.clear()

    def save(self, path: str):
        """Sauvegarde le buffer sur disque."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(list(self.buffer), f)

    def load(self, path: str):
        """Charge le buffer depuis le disque."""
        if Path(path).exists():
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.buffer = deque(data, maxlen=self.max_size)


# =============================================================================
# Self-Play
# =============================================================================

class SelfPlay:
    """
    Génère des parties de self-play pour l'entraînement.

    Utilise MCTS guidé par le réseau de neurones pour jouer des parties
    contre lui-même et collecter des données d'entraînement.
    """

    def __init__(
        self,
        network: AlphaZeroNetwork,
        num_simulations: int = 100,
        c_puct: float = 1.41,
        temperature_threshold: int = 10,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25
    ):
        """
        Initialise le générateur de self-play.

        Args:
            network: Réseau de neurones pour guider MCTS
            num_simulations: Nombre de simulations MCTS par coup
            c_puct: Constante d'exploration UCB
            temperature_threshold: Nombre de coups avant de passer en mode glouton
            dirichlet_alpha: Paramètre du bruit Dirichlet
            dirichlet_epsilon: Poids du bruit Dirichlet
        """
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature_threshold = temperature_threshold
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.encoder = StateEncoder()

    def play_game(self, verbose: bool = False) -> GameRecord:
        """
        Joue une partie complète de self-play.

        Args:
            verbose: Afficher les informations de la partie

        Returns:
            GameRecord contenant les exemples d'entraînement
        """
        start_time = time.time()
        game = Quarto()
        record = GameRecord()
        trajectory = []  # [(state, policy, piece_probs, player)]

        # Créer MCTS avec le réseau
        mcts = MCTS(
            num_simulations=self.num_simulations,
            c_puct=self.c_puct,
            network=self.network,
            use_dirichlet=True,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_epsilon=self.dirichlet_epsilon
        )

        move_count = 0

        # Premier joueur choisit une pièce pour l'adversaire
        if game.get_available_pieces():
            piece_probs = mcts.search_piece(game, temperature=1.0)
            piece = self._sample_action(piece_probs) + 1
            game.choose_piece(piece)

        while not game.game_over:
            move_count += 1

            # Température: exploration au début, glouton à la fin
            temperature = 1.0 if move_count <= self.temperature_threshold else 0.1

            # Encoder l'état actuel
            state = self.encoder.encode(game)
            current_player = game.current_player

            # Recherche MCTS pour le placement
            move_probs, _ = mcts.search(game, temperature=temperature)

            # Choisir et jouer le coup
            move = self._sample_action(move_probs)
            game.play_move(move)

            # Recherche MCTS pour le choix de pièce (si la partie continue)
            if not game.game_over and game.get_available_pieces():
                piece_probs = mcts.search_piece(game, temperature=temperature)
                piece = self._sample_action(piece_probs) + 1
                game.choose_piece(piece)
            else:
                piece_probs = np.zeros(NUM_PIECES)

            # Stocker dans la trajectoire
            trajectory.append((state, move_probs, piece_probs, current_player))

            if verbose:
                print(f"  Coup {move_count}: Joueur {current_player} place en {move}")

        # Déterminer le résultat
        record.winner = game.winner
        record.num_moves = move_count
        record.duration = time.time() - start_time

        # Assigner les valeurs cibles basées sur le résultat
        for state, policy, piece_probs, player in trajectory:
            if game.winner is None:
                # Match nul
                value = 0.0
            elif game.winner == player:
                # Ce joueur a gagné
                value = 1.0
            else:
                # Ce joueur a perdu
                value = -1.0

            example = TrainingExample(
                state=state,
                policy_target=policy,
                piece_target=piece_probs,
                value_target=value
            )
            record.examples.append(example)

        if verbose:
            result = "Nul" if game.winner is None else f"Joueur {game.winner} gagne"
            print(f"  Partie terminée: {result} en {move_count} coups ({record.duration:.2f}s)")

        return record

    def _sample_action(self, probs: np.ndarray) -> int:
        """Échantillonne une action selon la distribution."""
        if probs.sum() == 0:
            # Distribution uniforme si pas de probabilités
            valid = np.where(probs >= 0)[0]
            return np.random.choice(valid) if len(valid) > 0 else 0

        # Normaliser si nécessaire
        probs = probs / probs.sum()
        return np.random.choice(len(probs), p=probs)

    def generate_games(
        self,
        num_games: int,
        verbose: bool = False,
        progress_callback: Optional[callable] = None
    ) -> List[GameRecord]:
        """
        Génère plusieurs parties de self-play.

        Args:
            num_games: Nombre de parties à générer
            verbose: Afficher les informations
            progress_callback: Fonction appelée après chaque partie (game_idx, record)

        Returns:
            Liste des GameRecords
        """
        records = []

        for i in range(num_games):
            if verbose:
                print(f"Partie {i + 1}/{num_games}")

            record = self.play_game(verbose=verbose)
            records.append(record)

            if progress_callback:
                progress_callback(i, record)

        return records


# =============================================================================
# Trainer principal
# =============================================================================

class AlphaZeroTrainer:
    """
    Entraîneur complet AlphaZero.

    Orchestre le cycle d'entraînement:
    1. Génération de parties de self-play
    2. Stockage dans le replay buffer
    3. Entraînement du réseau
    4. Évaluation et sauvegarde des checkpoints
    """

    def __init__(
        self,
        network: AlphaZeroNetwork,
        mcts_sims: int = 100,
        buffer_size: int = 100000,
        checkpoint_dir: str = "data/checkpoints",
        model_dir: str = "data/models"
    ):
        """
        Initialise le trainer.

        Args:
            network: Réseau de neurones à entraîner
            mcts_sims: Nombre de simulations MCTS
            buffer_size: Taille du replay buffer
            checkpoint_dir: Répertoire pour les checkpoints
            model_dir: Répertoire pour les modèles sauvegardés
        """
        self.network = network
        self.mcts_sims = mcts_sims
        self.replay_buffer = ReplayBuffer(max_size=buffer_size)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_dir = Path(model_dir)

        # Créer les répertoires
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Statistiques d'entraînement
        self.training_stats: List[Dict[str, Any]] = []
        self.iteration = 0
        self.total_games = 0

        # Self-play
        self.self_play = SelfPlay(
            network=network,
            num_simulations=mcts_sims
        )

    def play_game(self) -> GameRecord:
        """Joue une partie de self-play."""
        return self.self_play.play_game()

    def generate_self_play_data(
        self,
        num_games: int,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Génère des données de self-play.

        Args:
            num_games: Nombre de parties à générer
            verbose: Afficher les informations

        Returns:
            Statistiques de génération
        """
        if verbose:
            print(f"\n=== Génération de {num_games} parties de self-play ===")

        start_time = time.time()
        wins = {0: 0, 1: 0, None: 0}
        total_moves = 0
        total_examples = 0

        for i in range(num_games):
            record = self.self_play.play_game(verbose=False)
            self.replay_buffer.add_game(record)

            wins[record.winner] = wins.get(record.winner, 0) + 1
            total_moves += record.num_moves
            total_examples += len(record.examples)
            self.total_games += 1

            if verbose and (i + 1) % max(1, num_games // 10) == 0:
                print(f"  Parties: {i + 1}/{num_games}, "
                      f"Buffer: {len(self.replay_buffer)}, "
                      f"Exemples: {total_examples}")

        duration = time.time() - start_time

        stats = {
            'num_games': num_games,
            'wins_player_0': wins[0],
            'wins_player_1': wins[1],
            'draws': wins[None],
            'avg_moves': total_moves / num_games if num_games > 0 else 0,
            'total_examples': total_examples,
            'buffer_size': len(self.replay_buffer),
            'duration': duration,
            'games_per_second': num_games / duration if duration > 0 else 0
        }

        if verbose:
            print(f"  Terminé en {duration:.1f}s ({stats['games_per_second']:.2f} parties/s)")
            print(f"  Victoires: J0={wins[0]}, J1={wins[1]}, Nuls={wins[None]}")

        return stats

    def train_on_buffer(
        self,
        epochs: int = 1,
        batch_size: int = 32,
        min_buffer_size: int = 1000,
        verbose: bool = False
    ) -> Dict[str, float]:
        """
        Entraîne le réseau sur les données du replay buffer.

        Args:
            epochs: Nombre d'époques d'entraînement
            batch_size: Taille des mini-batches
            min_buffer_size: Taille minimale du buffer pour commencer
            verbose: Afficher les informations

        Returns:
            Métriques d'entraînement moyennes
        """
        if len(self.replay_buffer) < min_buffer_size:
            if verbose:
                print(f"  Buffer trop petit ({len(self.replay_buffer)}/{min_buffer_size})")
            return {'loss': 0.0}

        if verbose:
            print(f"\n=== Entraînement ({epochs} époques, batch={batch_size}) ===")

        all_metrics = []

        for epoch in range(epochs):
            # Échantillonner tout le buffer pour une époque
            num_batches = len(self.replay_buffer) // batch_size

            epoch_metrics = []
            for _ in range(num_batches):
                states, policies, pieces, values = self.replay_buffer.sample(batch_size)
                metrics = self.network.train_on_batch(states, policies, pieces, values)
                epoch_metrics.append(metrics)

            # Moyenner les métriques de l'époque
            avg_metrics = {}
            for key in epoch_metrics[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in epoch_metrics])
            all_metrics.append(avg_metrics)

            if verbose:
                print(f"  Époque {epoch + 1}/{epochs}: "
                      f"loss={avg_metrics['loss']:.4f}, "
                      f"policy_acc={avg_metrics.get('policy_accuracy', 0):.3f}, "
                      f"value_mae={avg_metrics.get('value_mae', 0):.3f}")

        # Moyenner sur toutes les époques
        final_metrics = {}
        for key in all_metrics[0].keys():
            final_metrics[key] = np.mean([m[key] for m in all_metrics])

        return final_metrics

    def train_iteration(
        self,
        num_games: int = 100,
        epochs: int = 10,
        batch_size: int = 32,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Effectue une itération complète d'entraînement.

        Une itération comprend:
        1. Génération de parties de self-play
        2. Entraînement sur le replay buffer
        3. Sauvegarde du checkpoint

        Args:
            num_games: Nombre de parties de self-play
            epochs: Nombre d'époques d'entraînement
            batch_size: Taille des batches
            verbose: Afficher les informations

        Returns:
            Statistiques de l'itération
        """
        self.iteration += 1

        if verbose:
            print(f"\n{'='*60}")
            print(f"ITERATION {self.iteration}")
            print(f"{'='*60}")

        # Phase 1: Self-play
        selfplay_stats = self.generate_self_play_data(num_games, verbose=verbose)

        # Phase 2: Entraînement
        training_metrics = self.train_on_buffer(
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )

        # Sauvegarder le checkpoint
        self.save_checkpoint()

        # Statistiques de l'itération
        iteration_stats = {
            'iteration': self.iteration,
            'selfplay': selfplay_stats,
            'training': training_metrics,
            'total_games': self.total_games,
            'buffer_size': len(self.replay_buffer)
        }
        self.training_stats.append(iteration_stats)

        if verbose:
            print(f"\n  Itération {self.iteration} terminée")
            print(f"  Total parties: {self.total_games}, Buffer: {len(self.replay_buffer)}")

        return iteration_stats

    def train(
        self,
        num_iterations: int = 100,
        games_per_iteration: int = 100,
        epochs_per_iteration: int = 10,
        batch_size: int = 32,
        save_frequency: int = 10,
        verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Lance l'entraînement complet.

        Args:
            num_iterations: Nombre d'itérations d'entraînement
            games_per_iteration: Parties de self-play par itération
            epochs_per_iteration: Époques d'entraînement par itération
            batch_size: Taille des batches
            save_frequency: Fréquence de sauvegarde du meilleur modèle
            verbose: Afficher les informations

        Returns:
            Liste des statistiques par itération
        """
        if verbose:
            print(f"\n{'#'*60}")
            print(f"# ENTRAÎNEMENT ALPHAZERO POUR QUARTO")
            print(f"# Itérations: {num_iterations}")
            print(f"# Parties/itération: {games_per_iteration}")
            print(f"# Époques/itération: {epochs_per_iteration}")
            print(f"# MCTS simulations: {self.mcts_sims}")
            print(f"{'#'*60}")

        start_time = time.time()

        for i in range(num_iterations):
            self.train_iteration(
                num_games=games_per_iteration,
                epochs=epochs_per_iteration,
                batch_size=batch_size,
                verbose=verbose
            )

            # Sauvegarder le meilleur modèle périodiquement
            if (i + 1) % save_frequency == 0:
                self.save_best_model()

        total_time = time.time() - start_time

        if verbose:
            print(f"\n{'#'*60}")
            print(f"# ENTRAÎNEMENT TERMINÉ")
            print(f"# Durée totale: {total_time/60:.1f} minutes")
            print(f"# Parties jouées: {self.total_games}")
            print(f"# Itérations: {self.iteration}")
            print(f"{'#'*60}")

        # Sauvegarder le modèle final
        self.save_best_model()
        self.save_training_stats()

        return self.training_stats

    # =========================================================================
    # Sauvegarde et chargement
    # =========================================================================

    def save_checkpoint(self):
        """Sauvegarde un checkpoint de l'entraînement."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_iter_{self.iteration}.weights.h5"
        self.network.save_weights(str(checkpoint_path))

        # Sauvegarder les métadonnées
        meta_path = self.checkpoint_dir / f"checkpoint_iter_{self.iteration}_meta.json"
        meta = {
            'iteration': self.iteration,
            'total_games': self.total_games,
            'buffer_size': len(self.replay_buffer),
            'network_config': self.network.get_config()
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

    def save_best_model(self):
        """Sauvegarde le meilleur modèle."""
        model_path = self.model_dir / "best_model.weights.h5"
        self.network.save_weights(str(model_path))

        # Métadonnées
        meta_path = self.model_dir / "best_model_meta.json"
        meta = {
            'iteration': self.iteration,
            'total_games': self.total_games,
            'network_config': self.network.get_config()
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

    def save_training_stats(self):
        """Sauvegarde les statistiques d'entraînement."""
        stats_path = self.model_dir / "training_stats.json"

        # Convertir les numpy types en Python natifs
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj

        with open(stats_path, 'w') as f:
            json.dump(convert(self.training_stats), f, indent=2)

    def save_replay_buffer(self, path: Optional[str] = None):
        """Sauvegarde le replay buffer."""
        if path is None:
            path = str(self.checkpoint_dir / "replay_buffer.pkl")
        self.replay_buffer.save(path)

    def load_replay_buffer(self, path: Optional[str] = None):
        """Charge le replay buffer."""
        if path is None:
            path = str(self.checkpoint_dir / "replay_buffer.pkl")
        self.replay_buffer.load(path)

    def load_checkpoint(self, iteration: int):
        """Charge un checkpoint spécifique."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_iter_{iteration}.weights.h5"
        if checkpoint_path.exists():
            self.network.load_weights(str(checkpoint_path))
            self.iteration = iteration

            # Charger les métadonnées
            meta_path = self.checkpoint_dir / f"checkpoint_iter_{iteration}_meta.json"
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                    self.total_games = meta.get('total_games', 0)


# =============================================================================
# Évaluation
# =============================================================================

class Evaluator:
    """
    Évalue les performances du réseau.

    Compare deux réseaux en les faisant jouer l'un contre l'autre
    ou évalue un réseau contre un joueur de référence.
    """

    def __init__(self, num_simulations: int = 50):
        """
        Initialise l'évaluateur.

        Args:
            num_simulations: Simulations MCTS pour l'évaluation
        """
        self.num_simulations = num_simulations

    def compare_networks(
        self,
        network1: AlphaZeroNetwork,
        network2: AlphaZeroNetwork,
        num_games: int = 20,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Compare deux réseaux en les faisant jouer l'un contre l'autre.

        Args:
            network1: Premier réseau
            network2: Deuxième réseau
            num_games: Nombre de parties (chaque réseau joue les deux côtés)
            verbose: Afficher les informations

        Returns:
            Statistiques de comparaison
        """
        wins = {1: 0, 2: 0, 'draw': 0}

        mcts1 = MCTS(num_simulations=self.num_simulations, network=network1, use_dirichlet=False)
        mcts2 = MCTS(num_simulations=self.num_simulations, network=network2, use_dirichlet=False)

        for game_idx in range(num_games):
            # Alterner qui commence
            if game_idx % 2 == 0:
                first_mcts, second_mcts = mcts1, mcts2
                first_id, second_id = 1, 2
            else:
                first_mcts, second_mcts = mcts2, mcts1
                first_id, second_id = 2, 1

            game = Quarto()

            # Premier joueur choisit une pièce
            if game.get_available_pieces():
                piece_probs = first_mcts.search_piece(game, temperature=0)
                piece = int(np.argmax(piece_probs)) + 1
                game.choose_piece(piece)

            current_mcts = first_mcts
            current_id = first_id

            while not game.game_over:
                # Jouer le coup
                move_probs, _ = current_mcts.search(game, temperature=0)
                move = int(np.argmax(move_probs))
                game.play_move(move)

                if game.game_over:
                    break

                # Choisir une pièce pour l'adversaire
                if game.get_available_pieces():
                    piece_probs = current_mcts.search_piece(game, temperature=0)
                    piece = int(np.argmax(piece_probs)) + 1
                    game.choose_piece(piece)

                # Alterner
                current_mcts = second_mcts if current_mcts == first_mcts else first_mcts
                current_id = second_id if current_id == first_id else first_id

            # Résultat
            if game.winner is None:
                wins['draw'] += 1
            elif game.winner == 0:
                wins[first_id] += 1
            else:
                wins[second_id] += 1

            if verbose:
                result = "Nul" if game.winner is None else f"Réseau {first_id if game.winner == 0 else second_id}"
                print(f"  Partie {game_idx + 1}: {result}")

        total = num_games
        return {
            'network1_wins': wins[1],
            'network2_wins': wins[2],
            'draws': wins['draw'],
            'network1_winrate': wins[1] / total,
            'network2_winrate': wins[2] / total,
            'draw_rate': wins['draw'] / total
        }

    def evaluate_vs_random(
        self,
        network: AlphaZeroNetwork,
        num_games: int = 50,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Évalue le réseau contre un joueur aléatoire.

        Args:
            network: Réseau à évaluer
            num_games: Nombre de parties
            verbose: Afficher les informations

        Returns:
            Statistiques d'évaluation
        """
        wins = {'network': 0, 'random': 0, 'draw': 0}

        mcts = MCTS(num_simulations=self.num_simulations, network=network, use_dirichlet=False)

        for game_idx in range(num_games):
            game = Quarto()
            network_is_first = game_idx % 2 == 0

            # Initialiser avec une pièce
            available = game.get_available_pieces()
            if available:
                if network_is_first:
                    piece_probs = mcts.search_piece(game, temperature=0)
                    piece = int(np.argmax(piece_probs)) + 1
                else:
                    piece = np.random.choice(available)
                game.choose_piece(piece)

            is_network_turn = network_is_first

            while not game.game_over:
                if is_network_turn:
                    # Réseau joue
                    move_probs, _ = mcts.search(game, temperature=0)
                    move = int(np.argmax(move_probs))
                else:
                    # Random joue
                    legal = game.get_legal_moves()
                    move = np.random.choice(legal)

                game.play_move(move)

                if game.game_over:
                    break

                # Choisir pièce
                available = game.get_available_pieces()
                if available:
                    if is_network_turn:
                        piece_probs = mcts.search_piece(game, temperature=0)
                        piece = int(np.argmax(piece_probs)) + 1
                    else:
                        piece = np.random.choice(available)
                    game.choose_piece(piece)

                is_network_turn = not is_network_turn

            # Résultat
            if game.winner is None:
                wins['draw'] += 1
            elif (game.winner == 0) == network_is_first:
                wins['network'] += 1
            else:
                wins['random'] += 1

        total = num_games
        return {
            'network_wins': wins['network'],
            'random_wins': wins['random'],
            'draws': wins['draw'],
            'network_winrate': wins['network'] / total,
            'random_winrate': wins['random'] / total
        }

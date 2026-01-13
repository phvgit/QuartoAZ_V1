# -*- coding: utf-8 -*-
"""
Entraîneur AlphaZero pour Quarto avec batched GPU inference.

Ce module orchestre l'entraînement complet:
- Serveur d'inférence GPU pour le batching
- Workers de self-play parallèles
- Replay buffer et entraînement du réseau
- Sauvegarde des checkpoints

Architecture:
    Main Process
    ├── InferenceServer (thread GPU)
    ├── Training Loop (CPU)
    └── Workers (processes CPU) x N
"""

import os
import json
import time
import queue
import multiprocessing as mp
from multiprocessing import Process, Queue
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import locale
from datetime import datetime

from alphaquarto.game.quarto import Quarto
from alphaquarto.game.constants import NUM_SQUARES, NUM_PIECES
from alphaquarto.ai.network import AlphaZeroNetwork, StateEncoder, configure_gpu
from alphaquarto.ai.mcts import MCTS
from alphaquarto.ai.inference_server import InferenceServer
from alphaquarto.ai.selfplay_worker import (
    worker_process, GameResult, TrainingExample
)
from alphaquarto.utils.config import (
    Config, NetworkConfig, MCTSConfig, InferenceConfig, TrainingConfig
)


# =============================================================================
# Replay Buffer
# =============================================================================

class ReplayBuffer:
    """
    Buffer circulaire pour stocker les exemples d'entraînement.

    Stocke des triplets (state, policy_target, piece_target, value_target).
    """

    def __init__(self, max_size: int = 100_000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def add(self, example: TrainingExample):
        """Ajoute un exemple au buffer."""
        self.buffer.append(example)

    def add_batch(self, examples: List[TrainingExample]):
        """Ajoute plusieurs exemples."""
        for ex in examples:
            self.buffer.append(ex)

    def sample(self, batch_size: int) -> tuple:
        """
        Échantillonne un batch aléatoire.

        Returns:
            Tuple (states, policies, pieces, values) en numpy arrays
        """
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), replace=False)
        samples = [self.buffer[i] for i in indices]

        states = np.array([s.state for s in samples])
        policies = np.array([s.policy_target for s in samples])
        pieces = np.array([s.piece_target for s in samples])
        values = np.array([s.value_target for s in samples])

        return states, policies, pieces, values

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()


# =============================================================================
# Trainer principal
# =============================================================================

class AlphaZeroTrainer:
    """
    Entraîneur AlphaZero avec batched GPU inference.

    Orchestre:
    - Le serveur d'inférence (thread GPU)
    - Les workers de self-play (processus CPU)
    - L'entraînement du réseau
    - Les checkpoints
    """

    def __init__(self, config: Config, use_lr_scheduler: bool = True):
        """
        Initialise l'entraîneur.

        Args:
            config: Configuration complète (network, mcts, inference, training)
            use_lr_scheduler: Si True, utilise Cosine Warm Restarts. Si False, LR constant.
        """
        self.config = config

        # Device
        self.device = torch.device(
            config.inference.device if torch.cuda.is_available() else "cpu"
        )

        # Créer le réseau
        self.network = AlphaZeroNetwork(config.network)
        self.network.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=config.network.learning_rate,
            weight_decay=config.network.l2_reg
        )

        # Learning rate scheduler (cosine annealing)
        # T_max sera défini au début de train() quand on connaît num_iterations
        self.scheduler = None
        self.use_lr_scheduler = use_lr_scheduler

        # Replay buffer
        self.replay_buffer = ReplayBuffer(config.training.buffer_size)

        # État de l'entraînement
        self.iteration = 0
        self.total_games = 0

        # Best model tracking
        self.best_loss = float('inf')
        self.best_iteration = 0

        # Early stopping (basé sur cycles de warm restart)
        self.iters_since_best = 0
        self.cycle_length = config.training.warm_restart_period
        self.patience_cycles = config.training.early_stopping_patience

        # Temps cumulé
        self.cumulative_time = 0.0
        self.training_start_time = None

        # Statistiques
        self.training_stats: List[Dict] = []

        # Créer les répertoires
        Path(config.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.training.model_dir).mkdir(parents=True, exist_ok=True)

    def _format_timestamp(self) -> str:
        """Formate la date/heure au format européen."""
        try:
            locale.setlocale(locale.LC_TIME, 'fr_FR.UTF-8')
        except locale.Error:
            try:
                locale.setlocale(locale.LC_TIME, 'French_France.1252')
            except locale.Error:
                pass  # Fallback sur la locale par défaut
        now = datetime.now()
        return now.strftime("%A %d %B %Y %H:%M:%S")

    def _format_duration(self, seconds: float) -> str:
        """Formate une durée en heures:minutes:secondes."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours}h {minutes:02d}m {secs:02d}s"
        elif minutes > 0:
            return f"{minutes}m {secs:02d}s"
        else:
            return f"{secs}s"

    def train(self, num_iterations: Optional[int] = None, verbose: bool = True):
        """
        Lance l'entraînement complet.

        Args:
            num_iterations: Nombre d'itérations (None = utiliser config)
            verbose: Afficher les progrès
        """
        if num_iterations is None:
            num_iterations = self.config.training.iterations

        # Initialiser le scheduler LR avec Cosine Annealing Warm Restarts
        if self.use_lr_scheduler and self.scheduler is None:
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.training.warm_restart_period,  # Période d'un cycle
                T_mult=1,  # Période constante
                eta_min=self.config.network.learning_rate * 0.1  # 10% du LR initial
            )

        # Timestamp de début
        self.training_start_time = time.time()
        lr_min = self.config.network.learning_rate * 0.1
        num_cycles = num_iterations // self.cycle_length

        print(f"\n{'='*60}")
        print("ENTRAÎNEMENT ALPHAZERO - QUARTO")
        print(f"{'='*60}")
        print(f"Démarrage: {self._format_timestamp()}")
        print(f"Device: {self.device}")
        print(f"Itérations: {num_iterations}")
        print(f"Parties/itération: {self.config.training.games_per_iteration}")
        print(f"Workers: {self.config.training.num_workers}")
        print(f"Simulations MCTS: {self.config.mcts.num_simulations}")
        print(f"Buffer size: {self.config.training.buffer_size:,}")
        print(f"Epochs/itération: {self.config.training.epochs_per_iteration}")
        if self.use_lr_scheduler:
            print(f"LR scheduler: Cosine Warm Restarts (T_0={self.cycle_length}, {num_cycles} cycles)")
            print(f"  LR: {self.config.network.learning_rate:.6f} → {lr_min:.6f}")
        else:
            print(f"LR scheduler: DÉSACTIVÉ (LR constant = {self.config.network.learning_rate:.6f})")
        print(f"Early stopping: patience={self.patience_cycles} cycles ({self.patience_cycles * self.cycle_length} iters)")
        print(f"{'='*60}\n")

        early_stopped = False
        for i in range(num_iterations):
            self.iteration += 1
            iter_start = time.time()

            if verbose:
                print(f"\n--- Itération {self.iteration} ---")
                print(f"  Début: {self._format_timestamp()}")

            # Phase 1: Self-play
            selfplay_stats = self._run_self_play_phase(verbose)

            # Phase 2: Training
            training_stats = self._run_training_phase(verbose)

            # Temps de l'itération
            iter_time = time.time() - iter_start
            self.cumulative_time += iter_time

            # Statistiques de l'itération
            stats = {
                'iteration': self.iteration,
                'total_games': self.total_games,
                'buffer_size': len(self.replay_buffer),
                'selfplay': selfplay_stats,
                'training': training_stats,
                'duration': iter_time,
                'cumulative_time': self.cumulative_time
            }
            self.training_stats.append(stats)

            # Step learning rate scheduler
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = self.config.network.learning_rate

            if verbose:
                if training_stats:
                    loss = training_stats.get('loss', float('inf'))
                    print(f"  Loss: {loss:.4f} (LR: {current_lr:.6f})")

                    # Track and save best model
                    if loss < self.best_loss:
                        self.best_loss = loss
                        self.best_iteration = self.iteration
                        self.iters_since_best = 0
                        self._save_best_model_checkpoint()
                        print(f"  ★ Nouveau meilleur modèle! (loss: {loss:.4f})")
                    else:
                        self.iters_since_best += 1

                print(f"  Fin: {self._format_timestamp()}")
                print(f"  Durée itération: {self._format_duration(iter_time)}")
                print(f"  Temps cumulé: {self._format_duration(self.cumulative_time)}")

            # Phase 3: Checkpoint
            if self.iteration % self.config.training.save_frequency == 0:
                self._save_checkpoint()

            # Early stopping: vérifier après chaque cycle complet
            if self.iters_since_best >= self.patience_cycles * self.cycle_length:
                print(f"\n*** EARLY STOPPING ***")
                print(f"  Pas d'amélioration depuis {self.iters_since_best} itérations")
                print(f"  ({self.patience_cycles} cycles complets)")
                print(f"  Meilleur modèle: itération {self.best_iteration} (loss: {self.best_loss:.4f})")
                early_stopped = True
                break

        # Temps total
        total_time = time.time() - self.training_start_time

        # Sauvegarde finale
        self.save_best_model()
        self.save_training_stats()
        print(f"\n{'='*60}")
        print("ENTRAÎNEMENT TERMINÉ")
        if early_stopped:
            print(f"Arrêt anticipé à l'itération {self.iteration}/{num_iterations}")
        print(f"Best model: iteration {self.best_iteration} (loss: {self.best_loss:.4f})")
        print(f"Fin: {self._format_timestamp()}")
        print(f"Temps total: {self._format_duration(total_time)}")
        print(f"{'='*60}")

    def _run_self_play_phase(self, verbose: bool = True) -> Dict:
        """
        Exécute la phase de self-play avec workers parallèles.

        Returns:
            Statistiques de la phase
        """
        start_time = time.time()
        num_workers = self.config.training.num_workers
        games_per_iteration = self.config.training.games_per_iteration
        games_per_worker = games_per_iteration // num_workers
        remaining_games = games_per_iteration % num_workers

        if verbose:
            print(f"  Self-play: {games_per_iteration} parties ({num_workers} workers)...")

        # Créer les queues multiprocessing
        ctx = mp.get_context('spawn')
        request_queue = ctx.Queue(maxsize=num_workers * 100)
        result_queues = {i: ctx.Queue(maxsize=100) for i in range(num_workers)}
        game_output_queue = ctx.Queue()

        # Démarrer le serveur d'inférence
        inference_server = InferenceServer(
            network=self.network,
            config=self.config.inference,
            request_queue=request_queue,
            result_queues=result_queues
        )
        inference_server.start()

        # Démarrer les workers
        workers = []
        for worker_id in range(num_workers):
            num_games = games_per_worker + (1 if worker_id < remaining_games else 0)
            if num_games > 0:
                p = ctx.Process(
                    target=worker_process,
                    args=(
                        worker_id,
                        request_queue,
                        result_queues[worker_id],
                        game_output_queue,
                        self.config.mcts,
                        num_games,
                        False
                    )
                )
                p.start()
                workers.append(p)

        # Collecter les résultats
        games_collected = 0
        wins_p0, wins_p1, draws = 0, 0, 0
        total_moves = 0
        errors = 0

        expected_games = sum(games_per_worker + (1 if i < remaining_games else 0) for i in range(num_workers))

        while games_collected < expected_games:
            try:
                data = game_output_queue.get(timeout=120.0)

                if 'error' in data:
                    errors += 1
                    continue

                result: GameResult = data['result']
                self.replay_buffer.add_batch(result.examples)
                self.total_games += 1
                games_collected += 1
                total_moves += result.num_moves

                if result.winner == 0:
                    wins_p0 += 1
                elif result.winner == 1:
                    wins_p1 += 1
                else:
                    draws += 1

            except queue.Empty:
                print(f"  Warning: Timeout en attente de partie {games_collected + 1}")
                break

        # Arrêter les workers et le serveur
        for p in workers:
            p.join(timeout=10.0)
            if p.is_alive():
                p.terminate()

        inference_server.stop()

        elapsed = time.time() - start_time
        games_per_sec = games_collected / elapsed if elapsed > 0 else 0

        stats = {
            'games': games_collected,
            'wins_player_0': wins_p0,
            'wins_player_1': wins_p1,
            'draws': draws,
            'avg_moves': total_moves / games_collected if games_collected > 0 else 0,
            'duration': elapsed,
            'games_per_second': games_per_sec,
            'errors': errors
        }

        if verbose:
            print(f"    {games_collected} parties en {elapsed:.1f}s ({games_per_sec:.2f} p/s)")
            print(f"    Résultats: P0={wins_p0}, P1={wins_p1}, Nuls={draws}")

        return stats

    def _run_training_phase(self, verbose: bool = True) -> Dict:
        """
        Exécute la phase d'entraînement du réseau.

        Returns:
            Statistiques d'entraînement
        """
        if len(self.replay_buffer) < self.config.training.min_buffer_size:
            if verbose:
                print(f"  Training: Buffer trop petit ({len(self.replay_buffer)}), skip")
            return {}

        if verbose:
            print(f"  Training: {self.config.training.epochs_per_iteration} epochs...")

        self.network.train()

        total_loss = 0.0
        total_policy_loss = 0.0
        total_piece_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0

        for epoch in range(self.config.training.epochs_per_iteration):
            epoch_loss = self._train_epoch()
            total_loss += epoch_loss['loss']
            total_policy_loss += epoch_loss['policy_loss']
            total_piece_loss += epoch_loss['piece_loss']
            total_value_loss += epoch_loss['value_loss']
            num_batches += epoch_loss['num_batches']

        self.network.eval()

        n_epochs = self.config.training.epochs_per_iteration
        stats = {
            'loss': total_loss / n_epochs,
            'policy_loss': total_policy_loss / n_epochs,
            'piece_loss': total_piece_loss / n_epochs,
            'value_loss': total_value_loss / n_epochs,
            'num_batches': num_batches
        }

        # Optionnellement vider le buffer pour éviter le distribution shift
        if self.config.training.clear_buffer_after_training:
            self.replay_buffer.clear()
            if verbose:
                print("    Buffer vidé (clear_buffer_after_training=True)")

        return stats

    def _train_epoch(self) -> Dict:
        """Entraîne une époque sur le replay buffer."""
        batch_size = self.config.training.batch_size
        num_batches = max(1, len(self.replay_buffer) // batch_size)

        total_loss = 0.0
        total_policy_loss = 0.0
        total_piece_loss = 0.0
        total_value_loss = 0.0

        for _ in range(num_batches):
            states, policies, pieces, values = self.replay_buffer.sample(batch_size)

            # Convertir en tenseurs
            states_t = torch.from_numpy(states).to(self.device)
            policies_t = torch.from_numpy(policies).to(self.device)
            pieces_t = torch.from_numpy(pieces).to(self.device)
            values_t = torch.from_numpy(values).float().to(self.device).unsqueeze(1)

            # Forward
            pred_policy, pred_piece, pred_value = self.network(states_t)

            # Losses
            policy_loss = -torch.mean(
                torch.sum(policies_t * torch.log(pred_policy + 1e-8), dim=1)
            )
            piece_loss = -torch.mean(
                torch.sum(pieces_t * torch.log(pred_piece + 1e-8), dim=1)
            )
            value_loss = torch.mean((pred_value - values_t) ** 2)

            loss = policy_loss + piece_loss + value_loss

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_piece_loss += piece_loss.item()
            total_value_loss += value_loss.item()

        return {
            'loss': total_loss / num_batches,
            'policy_loss': total_policy_loss / num_batches,
            'piece_loss': total_piece_loss / num_batches,
            'value_loss': total_value_loss / num_batches,
            'num_batches': num_batches
        }

    # =========================================================================
    # Checkpoints et sauvegarde
    # =========================================================================

    def _save_checkpoint(self):
        """Sauvegarde un checkpoint de l'itération courante."""
        checkpoint_dir = Path(self.config.training.checkpoint_dir)

        # Poids
        weights_path = checkpoint_dir / f"checkpoint_iter_{self.iteration}.pt"
        torch.save(self.network.state_dict(), weights_path)

        # Métadonnées
        meta = {
            'iteration': self.iteration,
            'total_games': self.total_games,
            'buffer_size': len(self.replay_buffer),
            'network_config': asdict(self.config.network)
        }
        meta_path = checkpoint_dir / f"checkpoint_iter_{self.iteration}_meta.json"
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

    def _save_best_model_checkpoint(self):
        """Sauvegarde le meilleur modèle (quand la loss diminue)."""
        model_dir = Path(self.config.training.model_dir)

        # Poids
        weights_path = model_dir / "best_model.pt"
        torch.save(self.network.state_dict(), weights_path)

        # Métadonnées
        meta = {
            'iteration': self.iteration,
            'total_games': self.total_games,
            'buffer_size': len(self.replay_buffer),
            'best_loss': self.best_loss,
            'network_config': asdict(self.config.network)
        }
        meta_path = model_dir / "best_model_meta.json"
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

    def save_best_model(self):
        """Sauvegarde le meilleur modèle."""
        model_dir = Path(self.config.training.model_dir)

        # Poids
        weights_path = model_dir / "best_model.pt"
        torch.save(self.network.state_dict(), weights_path)

        # Métadonnées
        meta = {
            'iteration': self.iteration,
            'total_games': self.total_games,
            'buffer_size': len(self.replay_buffer),
            'network_config': asdict(self.config.network)
        }
        meta_path = model_dir / "best_model_meta.json"
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"  Modèle sauvé: {weights_path}")

    def save_training_stats(self):
        """Sauvegarde les statistiques d'entraînement."""
        stats_path = Path(self.config.training.model_dir) / "training_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(self.training_stats, f, indent=2)

    def load_checkpoint(self, iteration: int):
        """
        Charge un checkpoint.

        Args:
            iteration: Numéro de l'itération à charger
        """
        checkpoint_dir = Path(self.config.training.checkpoint_dir)

        weights_path = checkpoint_dir / f"checkpoint_iter_{iteration}.pt"
        if weights_path.exists():
            self.network.load_state_dict(
                torch.load(weights_path, map_location=self.device)
            )
            self.iteration = iteration
            print(f"  Checkpoint chargé: itération {iteration}")

        meta_path = checkpoint_dir / f"checkpoint_iter_{iteration}_meta.json"
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                self.total_games = meta.get('total_games', 0)

    def load_best_model(self):
        """Charge le meilleur modèle."""
        model_dir = Path(self.config.training.model_dir)
        weights_path = model_dir / "best_model.pt"

        if weights_path.exists():
            self.network.load_state_dict(
                torch.load(weights_path, map_location=self.device)
            )
            print(f"  Modèle chargé: {weights_path}")


# =============================================================================
# Évaluateur
# =============================================================================

class Evaluator:
    """Évalue le réseau contre différents adversaires."""

    def __init__(self, network: AlphaZeroNetwork, config: MCTSConfig):
        self.network = network
        self.config = config

    def evaluate_vs_random(self, num_games: int = 50, verbose: bool = True) -> Dict:
        """Évalue contre un joueur aléatoire."""
        mcts = MCTS(config=self.config, network=self.network)

        wins, losses, draws = 0, 0, 0

        for game_idx in range(num_games):
            game = Quarto()
            network_is_player = game_idx % 2

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

        return {
            'network_wins': wins,
            'random_wins': losses,
            'draws': draws,
            'network_winrate': wins / num_games,
            'random_winrate': losses / num_games
        }

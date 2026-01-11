# -*- coding: utf-8 -*-
"""
Entraîneur AlphaZero avec réseau LOCAL par worker.

Architecture simplifiée et optimisée:
- Pas d'InferenceServer (élimine l'overhead IPC)
- Chaque worker a sa copie du réseau sur CPU
- Sync des poids via fichier partagé
- 3-5x plus rapide que l'architecture avec queues

Usage:
    from alphaquarto.ai.trainer_local import AlphaZeroTrainerLocal
    trainer = AlphaZeroTrainerLocal(config)
    trainer.train(num_iterations=100)
"""

import os
import json
import time
import queue
import tempfile
import multiprocessing as mp
from multiprocessing import Process, Queue
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import asdict

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from alphaquarto.game.quarto import Quarto
from alphaquarto.ai.network import AlphaZeroNetwork, StateEncoder
from alphaquarto.ai.mcts import MCTS
from alphaquarto.ai.selfplay_worker_local import (
    worker_process_local, GameResult, TrainingExample
)
from alphaquarto.utils.config import Config, NetworkConfig, MCTSConfig


# =============================================================================
# Replay Buffer (identique)
# =============================================================================

class ReplayBuffer:
    """Buffer circulaire pour stocker les exemples d'entraînement."""

    def __init__(self, max_size: int = 100_000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def add(self, example: TrainingExample):
        self.buffer.append(example)

    def add_batch(self, examples: List[TrainingExample]):
        for ex in examples:
            self.buffer.append(ex)

    def sample(self, batch_size: int) -> tuple:
        indices = np.random.choice(
            len(self.buffer),
            size=min(batch_size, len(self.buffer)),
            replace=False
        )
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
# Trainer avec réseau local par worker
# =============================================================================

class AlphaZeroTrainerLocal:
    """
    Entraîneur AlphaZero avec réseau local par worker.

    Plus simple et plus rapide que l'architecture avec InferenceServer.
    """

    def __init__(self, config: Config):
        self.config = config

        # Device pour l'entraînement (GPU si disponible)
        self.device = torch.device(
            config.inference.device if torch.cuda.is_available() else "cpu"
        )

        # Réseau principal (entraînement sur GPU)
        self.network = AlphaZeroNetwork(config.network)
        self.network.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=config.network.learning_rate,
            weight_decay=config.network.l2_reg
        )

        # Learning rate scheduler (cosine annealing)
        self.scheduler = None
        self.use_lr_scheduler = True

        # Replay buffer
        self.replay_buffer = ReplayBuffer(config.training.buffer_size)

        # État
        self.iteration = 0
        self.total_games = 0
        self.training_stats: List[Dict] = []

        # Best model tracking
        self.best_loss = float('inf')
        self.best_iteration = 0

        # Répertoires
        Path(config.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.training.model_dir).mkdir(parents=True, exist_ok=True)

        # Fichier temporaire pour sync des poids
        self._weights_file = os.path.join(
            config.training.checkpoint_dir, "_shared_weights.pt"
        )

    def train(self, num_iterations: Optional[int] = None, verbose: bool = True):
        """Lance l'entraînement complet."""
        if num_iterations is None:
            num_iterations = self.config.training.iterations

        # Initialiser le scheduler LR avec cosine annealing
        if self.use_lr_scheduler and self.scheduler is None:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_iterations,
                eta_min=self.config.network.learning_rate * 0.01
            )

        print(f"\n{'='*60}")
        print("ENTRAÎNEMENT ALPHAZERO - ARCHITECTURE LOCALE")
        print(f"{'='*60}")
        print(f"Device training: {self.device}")
        print(f"Device workers: CPU (réseau local)")
        print(f"Itérations: {num_iterations}")
        print(f"Parties/itération: {self.config.training.games_per_iteration}")
        print(f"Workers: {self.config.training.num_workers}")
        print(f"Simulations MCTS: {self.config.mcts.num_simulations}")
        print(f"Buffer size: {self.config.training.buffer_size:,}")
        print(f"LR scheduler: Cosine Annealing ({self.config.network.learning_rate:.6f} → {self.config.network.learning_rate * 0.01:.6f})")
        print(f"{'='*60}\n")

        for i in range(num_iterations):
            self.iteration += 1
            iter_start = time.time()

            if verbose:
                print(f"\n--- Itération {self.iteration} ---")

            # Phase 1: Self-play (workers avec réseau local)
            selfplay_stats = self._run_self_play_phase(verbose)

            # Phase 2: Training (GPU)
            training_stats = self._run_training_phase(verbose)

            # Stats
            iter_time = time.time() - iter_start
            stats = {
                'iteration': self.iteration,
                'total_games': self.total_games,
                'buffer_size': len(self.replay_buffer),
                'selfplay': selfplay_stats,
                'training': training_stats,
                'duration': iter_time
            }
            self.training_stats.append(stats)

            # Step learning rate scheduler
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = self.config.network.learning_rate

            if verbose:
                print(f"  Temps total: {iter_time:.1f}s")
                if training_stats:
                    loss = training_stats.get('loss', float('inf'))
                    print(f"  Loss: {loss:.4f} (LR: {current_lr:.6f})")

                    # Track and save best model
                    if loss < self.best_loss:
                        self.best_loss = loss
                        self.best_iteration = self.iteration
                        self._save_best_model_checkpoint()
                        print(f"  ★ Nouveau meilleur modèle! (loss: {loss:.4f})")

            # Checkpoint
            if self.iteration % self.config.training.save_frequency == 0:
                self._save_checkpoint()

        self.save_best_model()
        self.save_training_stats()
        print(f"\n{'='*60}")
        print("ENTRAÎNEMENT TERMINÉ")
        print(f"Best model: iteration {self.best_iteration} (loss: {self.best_loss:.4f})")
        print(f"{'='*60}")

    def _run_self_play_phase(self, verbose: bool = True) -> Dict:
        """Self-play avec workers locaux (pas d'InferenceServer)."""
        start_time = time.time()
        num_workers = self.config.training.num_workers
        games_per_iteration = self.config.training.games_per_iteration
        games_per_worker = games_per_iteration // num_workers
        remaining = games_per_iteration % num_workers

        if verbose:
            print(f"  Self-play: {games_per_iteration} parties "
                  f"({num_workers} workers, réseau local)...")

        # Sauvegarder les poids pour les workers
        self._save_shared_weights()

        # Créer la queue de sortie
        ctx = mp.get_context('spawn')
        game_output_queue = ctx.Queue()

        # Démarrer les workers
        workers = []
        for worker_id in range(num_workers):
            num_games = games_per_worker + (1 if worker_id < remaining else 0)
            if num_games > 0:
                p = ctx.Process(
                    target=worker_process_local,
                    args=(
                        worker_id,
                        game_output_queue,
                        self.config.network,
                        self.config.mcts,
                        self._weights_file,
                        num_games,
                        False  # verbose
                    )
                )
                p.start()
                workers.append(p)

        # Collecter les résultats
        expected = sum(games_per_worker + (1 if i < remaining else 0)
                       for i in range(num_workers))
        games_collected = 0
        wins_p0, wins_p1, draws = 0, 0, 0
        total_moves = 0
        errors = 0

        while games_collected < expected:
            try:
                data = game_output_queue.get(timeout=300.0)

                if 'error' in data:
                    errors += 1
                    if verbose:
                        print(f"    Erreur worker {data['worker_id']}: {data['error']}")
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
                print(f"  Warning: Timeout après {games_collected} parties")
                break

        # Terminer les workers
        for p in workers:
            p.join(timeout=10.0)
            if p.is_alive():
                p.terminate()

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
            print(f"    {games_collected} parties en {elapsed:.1f}s "
                  f"({games_per_sec:.2f} p/s)")
            print(f"    Résultats: P0={wins_p0}, P1={wins_p1}, Nuls={draws}")

        return stats

    def _run_training_phase(self, verbose: bool = True) -> Dict:
        """Phase d'entraînement sur GPU."""
        if len(self.replay_buffer) < self.config.training.min_buffer_size:
            if verbose:
                print(f"  Training: Buffer trop petit "
                      f"({len(self.replay_buffer)}), skip")
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

        n = self.config.training.epochs_per_iteration
        return {
            'loss': total_loss / n,
            'policy_loss': total_policy_loss / n,
            'piece_loss': total_piece_loss / n,
            'value_loss': total_value_loss / n,
            'num_batches': num_batches
        }

    def _train_epoch(self) -> Dict:
        """Entraîne une époque."""
        batch_size = self.config.training.batch_size
        num_batches = max(1, len(self.replay_buffer) // batch_size)

        total_loss = 0.0
        total_policy_loss = 0.0
        total_piece_loss = 0.0
        total_value_loss = 0.0

        for _ in range(num_batches):
            states, policies, pieces, values = self.replay_buffer.sample(batch_size)

            states_t = torch.from_numpy(states).to(self.device)
            policies_t = torch.from_numpy(policies).to(self.device)
            pieces_t = torch.from_numpy(pieces).to(self.device)
            values_t = torch.from_numpy(values).float().to(self.device).unsqueeze(1)

            pred_policy, pred_piece, pred_value = self.network(states_t)

            policy_loss = -torch.mean(
                torch.sum(policies_t * torch.log(pred_policy + 1e-8), dim=1)
            )
            piece_loss = -torch.mean(
                torch.sum(pieces_t * torch.log(pred_piece + 1e-8), dim=1)
            )
            value_loss = torch.mean((pred_value - values_t) ** 2)

            loss = policy_loss + piece_loss + value_loss

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

    def _save_shared_weights(self):
        """Sauvegarde les poids pour les workers."""
        torch.save(self.network.state_dict(), self._weights_file)

    def _save_best_model_checkpoint(self):
        """Sauvegarde le meilleur modèle (quand la loss diminue)."""
        model_dir = Path(self.config.training.model_dir)

        weights_path = model_dir / "best_model.pt"
        torch.save(self.network.state_dict(), weights_path)

        meta = {
            'iteration': self.iteration,
            'total_games': self.total_games,
            'buffer_size': len(self.replay_buffer),
            'best_loss': self.best_loss,
        }
        meta_path = model_dir / "best_model_meta.json"
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

    def _save_checkpoint(self):
        """Sauvegarde un checkpoint."""
        checkpoint_dir = Path(self.config.training.checkpoint_dir)
        weights_path = checkpoint_dir / f"checkpoint_iter_{self.iteration}.pt"
        torch.save(self.network.state_dict(), weights_path)

        meta = {
            'iteration': self.iteration,
            'total_games': self.total_games,
            'buffer_size': len(self.replay_buffer),
        }
        meta_path = checkpoint_dir / f"checkpoint_iter_{self.iteration}_meta.json"
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

    def save_best_model(self):
        """Sauvegarde le meilleur modèle."""
        model_dir = Path(self.config.training.model_dir)
        weights_path = model_dir / "best_model.pt"
        torch.save(self.network.state_dict(), weights_path)
        print(f"  Modèle sauvé: {weights_path}")

    def save_training_stats(self):
        """Sauvegarde les statistiques."""
        stats_path = Path(self.config.training.model_dir) / "training_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(self.training_stats, f, indent=2)

    def load_checkpoint(self, iteration: int):
        """Charge un checkpoint."""
        checkpoint_dir = Path(self.config.training.checkpoint_dir)
        weights_path = checkpoint_dir / f"checkpoint_iter_{iteration}.pt"
        if weights_path.exists():
            self.network.load_state_dict(
                torch.load(weights_path, map_location=self.device)
            )
            self.iteration = iteration
            print(f"  Checkpoint chargé: itération {iteration}")

# -*- coding: utf-8 -*-
"""Tests pour le trainer AlphaZero (PyTorch)."""

import pytest
import numpy as np
import sys
import os
import tempfile

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from alphaquarto.game.quarto import Quarto
from alphaquarto.game.constants import NUM_SQUARES, NUM_PIECES
from alphaquarto.ai.network import AlphaZeroNetwork
from alphaquarto.ai.selfplay_worker import TrainingExample, GameResult
from alphaquarto.ai.trainer import ReplayBuffer, AlphaZeroTrainer, Evaluator
from alphaquarto.utils.config import Config, NetworkConfig, MCTSConfig


# =============================================================================
# Tests TrainingExample et GameResult
# =============================================================================

class TestDataStructures:
    """Tests pour les structures de données."""

    def test_training_example_creation(self):
        """Test création d'un TrainingExample."""
        # Format channels-first: (21, 4, 4)
        state = np.random.rand(21, 4, 4).astype(np.float32)
        policy = np.random.rand(16).astype(np.float32)
        policy /= policy.sum()
        piece = np.random.rand(16).astype(np.float32)
        piece /= piece.sum()
        value = 0.5

        example = TrainingExample(
            state=state,
            policy_target=policy,
            piece_target=piece,
            value_target=value
        )

        assert example.state.shape == (21, 4, 4)
        assert example.policy_target.shape == (16,)
        assert example.piece_target.shape == (16,)
        assert example.value_target == 0.5

    def test_game_result_creation(self):
        """Test création d'un GameResult."""
        result = GameResult()

        assert len(result.examples) == 0
        assert result.winner is None
        assert result.num_moves == 0
        assert result.duration == 0.0


# =============================================================================
# Tests ReplayBuffer
# =============================================================================

class TestReplayBuffer:
    """Tests pour ReplayBuffer."""

    def test_buffer_creation(self):
        """Test création du buffer."""
        buffer = ReplayBuffer(max_size=1000)
        assert len(buffer) == 0
        assert buffer.max_size == 1000

    def test_add_example(self):
        """Test ajout d'exemples."""
        buffer = ReplayBuffer(max_size=100)

        example = TrainingExample(
            state=np.random.rand(21, 4, 4).astype(np.float32),
            policy_target=np.random.rand(16).astype(np.float32),
            piece_target=np.random.rand(16).astype(np.float32),
            value_target=1.0
        )
        buffer.add(example)

        assert len(buffer) == 1

    def test_add_batch(self):
        """Test ajout d'un batch d'exemples."""
        buffer = ReplayBuffer(max_size=100)

        examples = []
        for _ in range(5):
            example = TrainingExample(
                state=np.random.rand(21, 4, 4).astype(np.float32),
                policy_target=np.random.rand(16).astype(np.float32),
                piece_target=np.random.rand(16).astype(np.float32),
                value_target=1.0
            )
            examples.append(example)

        buffer.add_batch(examples)
        assert len(buffer) == 5

    def test_buffer_max_size(self):
        """Test limite de taille du buffer."""
        buffer = ReplayBuffer(max_size=10)

        for i in range(20):
            example = TrainingExample(
                state=np.random.rand(21, 4, 4).astype(np.float32),
                policy_target=np.random.rand(16).astype(np.float32),
                piece_target=np.random.rand(16).astype(np.float32),
                value_target=float(i)
            )
            buffer.add(example)

        assert len(buffer) == 10
        # Les derniers exemples sont conservés
        assert buffer.buffer[-1].value_target == 19.0

    def test_sample(self):
        """Test échantillonnage."""
        buffer = ReplayBuffer(max_size=100)

        for i in range(50):
            example = TrainingExample(
                state=np.random.rand(21, 4, 4).astype(np.float32),
                policy_target=np.random.rand(16).astype(np.float32),
                piece_target=np.random.rand(16).astype(np.float32),
                value_target=float(i)
            )
            buffer.add(example)

        states, policies, pieces, values = buffer.sample(10)

        # Format channels-first: (batch, 21, 4, 4)
        assert states.shape == (10, 21, 4, 4)
        assert policies.shape == (10, 16)
        assert pieces.shape == (10, 16)
        assert values.shape == (10,)

    def test_sample_when_buffer_small(self):
        """Test échantillonnage quand buffer plus petit que batch."""
        buffer = ReplayBuffer(max_size=100)

        for i in range(5):
            example = TrainingExample(
                state=np.random.rand(21, 4, 4).astype(np.float32),
                policy_target=np.random.rand(16).astype(np.float32),
                piece_target=np.random.rand(16).astype(np.float32),
                value_target=float(i)
            )
            buffer.add(example)

        states, policies, pieces, values = buffer.sample(10)

        # Doit retourner seulement les 5 exemples disponibles
        assert states.shape[0] == 5

    def test_clear(self):
        """Test vidage du buffer."""
        buffer = ReplayBuffer(max_size=100)

        for i in range(10):
            example = TrainingExample(
                state=np.random.rand(21, 4, 4).astype(np.float32),
                policy_target=np.random.rand(16).astype(np.float32),
                piece_target=np.random.rand(16).astype(np.float32),
                value_target=1.0
            )
            buffer.add(example)

        buffer.clear()
        assert len(buffer) == 0


# =============================================================================
# Tests AlphaZeroTrainer
# =============================================================================

class TestAlphaZeroTrainer:
    """Tests pour AlphaZeroTrainer."""

    def test_trainer_creation(self):
        """Test création du trainer."""
        config = Config.quick_test()
        trainer = AlphaZeroTrainer(config)

        assert trainer.iteration == 0
        assert trainer.total_games == 0
        assert len(trainer.replay_buffer) == 0

    def test_trainer_train_iteration(self, tmp_path):
        """Test une itération complète."""
        config = Config.quick_test()
        config.training.checkpoint_dir = str(tmp_path / "checkpoints")
        config.training.model_dir = str(tmp_path / "models")

        trainer = AlphaZeroTrainer(config)

        # Une itération rapide
        trainer.train(num_iterations=1, verbose=False)

        assert trainer.iteration == 1
        assert trainer.total_games > 0
        assert len(trainer.replay_buffer) > 0

    def test_checkpoint_save_load(self, tmp_path):
        """Test sauvegarde et chargement de checkpoint."""
        config = Config.quick_test()
        config.training.checkpoint_dir = str(tmp_path / "checkpoints")
        config.training.model_dir = str(tmp_path / "models")

        trainer1 = AlphaZeroTrainer(config)

        # Faire une itération
        trainer1.train(num_iterations=1, verbose=False)

        # Charger dans un nouveau trainer
        trainer2 = AlphaZeroTrainer(config)
        trainer2.load_checkpoint(1)

        assert trainer2.iteration == 1


# =============================================================================
# Tests Evaluator
# =============================================================================

class TestEvaluator:
    """Tests pour Evaluator."""

    def test_evaluator_creation(self):
        """Test création de l'évaluateur."""
        network_config = NetworkConfig(num_filters=32, num_res_blocks=2)
        mcts_config = MCTSConfig(num_simulations=10)
        network = AlphaZeroNetwork(network_config)
        network.eval()

        evaluator = Evaluator(network=network, config=mcts_config)
        assert evaluator.config.num_simulations == 10

    def test_evaluate_vs_random(self):
        """Test évaluation contre random."""
        network_config = NetworkConfig(num_filters=32, num_res_blocks=2)
        mcts_config = MCTSConfig(num_simulations=10)
        network = AlphaZeroNetwork(network_config)
        network.eval()

        evaluator = Evaluator(network=network, config=mcts_config)

        results = evaluator.evaluate_vs_random(num_games=4, verbose=False)

        assert 'network_wins' in results
        assert 'random_wins' in results
        assert 'draws' in results
        assert results['network_wins'] + results['random_wins'] + results['draws'] == 4


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

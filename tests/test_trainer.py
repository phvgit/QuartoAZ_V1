# -*- coding: utf-8 -*-
"""Tests pour le trainer AlphaZero."""

import pytest
import numpy as np
import sys
import os
import tempfile

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alphaquarto.game.quarto import Quarto
from alphaquarto.game.constants import NUM_SQUARES, NUM_PIECES
from alphaquarto.ai.network import AlphaZeroNetwork, TF_AVAILABLE
from alphaquarto.ai.trainer import (
    TrainingExample,
    GameRecord,
    ReplayBuffer,
    SelfPlay,
    AlphaZeroTrainer,
    Evaluator
)


# =============================================================================
# Tests TrainingExample et GameRecord
# =============================================================================

class TestDataStructures:
    """Tests pour les structures de données."""

    def test_training_example_creation(self):
        """Test création d'un TrainingExample."""
        state = np.random.rand(4, 4, 21)
        policy = np.random.rand(16)
        policy /= policy.sum()
        piece = np.random.rand(16)
        piece /= piece.sum()
        value = 0.5

        example = TrainingExample(
            state=state,
            policy_target=policy,
            piece_target=piece,
            value_target=value
        )

        assert example.state.shape == (4, 4, 21)
        assert example.policy_target.shape == (16,)
        assert example.piece_target.shape == (16,)
        assert example.value_target == 0.5

    def test_game_record_creation(self):
        """Test création d'un GameRecord."""
        record = GameRecord()

        assert len(record.examples) == 0
        assert record.winner is None
        assert record.num_moves == 0
        assert record.duration == 0.0


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
            state=np.random.rand(4, 4, 21),
            policy_target=np.random.rand(16),
            piece_target=np.random.rand(16),
            value_target=1.0
        )
        buffer.add(example)

        assert len(buffer) == 1

    def test_add_game(self):
        """Test ajout d'une partie."""
        buffer = ReplayBuffer(max_size=100)

        record = GameRecord()
        for _ in range(5):
            example = TrainingExample(
                state=np.random.rand(4, 4, 21),
                policy_target=np.random.rand(16),
                piece_target=np.random.rand(16),
                value_target=1.0
            )
            record.examples.append(example)

        buffer.add_game(record)
        assert len(buffer) == 5

    def test_buffer_max_size(self):
        """Test limite de taille du buffer."""
        buffer = ReplayBuffer(max_size=10)

        for i in range(20):
            example = TrainingExample(
                state=np.random.rand(4, 4, 21),
                policy_target=np.random.rand(16),
                piece_target=np.random.rand(16),
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
                state=np.random.rand(4, 4, 21).astype(np.float32),
                policy_target=np.random.rand(16).astype(np.float32),
                piece_target=np.random.rand(16).astype(np.float32),
                value_target=float(i)
            )
            buffer.add(example)

        states, policies, pieces, values = buffer.sample(10)

        assert states.shape == (10, 4, 4, 21)
        assert policies.shape == (10, 16)
        assert pieces.shape == (10, 16)
        assert values.shape == (10,)

    def test_sample_when_buffer_small(self):
        """Test échantillonnage quand buffer plus petit que batch."""
        buffer = ReplayBuffer(max_size=100)

        for i in range(5):
            example = TrainingExample(
                state=np.random.rand(4, 4, 21).astype(np.float32),
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
                state=np.random.rand(4, 4, 21),
                policy_target=np.random.rand(16),
                piece_target=np.random.rand(16),
                value_target=1.0
            )
            buffer.add(example)

        buffer.clear()
        assert len(buffer) == 0

    def test_save_load(self, tmp_path):
        """Test sauvegarde et chargement."""
        buffer1 = ReplayBuffer(max_size=100)

        for i in range(10):
            example = TrainingExample(
                state=np.random.rand(4, 4, 21).astype(np.float32),
                policy_target=np.random.rand(16).astype(np.float32),
                piece_target=np.random.rand(16).astype(np.float32),
                value_target=float(i)
            )
            buffer1.add(example)

        path = str(tmp_path / "buffer.pkl")
        buffer1.save(path)

        buffer2 = ReplayBuffer(max_size=100)
        buffer2.load(path)

        assert len(buffer2) == len(buffer1)


# =============================================================================
# Tests SelfPlay
# =============================================================================

@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available")
class TestSelfPlay:
    """Tests pour SelfPlay."""

    def test_selfplay_creation(self):
        """Test création du self-play."""
        network = AlphaZeroNetwork(num_filters=32, num_res_blocks=2)
        selfplay = SelfPlay(network, num_simulations=10)

        assert selfplay.num_simulations == 10

    def test_play_game(self):
        """Test jouer une partie."""
        network = AlphaZeroNetwork(num_filters=32, num_res_blocks=2)
        selfplay = SelfPlay(network, num_simulations=10)

        record = selfplay.play_game()

        assert isinstance(record, GameRecord)
        assert len(record.examples) > 0
        assert record.num_moves > 0
        assert record.duration > 0
        assert record.winner in [0, 1, None]

    def test_play_game_examples_valid(self):
        """Test que les exemples générés sont valides."""
        network = AlphaZeroNetwork(num_filters=32, num_res_blocks=2)
        selfplay = SelfPlay(network, num_simulations=10)

        record = selfplay.play_game()

        for example in record.examples:
            # État valide
            assert example.state.shape == (4, 4, 21)

            # Policy est une distribution
            assert example.policy_target.shape == (16,)
            assert np.all(example.policy_target >= 0)
            assert np.isclose(example.policy_target.sum(), 1.0, atol=1e-5)

            # Value dans [-1, 1]
            assert -1 <= example.value_target <= 1


# =============================================================================
# Tests AlphaZeroTrainer
# =============================================================================

@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available")
class TestAlphaZeroTrainer:
    """Tests pour AlphaZeroTrainer."""

    def test_trainer_creation(self):
        """Test création du trainer."""
        network = AlphaZeroNetwork(num_filters=32, num_res_blocks=2)
        trainer = AlphaZeroTrainer(network, mcts_sims=10)

        assert trainer.mcts_sims == 10
        assert trainer.iteration == 0
        assert trainer.total_games == 0
        assert len(trainer.replay_buffer) == 0

    def test_play_game(self):
        """Test jouer une partie via trainer."""
        network = AlphaZeroNetwork(num_filters=32, num_res_blocks=2)
        trainer = AlphaZeroTrainer(network, mcts_sims=10)

        record = trainer.play_game()

        assert isinstance(record, GameRecord)
        assert len(record.examples) > 0

    def test_generate_self_play_data(self):
        """Test génération de données."""
        network = AlphaZeroNetwork(num_filters=32, num_res_blocks=2)
        trainer = AlphaZeroTrainer(network, mcts_sims=10)

        stats = trainer.generate_self_play_data(num_games=2, verbose=False)

        assert stats['num_games'] == 2
        assert trainer.total_games == 2
        assert len(trainer.replay_buffer) > 0
        assert stats['total_examples'] > 0

    def test_train_on_buffer(self):
        """Test entraînement sur le buffer."""
        network = AlphaZeroNetwork(num_filters=32, num_res_blocks=2)
        trainer = AlphaZeroTrainer(network, mcts_sims=10)

        # Générer quelques données
        trainer.generate_self_play_data(num_games=3, verbose=False)

        # Entraîner avec min_buffer_size bas
        metrics = trainer.train_on_buffer(
            epochs=1,
            batch_size=8,
            min_buffer_size=1,
            verbose=False
        )

        assert 'loss' in metrics

    def test_train_iteration(self, tmp_path):
        """Test une itération complète."""
        network = AlphaZeroNetwork(num_filters=32, num_res_blocks=2)
        trainer = AlphaZeroTrainer(
            network,
            mcts_sims=10,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            model_dir=str(tmp_path / "models")
        )

        stats = trainer.train_iteration(
            num_games=2,
            epochs=1,
            batch_size=8,
            verbose=False
        )

        assert stats['iteration'] == 1
        assert stats['total_games'] == 2
        assert 'selfplay' in stats
        assert 'training' in stats

    def test_checkpoint_save_load(self, tmp_path):
        """Test sauvegarde et chargement de checkpoint."""
        network1 = AlphaZeroNetwork(num_filters=32, num_res_blocks=2)
        trainer1 = AlphaZeroTrainer(
            network1,
            mcts_sims=10,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            model_dir=str(tmp_path / "models")
        )

        # Faire une itération
        trainer1.train_iteration(num_games=2, epochs=1, batch_size=8, verbose=False)
        trainer1.save_checkpoint()

        # Charger dans un nouveau trainer
        network2 = AlphaZeroNetwork(num_filters=32, num_res_blocks=2)
        trainer2 = AlphaZeroTrainer(
            network2,
            mcts_sims=10,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            model_dir=str(tmp_path / "models")
        )
        trainer2.load_checkpoint(1)

        assert trainer2.iteration == 1


# =============================================================================
# Tests Evaluator
# =============================================================================

@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available")
class TestEvaluator:
    """Tests pour Evaluator."""

    def test_evaluator_creation(self):
        """Test création de l'évaluateur."""
        evaluator = Evaluator(num_simulations=10)
        assert evaluator.num_simulations == 10

    def test_evaluate_vs_random(self):
        """Test évaluation contre random."""
        network = AlphaZeroNetwork(num_filters=32, num_res_blocks=2)
        evaluator = Evaluator(num_simulations=10)

        results = evaluator.evaluate_vs_random(network, num_games=4, verbose=False)

        assert 'network_wins' in results
        assert 'random_wins' in results
        assert 'draws' in results
        assert results['network_wins'] + results['random_wins'] + results['draws'] == 4


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

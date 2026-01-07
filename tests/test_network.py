# -*- coding: utf-8 -*-
"""Tests pour le réseau de neurones AlphaZero."""

import pytest
import numpy as np
import sys
import os

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alphaquarto.game.quarto import Quarto
from alphaquarto.game.constants import BOARD_SIZE, NUM_SQUARES, NUM_PIECES
from alphaquarto.ai.network import (
    AlphaZeroNetwork,
    StateEncoder,
    create_network,
    TF_AVAILABLE
)


# =============================================================================
# Tests de l'encodeur d'état
# =============================================================================

class TestStateEncoder:
    """Tests pour StateEncoder."""

    def test_encode_empty_board(self):
        """Test encodage d'un plateau vide."""
        game = Quarto()
        state = StateEncoder.encode(game)

        assert state.shape == (4, 4, 21)
        # Pas de pièces sur le plateau
        assert np.sum(state[:, :, :16]) == 0
        # Pas de pièce courante
        assert np.sum(state[:, :, 16:20]) == 0
        assert np.sum(state[:, :, 20]) == 0

    def test_encode_with_current_piece(self):
        """Test encodage avec une pièce en main."""
        game = Quarto()
        game.choose_piece(1)  # Pièce 1: bits = 0000

        state = StateEncoder.encode(game)

        # Canal 20 doit être 1 (pièce en main)
        assert np.all(state[:, :, 20] == 1.0)
        # Propriétés de la pièce 1 (tous les bits à 0)
        assert np.all(state[:, :, 16] == 0)  # couleur
        assert np.all(state[:, :, 17] == 0)  # forme
        assert np.all(state[:, :, 18] == 0)  # taille
        assert np.all(state[:, :, 19] == 0)  # trou

    def test_encode_with_piece_16(self):
        """Test encodage avec la pièce 16 (tous bits à 1)."""
        game = Quarto()
        game.choose_piece(16)  # Pièce 16: bits = 1111

        state = StateEncoder.encode(game)

        # Propriétés de la pièce 16 (tous les bits à 1)
        assert np.all(state[:, :, 16] == 1)  # couleur
        assert np.all(state[:, :, 17] == 1)  # forme
        assert np.all(state[:, :, 18] == 1)  # taille
        assert np.all(state[:, :, 19] == 1)  # trou

    def test_encode_with_pieces_on_board(self):
        """Test encodage avec des pièces placées."""
        game = Quarto()
        game.choose_piece(5)
        game.play_move(0)  # Place pièce 5 en case 0

        state = StateEncoder.encode(game)

        # La pièce 5 doit être encodée en case (0,0)
        assert state[0, 0, 4] == 1.0  # piece_id - 1 = 4
        # Les autres cases du canal 4 doivent être 0
        assert np.sum(state[:, :, 4]) == 1.0

    def test_encode_batch(self):
        """Test encodage d'un batch."""
        games = [Quarto() for _ in range(5)]
        for i, game in enumerate(games):
            game.choose_piece(i + 1)

        states = StateEncoder.encode_batch(games)

        assert states.shape == (5, 4, 4, 21)


# =============================================================================
# Tests du réseau AlphaZero
# =============================================================================

@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available")
class TestAlphaZeroNetwork:
    """Tests pour AlphaZeroNetwork."""

    def test_network_creation(self):
        """Test création du réseau."""
        network = AlphaZeroNetwork(num_filters=32, num_res_blocks=2)

        assert network.model is not None
        assert network.num_filters == 32
        assert network.num_res_blocks == 2

    def test_network_output_shapes(self):
        """Test formes des sorties du réseau."""
        network = AlphaZeroNetwork(num_filters=32, num_res_blocks=2)
        game = Quarto()
        game.choose_piece(1)

        policy, piece, value = network.predict(game)

        assert policy.shape == (NUM_SQUARES,)
        assert piece.shape == (NUM_PIECES,)
        assert isinstance(value, float)

    def test_policy_is_distribution(self):
        """Test que policy est une distribution de probabilité."""
        network = AlphaZeroNetwork(num_filters=32, num_res_blocks=2)
        game = Quarto()
        game.choose_piece(1)

        policy, piece, value = network.predict(game)

        assert np.all(policy >= 0)
        assert np.isclose(np.sum(policy), 1.0, atol=1e-5)
        assert np.all(piece >= 0)
        assert np.isclose(np.sum(piece), 1.0, atol=1e-5)

    def test_value_in_range(self):
        """Test que value est dans [-1, 1]."""
        network = AlphaZeroNetwork(num_filters=32, num_res_blocks=2)
        game = Quarto()
        game.choose_piece(1)

        _, _, value = network.predict(game)

        assert -1 <= value <= 1

    def test_predict_batch(self):
        """Test prédiction batch."""
        network = AlphaZeroNetwork(num_filters=32, num_res_blocks=2)
        games = []
        for i in range(5):
            game = Quarto()
            game.choose_piece(i + 1)
            games.append(game)

        policies, pieces, values = network.predict_batch(games)

        assert policies.shape == (5, NUM_SQUARES)
        assert pieces.shape == (5, NUM_PIECES)
        assert values.shape == (5,)

    def test_callable_interface(self):
        """Test interface callable pour MCTS."""
        network = AlphaZeroNetwork(num_filters=32, num_res_blocks=2)
        game = Quarto()
        game.choose_piece(1)

        # Utiliser l'ancien format d'état (4, 4, 16)
        old_state = game.get_state()
        policy, value = network(old_state)

        assert policy.shape == (NUM_SQUARES,)
        assert isinstance(value, float)

    def test_train_on_batch(self):
        """Test entraînement sur un batch."""
        network = AlphaZeroNetwork(num_filters=32, num_res_blocks=2)

        # Créer des données d'entraînement synthétiques
        batch_size = 8
        states = np.random.rand(batch_size, 4, 4, 21).astype(np.float32)
        policy_targets = np.random.rand(batch_size, NUM_SQUARES)
        policy_targets /= policy_targets.sum(axis=1, keepdims=True)
        piece_targets = np.random.rand(batch_size, NUM_PIECES)
        piece_targets /= piece_targets.sum(axis=1, keepdims=True)
        value_targets = np.random.uniform(-1, 1, batch_size)

        metrics = network.train_on_batch(
            states, policy_targets, piece_targets, value_targets
        )

        assert 'loss' in metrics
        assert 'policy_loss' in metrics
        assert 'piece_loss' in metrics
        assert 'value_loss' in metrics
        assert metrics['loss'] > 0

    def test_save_load_weights(self, tmp_path):
        """Test sauvegarde et chargement des poids."""
        network1 = AlphaZeroNetwork(num_filters=32, num_res_blocks=2)
        game = Quarto()
        game.choose_piece(1)

        # Obtenir les prédictions initiales
        policy1, piece1, value1 = network1.predict(game)

        # Sauvegarder
        weights_path = str(tmp_path / "weights.weights.h5")
        network1.save_weights(weights_path)

        # Créer un nouveau réseau et charger
        network2 = AlphaZeroNetwork(num_filters=32, num_res_blocks=2)
        network2.load_weights(weights_path)

        # Comparer les prédictions
        policy2, piece2, value2 = network2.predict(game)

        np.testing.assert_array_almost_equal(policy1, policy2)
        np.testing.assert_array_almost_equal(piece1, piece2)
        assert abs(value1 - value2) < 1e-5

    def test_count_parameters(self):
        """Test comptage des paramètres."""
        network = AlphaZeroNetwork(num_filters=32, num_res_blocks=2)
        params = network.count_parameters()

        assert params > 0
        print(f"Nombre de paramètres: {params:,}")

    def test_get_config(self):
        """Test récupération de la configuration."""
        network = AlphaZeroNetwork(
            num_filters=64,
            num_res_blocks=6,
            l2_reg=1e-4,
            learning_rate=0.001
        )
        config = network.get_config()

        assert config['num_filters'] == 64
        assert config['num_res_blocks'] == 6
        assert config['l2_reg'] == 1e-4
        assert config['learning_rate'] == 0.001
        assert config['input_channels'] == 21


# =============================================================================
# Tests de create_network
# =============================================================================

@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available")
class TestCreateNetwork:
    """Tests pour la fonction create_network."""

    def test_create_small(self):
        """Test création réseau small."""
        network = create_network('small')
        assert network.num_filters == 32
        assert network.num_res_blocks == 4

    def test_create_medium(self):
        """Test création réseau medium."""
        network = create_network('medium')
        assert network.num_filters == 64
        assert network.num_res_blocks == 6

    def test_create_large(self):
        """Test création réseau large."""
        network = create_network('large')
        assert network.num_filters == 128
        assert network.num_res_blocks == 10


# =============================================================================
# Tests d'intégration avec MCTS
# =============================================================================

@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available")
class TestMCTSIntegration:
    """Tests d'intégration avec MCTS."""

    def test_network_with_mcts(self):
        """Test que le réseau fonctionne avec MCTS."""
        from alphaquarto.ai.mcts import MCTS

        network = AlphaZeroNetwork(num_filters=32, num_res_blocks=2)
        mcts = MCTS(num_simulations=10, network=network)

        game = Quarto()
        game.choose_piece(1)

        # Le MCTS doit pouvoir utiliser le réseau
        move_probs, piece_probs = mcts.search(game, temperature=1.0)

        assert move_probs.shape == (NUM_SQUARES,)
        assert np.isclose(np.sum(move_probs), 1.0, atol=1e-5)


# =============================================================================
# Tests de robustesse
# =============================================================================

@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available")
class TestRobustness:
    """Tests de robustesse."""

    def test_multiple_predictions(self):
        """Test plusieurs prédictions consécutives."""
        network = AlphaZeroNetwork(num_filters=32, num_res_blocks=2)

        for i in range(10):
            game = Quarto()
            game.choose_piece((i % 16) + 1)
            policy, piece, value = network.predict(game)

            assert policy.shape == (NUM_SQUARES,)
            assert np.isclose(np.sum(policy), 1.0, atol=1e-5)

    def test_game_progression(self):
        """Test prédictions à différents états du jeu."""
        network = AlphaZeroNetwork(num_filters=32, num_res_blocks=2)
        game = Quarto()

        # Premier tour
        game.choose_piece(1)
        policy1, piece1, value1 = network.predict(game)
        game.play_move(0)

        # Deuxième tour
        game.choose_piece(2)
        policy2, piece2, value2 = network.predict(game)
        game.play_move(5)

        # Les prédictions doivent être valides
        assert policy1.shape == policy2.shape
        # Les états sont différents donc les prédictions devraient différer
        # (pas garanti mais très probable avec des poids aléatoires)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

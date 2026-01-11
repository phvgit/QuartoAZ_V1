# -*- coding: utf-8 -*-
"""Tests pour le réseau de neurones AlphaZero (PyTorch)."""

import pytest
import numpy as np
import sys
import os

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from alphaquarto.game.quarto import Quarto
from alphaquarto.game.constants import BOARD_SIZE, NUM_SQUARES, NUM_PIECES
from alphaquarto.ai.network import AlphaZeroNetwork, StateEncoder
from alphaquarto.utils.config import NetworkConfig


# =============================================================================
# Tests de l'encodeur d'état
# =============================================================================

class TestStateEncoder:
    """Tests pour StateEncoder."""

    def test_encode_empty_board(self):
        """Test encodage d'un plateau vide."""
        game = Quarto()
        state = StateEncoder.encode(game)

        # Format channels-first: (21, 4, 4)
        assert state.shape == (21, 4, 4)
        # Pas de pièces sur le plateau
        assert np.sum(state[:16, :, :]) == 0
        # Pas de pièce courante
        assert np.sum(state[16:20, :, :]) == 0
        assert np.sum(state[20, :, :]) == 0

    def test_encode_with_current_piece(self):
        """Test encodage avec une pièce en main."""
        game = Quarto()
        game.choose_piece(1)  # Pièce 1: bits = 0000

        state = StateEncoder.encode(game)

        # Canal 20 doit être 1 (pièce en main)
        assert np.all(state[20, :, :] == 1.0)
        # Propriétés de la pièce 1 (tous les bits à 0)
        assert np.all(state[16, :, :] == 0)  # couleur
        assert np.all(state[17, :, :] == 0)  # forme
        assert np.all(state[18, :, :] == 0)  # taille
        assert np.all(state[19, :, :] == 0)  # trou

    def test_encode_with_piece_16(self):
        """Test encodage avec la pièce 16 (tous bits à 1)."""
        game = Quarto()
        game.choose_piece(16)  # Pièce 16: bits = 1111

        state = StateEncoder.encode(game)

        # Propriétés de la pièce 16 (tous les bits à 1)
        assert np.all(state[16, :, :] == 1)  # couleur
        assert np.all(state[17, :, :] == 1)  # forme
        assert np.all(state[18, :, :] == 1)  # taille
        assert np.all(state[19, :, :] == 1)  # trou

    def test_encode_with_pieces_on_board(self):
        """Test encodage avec des pièces placées."""
        game = Quarto()
        game.choose_piece(5)
        game.play_move(0)  # Place pièce 5 en case 0

        state = StateEncoder.encode(game)

        # La pièce 5 doit être encodée en case (0,0)
        # Canal = piece_id - 1 = 4
        assert state[4, 0, 0] == 1.0
        # Les autres positions du canal 4 doivent être 0
        assert np.sum(state[4, :, :]) == 1.0

    def test_encode_batch(self):
        """Test encodage d'un batch."""
        games = [Quarto() for _ in range(5)]
        for i, game in enumerate(games):
            game.choose_piece(i + 1)

        states = StateEncoder.encode_batch(games)

        # Format batch channels-first: (5, 21, 4, 4)
        assert states.shape == (5, 21, 4, 4)


# =============================================================================
# Tests du réseau AlphaZero
# =============================================================================

class TestAlphaZeroNetwork:
    """Tests pour AlphaZeroNetwork."""

    def test_network_creation(self):
        """Test création du réseau."""
        config = NetworkConfig(num_filters=32, num_res_blocks=2)
        network = AlphaZeroNetwork(config)

        assert network.config.num_filters == 32
        assert network.config.num_res_blocks == 2

    def test_network_forward(self):
        """Test forward pass du réseau."""
        config = NetworkConfig(num_filters=32, num_res_blocks=2)
        network = AlphaZeroNetwork(config)
        network.eval()

        # Batch de 4 états
        states = torch.randn(4, 21, 4, 4)
        policy, piece, value = network(states)

        assert policy.shape == (4, NUM_SQUARES)
        assert piece.shape == (4, NUM_PIECES)
        assert value.shape == (4, 1)

    def test_network_output_shapes(self):
        """Test formes des sorties du réseau."""
        config = NetworkConfig(num_filters=32, num_res_blocks=2)
        network = AlphaZeroNetwork(config)
        network.eval()

        game = Quarto()
        game.choose_piece(1)

        policy, piece, value = network.predict(game)

        assert policy.shape == (NUM_SQUARES,)
        assert piece.shape == (NUM_PIECES,)
        assert isinstance(value, float)

    def test_policy_is_distribution(self):
        """Test que policy est une distribution de probabilité."""
        config = NetworkConfig(num_filters=32, num_res_blocks=2)
        network = AlphaZeroNetwork(config)
        network.eval()

        game = Quarto()
        game.choose_piece(1)

        policy, piece, value = network.predict(game)

        assert np.all(policy >= 0)
        assert np.isclose(np.sum(policy), 1.0, atol=1e-5)
        assert np.all(piece >= 0)
        assert np.isclose(np.sum(piece), 1.0, atol=1e-5)

    def test_value_in_range(self):
        """Test que value est dans [-1, 1]."""
        config = NetworkConfig(num_filters=32, num_res_blocks=2)
        network = AlphaZeroNetwork(config)
        network.eval()

        game = Quarto()
        game.choose_piece(1)

        _, _, value = network.predict(game)

        assert -1 <= value <= 1

    def test_predict_batch(self):
        """Test prédiction batch."""
        config = NetworkConfig(num_filters=32, num_res_blocks=2)
        network = AlphaZeroNetwork(config)
        network.eval()

        games = []
        for i in range(5):
            game = Quarto()
            game.choose_piece(i + 1)
            games.append(game)

        policies, pieces, values = network.predict_batch(games)

        assert policies.shape == (5, NUM_SQUARES)
        assert pieces.shape == (5, NUM_PIECES)
        assert values.shape == (5,)

    def test_save_load_weights(self, tmp_path):
        """Test sauvegarde et chargement des poids."""
        config = NetworkConfig(num_filters=32, num_res_blocks=2)
        network1 = AlphaZeroNetwork(config)
        network1.eval()

        game = Quarto()
        game.choose_piece(1)

        # Obtenir les prédictions initiales
        policy1, piece1, value1 = network1.predict(game)

        # Sauvegarder
        weights_path = str(tmp_path / "weights.pt")
        torch.save(network1.state_dict(), weights_path)

        # Créer un nouveau réseau et charger
        network2 = AlphaZeroNetwork(config)
        network2.load_state_dict(torch.load(weights_path, weights_only=True))
        network2.eval()

        # Comparer les prédictions
        policy2, piece2, value2 = network2.predict(game)

        np.testing.assert_array_almost_equal(policy1, policy2)
        np.testing.assert_array_almost_equal(piece1, piece2)
        assert abs(value1 - value2) < 1e-5

    def test_count_parameters(self):
        """Test comptage des paramètres."""
        config = NetworkConfig(num_filters=32, num_res_blocks=2)
        network = AlphaZeroNetwork(config)
        params = network.count_parameters()

        assert params > 0
        print(f"Nombre de paramètres: {params:,}")

    def test_get_config(self):
        """Test récupération de la configuration."""
        config = NetworkConfig(
            num_filters=64,
            num_res_blocks=6,
            l2_reg=1e-4,
            learning_rate=0.001
        )
        network = AlphaZeroNetwork(config)
        retrieved = network.get_config()

        assert retrieved['num_filters'] == 64
        assert retrieved['num_res_blocks'] == 6
        assert retrieved['l2_reg'] == 1e-4
        assert retrieved['learning_rate'] == 0.001


# =============================================================================
# Tests d'intégration avec MCTS
# =============================================================================

class TestMCTSIntegration:
    """Tests d'intégration avec MCTS."""

    def test_network_with_mcts(self):
        """Test que le réseau fonctionne avec MCTS."""
        from alphaquarto.ai.mcts import MCTS
        from alphaquarto.utils.config import MCTSConfig

        config = NetworkConfig(num_filters=32, num_res_blocks=2)
        network = AlphaZeroNetwork(config)
        network.eval()

        mcts_config = MCTSConfig(num_simulations=10)
        mcts = MCTS(config=mcts_config, network=network)

        game = Quarto()
        game.choose_piece(1)

        # Le MCTS doit pouvoir utiliser le réseau
        move_probs, piece_probs = mcts.search(game, temperature=1.0)

        assert move_probs.shape == (NUM_SQUARES,)
        assert np.isclose(np.sum(move_probs), 1.0, atol=1e-5)


# =============================================================================
# Tests de robustesse
# =============================================================================

class TestRobustness:
    """Tests de robustesse."""

    def test_multiple_predictions(self):
        """Test plusieurs prédictions consécutives."""
        config = NetworkConfig(num_filters=32, num_res_blocks=2)
        network = AlphaZeroNetwork(config)
        network.eval()

        for i in range(10):
            game = Quarto()
            game.choose_piece((i % 16) + 1)
            policy, piece, value = network.predict(game)

            assert policy.shape == (NUM_SQUARES,)
            assert np.isclose(np.sum(policy), 1.0, atol=1e-5)

    def test_game_progression(self):
        """Test prédictions à différents états du jeu."""
        config = NetworkConfig(num_filters=32, num_res_blocks=2)
        network = AlphaZeroNetwork(config)
        network.eval()

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

    def test_gradient_computation(self):
        """Test que les gradients peuvent être calculés."""
        config = NetworkConfig(num_filters=32, num_res_blocks=2)
        network = AlphaZeroNetwork(config)
        network.train()

        # Forward pass
        states = torch.randn(4, 21, 4, 4)
        policy, piece, value = network(states)

        # Créer des targets
        policy_target = torch.softmax(torch.randn(4, NUM_SQUARES), dim=1)
        piece_target = torch.softmax(torch.randn(4, NUM_PIECES), dim=1)
        value_target = torch.randn(4, 1)

        # Calculer la loss
        policy_loss = -torch.mean(torch.sum(policy_target * torch.log(policy + 1e-8), dim=1))
        piece_loss = -torch.mean(torch.sum(piece_target * torch.log(piece + 1e-8), dim=1))
        value_loss = torch.mean((value - value_target) ** 2)
        loss = policy_loss + piece_loss + value_loss

        # Backward
        loss.backward()

        # Vérifier que les gradients existent
        for param in network.parameters():
            if param.requires_grad:
                assert param.grad is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

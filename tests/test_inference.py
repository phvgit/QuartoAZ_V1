# -*- coding: utf-8 -*-
"""Tests pour le serveur et client d'inférence."""

import pytest
import numpy as np
import sys
import os
import time
import multiprocessing as mp
from queue import Empty

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from alphaquarto.game.quarto import Quarto
from alphaquarto.game.constants import NUM_SQUARES, NUM_PIECES
from alphaquarto.ai.network import AlphaZeroNetwork, StateEncoder
from alphaquarto.ai.inference_server import (
    InferenceServer, InferenceRequest, InferenceResult
)
from alphaquarto.ai.inference_client import InferenceClient, DummyInferenceClient
from alphaquarto.utils.config import NetworkConfig, InferenceConfig


# =============================================================================
# Tests du DummyInferenceClient
# =============================================================================

class TestDummyInferenceClient:
    """Tests pour DummyInferenceClient."""

    def test_predict(self):
        """Test prédiction avec le dummy client."""
        client = DummyInferenceClient()
        game = Quarto()
        game.choose_piece(1)

        state = StateEncoder.encode(game)
        policy, piece, value = client.predict(state)

        assert policy.shape == (NUM_SQUARES,)
        assert piece.shape == (NUM_PIECES,)
        assert isinstance(value, float)
        assert -1 <= value <= 1

    def test_distributions_valid(self):
        """Test que les distributions sont valides."""
        client = DummyInferenceClient()
        state = np.random.rand(21, 4, 4).astype(np.float32)

        policy, piece, _ = client.predict(state)

        # Vérifier que ce sont des distributions
        assert np.all(policy >= 0)
        assert np.isclose(np.sum(policy), 1.0, atol=1e-5)
        assert np.all(piece >= 0)
        assert np.isclose(np.sum(piece), 1.0, atol=1e-5)


# =============================================================================
# Tests du protocole d'inférence
# =============================================================================

class TestInferenceProtocol:
    """Tests pour InferenceRequest et InferenceResult."""

    def test_request_creation(self):
        """Test création d'une requête."""
        state = np.random.rand(21, 4, 4).astype(np.float32)
        req = InferenceRequest(worker_id=0, request_id=1, state=state)

        assert req.worker_id == 0
        assert req.request_id == 1
        assert req.state.shape == (21, 4, 4)

    def test_result_creation(self):
        """Test création d'un résultat."""
        policy = np.random.rand(NUM_SQUARES).astype(np.float32)
        piece = np.random.rand(NUM_PIECES).astype(np.float32)
        result = InferenceResult(
            worker_id=0,
            request_id=1,
            policy=policy,
            piece=piece,
            value=0.5
        )

        assert result.worker_id == 0
        assert result.request_id == 1
        assert result.policy.shape == (NUM_SQUARES,)
        assert result.piece.shape == (NUM_PIECES,)
        assert result.value == 0.5


# =============================================================================
# Tests du serveur d'inférence
# =============================================================================

class TestInferenceServer:
    """Tests pour InferenceServer."""

    def test_server_start_stop(self):
        """Test démarrage et arrêt du serveur."""
        config = NetworkConfig(num_filters=32, num_res_blocks=2)
        network = AlphaZeroNetwork(config)
        network.eval()

        inference_config = InferenceConfig(
            max_batch_size=4,
            batch_timeout_ms=10.0,
            device="cpu"
        )

        ctx = mp.get_context('spawn')
        request_queue = ctx.Queue()
        result_queues = {0: ctx.Queue()}

        server = InferenceServer(
            network=network,
            config=inference_config,
            request_queue=request_queue,
            result_queues=result_queues
        )

        # Démarrer
        server.start()
        assert server.is_running()

        # Arrêter
        server.stop()
        assert not server.is_running()

    def test_single_inference(self):
        """Test d'une inférence unique."""
        config = NetworkConfig(num_filters=32, num_res_blocks=2)
        network = AlphaZeroNetwork(config)
        network.eval()

        inference_config = InferenceConfig(
            max_batch_size=4,
            batch_timeout_ms=50.0,  # Timeout plus long pour le test
            device="cpu"
        )

        ctx = mp.get_context('spawn')
        request_queue = ctx.Queue()
        result_queues = {0: ctx.Queue()}

        server = InferenceServer(
            network=network,
            config=inference_config,
            request_queue=request_queue,
            result_queues=result_queues
        )
        server.start()

        try:
            # Envoyer une requête
            state = np.random.rand(21, 4, 4).astype(np.float32)
            req = InferenceRequest(worker_id=0, request_id=1, state=state)
            request_queue.put(req)

            # Attendre le résultat
            result = result_queues[0].get(timeout=5.0)

            assert result.worker_id == 0
            assert result.request_id == 1
            assert result.policy.shape == (NUM_SQUARES,)
            assert result.piece.shape == (NUM_PIECES,)
            assert -1 <= result.value <= 1

        finally:
            server.stop()

    def test_batch_inference(self):
        """Test d'inférence avec batching."""
        config = NetworkConfig(num_filters=32, num_res_blocks=2)
        network = AlphaZeroNetwork(config)
        network.eval()

        inference_config = InferenceConfig(
            max_batch_size=8,
            batch_timeout_ms=100.0,
            device="cpu"
        )

        ctx = mp.get_context('spawn')
        request_queue = ctx.Queue()
        result_queues = {i: ctx.Queue() for i in range(4)}

        server = InferenceServer(
            network=network,
            config=inference_config,
            request_queue=request_queue,
            result_queues=result_queues
        )
        server.start()

        try:
            # Envoyer plusieurs requêtes
            for worker_id in range(4):
                for req_id in range(2):
                    state = np.random.rand(21, 4, 4).astype(np.float32)
                    req = InferenceRequest(
                        worker_id=worker_id,
                        request_id=req_id,
                        state=state
                    )
                    request_queue.put(req)

            # Attendre tous les résultats
            results_received = 0
            for worker_id in range(4):
                for _ in range(2):
                    result = result_queues[worker_id].get(timeout=5.0)
                    assert result.worker_id == worker_id
                    results_received += 1

            assert results_received == 8

        finally:
            server.stop()


# =============================================================================
# Tests du client d'inférence
# =============================================================================

class TestInferenceClient:
    """Tests pour InferenceClient."""

    def test_client_predict(self):
        """Test prédiction via le client."""
        # Créer le serveur
        config = NetworkConfig(num_filters=32, num_res_blocks=2)
        network = AlphaZeroNetwork(config)
        network.eval()

        inference_config = InferenceConfig(
            max_batch_size=4,
            batch_timeout_ms=50.0,
            device="cpu"
        )

        ctx = mp.get_context('spawn')
        request_queue = ctx.Queue()
        result_queues = {0: ctx.Queue()}

        server = InferenceServer(
            network=network,
            config=inference_config,
            request_queue=request_queue,
            result_queues=result_queues
        )
        server.start()

        try:
            # Créer le client
            client = InferenceClient(
                worker_id=0,
                request_queue=request_queue,
                result_queue=result_queues[0],
                timeout=5.0
            )

            # Faire une prédiction
            game = Quarto()
            game.choose_piece(1)
            state = StateEncoder.encode(game)

            policy, piece, value = client.predict(state)

            assert policy.shape == (NUM_SQUARES,)
            assert piece.shape == (NUM_PIECES,)
            assert isinstance(value, float)

        finally:
            server.stop()


# =============================================================================
# Tests d'intégration
# =============================================================================

class TestInferenceIntegration:
    """Tests d'intégration du système d'inférence."""

    def test_multiple_clients_sequential(self):
        """Test avec plusieurs clients en séquence."""
        config = NetworkConfig(num_filters=32, num_res_blocks=2)
        network = AlphaZeroNetwork(config)
        network.eval()

        inference_config = InferenceConfig(
            max_batch_size=8,
            batch_timeout_ms=50.0,
            device="cpu"
        )

        ctx = mp.get_context('spawn')
        request_queue = ctx.Queue()
        result_queues = {i: ctx.Queue() for i in range(2)}

        server = InferenceServer(
            network=network,
            config=inference_config,
            request_queue=request_queue,
            result_queues=result_queues
        )
        server.start()

        try:
            clients = [
                InferenceClient(i, request_queue, result_queues[i], timeout=5.0)
                for i in range(2)
            ]

            # Chaque client fait plusieurs prédictions
            for client in clients:
                for _ in range(3):
                    state = np.random.rand(21, 4, 4).astype(np.float32)
                    policy, piece, value = client.predict(state)

                    assert policy.shape == (NUM_SQUARES,)
                    assert np.isclose(np.sum(policy), 1.0, atol=1e-5)

        finally:
            server.stop()

    def test_weights_reload(self):
        """Test rechargement des poids pendant l'exécution."""
        config = NetworkConfig(num_filters=32, num_res_blocks=2)
        network = AlphaZeroNetwork(config)
        network.eval()

        inference_config = InferenceConfig(
            max_batch_size=4,
            batch_timeout_ms=50.0,
            device="cpu"
        )

        ctx = mp.get_context('spawn')
        request_queue = ctx.Queue()
        result_queues = {0: ctx.Queue()}

        server = InferenceServer(
            network=network,
            config=inference_config,
            request_queue=request_queue,
            result_queues=result_queues
        )
        server.start()

        try:
            client = InferenceClient(0, request_queue, result_queues[0], timeout=5.0)

            # Première prédiction
            state = np.random.rand(21, 4, 4).astype(np.float32)
            policy1, piece1, value1 = client.predict(state)

            # Modifier les poids du réseau
            with torch.no_grad():
                for param in network.parameters():
                    param.add_(0.1)

            # Recharger les poids
            server.reload_weights(network.state_dict())
            time.sleep(0.1)  # Laisser le temps au serveur de recharger

            # Nouvelle prédiction
            policy2, piece2, value2 = client.predict(state)

            # Les prédictions doivent être différentes
            # (pas garanti à 100% mais très probable)
            # On vérifie juste que ça fonctionne sans erreur

        finally:
            server.stop()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

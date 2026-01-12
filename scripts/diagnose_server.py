#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de diagnostic pour l'architecture SERVER.

Teste chaque composant séparément pour identifier le problème.
"""

import sys
import os
import time
import queue
import threading
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from alphaquarto.game.quarto import Quarto
from alphaquarto.ai.network import AlphaZeroNetwork, StateEncoder
from alphaquarto.ai.mcts import MCTS
from alphaquarto.utils.config import NetworkConfig, MCTSConfig


def test_gpu():
    """Test 1: GPU disponible et fonctionnel."""
    print("\n" + "="*60)
    print("TEST 1: GPU")
    print("="*60)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

        # Test simple
        x = torch.randn(100, 100, device=device)
        y = torch.matmul(x, x)
        print(f"  Test matmul: OK ({y.shape})")
        return True
    else:
        print("  GPU: NON DISPONIBLE")
        return False


def test_network_gpu():
    """Test 2: Réseau sur GPU."""
    print("\n" + "="*60)
    print("TEST 2: Réseau sur GPU")
    print("="*60)

    config = NetworkConfig(num_filters=64, num_res_blocks=6)
    network = AlphaZeroNetwork(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network.to(device)
    network.eval()
    print(f"  Réseau sur: {device}")

    # Charger les poids
    weights_path = "data/models/best_model.pt"
    if os.path.exists(weights_path):
        network.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"  Poids chargés: {weights_path}")

    # Test inférence
    state = np.zeros((1, 21, 4, 4), dtype=np.float32)
    state_tensor = torch.from_numpy(state).to(device)

    start = time.time()
    with torch.no_grad():
        policy, piece, value = network(state_tensor)
    elapsed = time.time() - start

    print(f"  Inférence: {elapsed*1000:.1f}ms")
    print(f"  Policy shape: {policy.shape}, Value: {value.item():.4f}")

    return network, device


def test_mcts_gpu(network, device):
    """Test 3: MCTS avec réseau sur GPU."""
    print("\n" + "="*60)
    print("TEST 3: MCTS avec réseau GPU (10 sims)")
    print("="*60)

    config = MCTSConfig(num_simulations=10, c_puct=1.5)
    mcts = MCTS(config=config, network=network)

    game = Quarto()
    game.choose_piece(1)

    start = time.time()
    probs, _ = mcts.search(game, temperature=1.0)
    elapsed = time.time() - start

    print(f"  MCTS 10 sims: {elapsed:.2f}s")
    print(f"  Probabilités: {probs[:4]}...")

    return elapsed < 5.0  # Devrait être rapide


def test_mcts_gpu_100(network, device):
    """Test 4: MCTS avec 100 simulations sur GPU."""
    print("\n" + "="*60)
    print("TEST 4: MCTS avec réseau GPU (100 sims)")
    print("="*60)

    config = MCTSConfig(num_simulations=100, c_puct=1.5)
    mcts = MCTS(config=config, network=network)

    game = Quarto()
    game.choose_piece(1)

    start = time.time()
    probs, _ = mcts.search(game, temperature=1.0)
    elapsed = time.time() - start

    print(f"  MCTS 100 sims: {elapsed:.2f}s")

    if elapsed > 60:
        print(f"  ATTENTION: Très lent! Vérifier que le réseau est sur GPU")
        return False
    elif elapsed > 10:
        print(f"  OK mais lent (normal pour 100 sims)")
        return True
    else:
        print(f"  Excellent!")
        return True


def test_inference_server_thread():
    """Test 5: InferenceServer dans un thread."""
    print("\n" + "="*60)
    print("TEST 5: InferenceServer (thread)")
    print("="*60)

    from alphaquarto.ai.inference_server import InferenceServer, InferenceRequest
    from alphaquarto.utils.config import InferenceConfig

    # Configuration
    net_config = NetworkConfig(num_filters=64, num_res_blocks=6)
    inf_config = InferenceConfig(
        max_batch_size=64,
        batch_timeout_ms=10.0,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Créer le réseau
    network = AlphaZeroNetwork(net_config)
    device = torch.device(inf_config.device)
    network.to(device)
    network.eval()

    # Charger les poids
    weights_path = "data/models/best_model.pt"
    if os.path.exists(weights_path):
        network.load_state_dict(torch.load(weights_path, map_location=device))

    # Créer les queues STANDARD (pas multiprocessing)
    request_queue = queue.Queue(maxsize=100)
    result_queues = {0: queue.Queue(maxsize=100)}

    # Créer et démarrer le serveur
    server = InferenceServer(
        network=network,
        config=inf_config,
        request_queue=request_queue,
        result_queues=result_queues
    )
    server.start()
    print(f"  Serveur démarré: {server.is_running()}")

    # Envoyer une requête de test
    state = np.zeros((21, 4, 4), dtype=np.float32)
    request = InferenceRequest(worker_id=0, request_id=0, state=state)

    start = time.time()
    request_queue.put(request)

    try:
        result = result_queues[0].get(timeout=5.0)
        elapsed = time.time() - start
        print(f"  Réponse reçue: {elapsed*1000:.1f}ms")
        print(f"  Policy shape: {result.policy.shape}")
        success = True
    except queue.Empty:
        print(f"  ERREUR: Timeout en attente de réponse!")
        success = False

    server.stop()
    return success


def test_mp_queues():
    """Test 6: Communication via multiprocessing.Queue."""
    print("\n" + "="*60)
    print("TEST 6: Multiprocessing Queues")
    print("="*60)

    ctx = mp.get_context('spawn')
    q = ctx.Queue()

    # Test simple: envoyer/recevoir un array numpy
    arr = np.random.randn(21, 4, 4).astype(np.float32)
    q.put(arr)

    try:
        received = q.get(timeout=5.0)
        print(f"  Numpy array: OK")
        success = True
    except:
        print(f"  ERREUR: Échec transfert array")
        success = False

    return success


def test_worker_simple():
    """Test 7: Worker simple avec réseau local CPU."""
    print("\n" + "="*60)
    print("TEST 7: Worker simple (réseau CPU, 5 sims)")
    print("="*60)

    def simple_worker(output_queue, weights_path):
        try:
            # Créer réseau sur CPU
            config = NetworkConfig(num_filters=64, num_res_blocks=6)
            network = AlphaZeroNetwork(config)
            network.to('cpu')
            network.eval()

            if os.path.exists(weights_path):
                network.load_state_dict(torch.load(weights_path, map_location='cpu'))

            # MCTS avec peu de simulations
            mcts_config = MCTSConfig(num_simulations=5, c_puct=1.5)
            mcts = MCTS(config=mcts_config, network=network)

            game = Quarto()
            game.choose_piece(1)

            start = time.time()
            probs, _ = mcts.search(game, temperature=1.0)
            elapsed = time.time() - start

            output_queue.put({'success': True, 'time': elapsed})
        except Exception as e:
            output_queue.put({'success': False, 'error': str(e)})

    ctx = mp.get_context('spawn')
    output_queue = ctx.Queue()

    p = ctx.Process(
        target=simple_worker,
        args=(output_queue, "data/models/best_model.pt")
    )
    p.start()

    try:
        result = output_queue.get(timeout=120.0)
        p.join(timeout=5.0)

        if result.get('success'):
            print(f"  Worker OK: {result['time']:.2f}s pour 5 sims")
            return True
        else:
            print(f"  Worker ERREUR: {result.get('error')}")
            return False
    except:
        print(f"  TIMEOUT: Worker ne répond pas")
        p.terminate()
        return False


def main():
    print("="*60)
    print("DIAGNOSTIC ARCHITECTURE SERVER")
    print("="*60)

    results = {}

    # Test 1: GPU
    results['gpu'] = test_gpu()

    # Test 2: Réseau GPU
    if results['gpu']:
        network, device = test_network_gpu()
        results['network'] = network is not None
    else:
        results['network'] = False
        network, device = None, 'cpu'

    # Test 3: MCTS 10 sims
    if results['network']:
        results['mcts_10'] = test_mcts_gpu(network, device)

    # Test 4: MCTS 100 sims
    if results.get('mcts_10'):
        results['mcts_100'] = test_mcts_gpu_100(network, device)

    # Test 5: InferenceServer
    results['inference_server'] = test_inference_server_thread()

    # Test 6: MP Queues
    results['mp_queues'] = test_mp_queues()

    # Test 7: Worker simple
    results['worker_simple'] = test_worker_simple()

    # Résumé
    print("\n" + "="*60)
    print("RÉSUMÉ")
    print("="*60)
    for test, success in results.items():
        status = "✓ OK" if success else "✗ ÉCHEC"
        print(f"  {test}: {status}")

    # Diagnostic
    print("\n" + "="*60)
    print("DIAGNOSTIC")
    print("="*60)

    if not results.get('gpu'):
        print("  → GPU non disponible. Vérifier l'installation CUDA.")
    elif not results.get('mcts_100'):
        print("  → MCTS 100 sims trop lent. Le réseau n'est peut-être pas sur GPU.")
    elif not results.get('inference_server'):
        print("  → InferenceServer ne répond pas. Problème de threading.")
    elif not results.get('worker_simple'):
        print("  → Worker ne démarre pas. Problème de multiprocessing/spawn.")
    else:
        print("  → Tous les composants fonctionnent!")
        print("  → Le problème est dans l'intégration. Essayer --arch server.")


if __name__ == '__main__':
    main()

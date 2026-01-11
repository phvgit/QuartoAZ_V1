# -*- coding: utf-8 -*-
"""
Serveur d'inférence batch GPU pour AlphaZero Quarto.

Ce module implémente un serveur centralisé qui:
- Collecte les requêtes d'inférence de plusieurs workers
- Les regroupe en batches pour une inférence GPU efficace
- Redistribue les résultats aux workers correspondants

Architecture:
    Workers (CPU) --> request_queue --> InferenceServer (GPU) --> result_queues --> Workers
"""

import threading
import queue
import time
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import numpy as np
import torch

from alphaquarto.ai.network import AlphaZeroNetwork
from alphaquarto.utils.config import InferenceConfig


# =============================================================================
# Protocole de communication
# =============================================================================

@dataclass
class InferenceRequest:
    """Requête d'inférence envoyée par un worker."""
    worker_id: int
    request_id: int
    state: np.ndarray  # Shape: (21, 4, 4)


@dataclass
class InferenceResult:
    """Résultat d'inférence renvoyé au worker."""
    worker_id: int
    request_id: int
    policy: np.ndarray    # Shape: (16,)
    piece: np.ndarray     # Shape: (16,)
    value: float


# Signaux de contrôle
SHUTDOWN_SIGNAL = "SHUTDOWN"
RELOAD_WEIGHTS_SIGNAL = "RELOAD"


# =============================================================================
# Serveur d'inférence
# =============================================================================

class InferenceServer:
    """
    Serveur d'inférence centralisé avec batching GPU.

    Fonctionne dans un thread séparé, collecte les requêtes de plusieurs
    workers, les regroupe en batches et exécute l'inférence sur GPU.

    Attributes:
        network: Réseau de neurones AlphaZero
        config: Configuration du serveur
        request_queue: Queue partagée pour recevoir les requêtes
        result_queues: Dict worker_id -> Queue pour renvoyer les résultats
    """

    def __init__(
        self,
        network: AlphaZeroNetwork,
        config: InferenceConfig,
        request_queue: queue.Queue,
        result_queues: Dict[int, queue.Queue],
    ):
        """
        Initialise le serveur d'inférence.

        Args:
            network: Réseau AlphaZero (doit être sur le bon device)
            config: Configuration d'inférence
            request_queue: Queue pour recevoir les requêtes des workers
            result_queues: Dictionnaire worker_id -> queue de résultats
        """
        self.network = network
        self.config = config
        self.request_queue = request_queue
        self.result_queues = result_queues

        # Configuration device
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)
        self.network.eval()

        # État du serveur
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._weights_path: Optional[str] = None
        self._pending_state_dict: Optional[dict] = None

        # Statistiques
        self.stats = {
            'total_requests': 0,
            'total_batches': 0,
            'total_inference_time': 0.0,
            'avg_batch_size': 0.0
        }

    def start(self):
        """Démarre le serveur d'inférence dans un thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="InferenceServer")
        self._thread.start()

    def stop(self, timeout: float = 5.0):
        """
        Arrête proprement le serveur.

        Args:
            timeout: Temps d'attente max pour l'arrêt du thread
        """
        if not self._running:
            return

        self._running = False
        # Envoyer un signal pour débloquer le thread
        try:
            self.request_queue.put(SHUTDOWN_SIGNAL, timeout=1.0)
        except queue.Full:
            pass

        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None

    def reload_weights(self, weights_source):
        """
        Signale au serveur de recharger les poids.

        Args:
            weights_source: Chemin vers le fichier de poids (str)
                           ou state_dict directement (dict)
        """
        if isinstance(weights_source, str):
            self._weights_path = weights_source
            self._pending_state_dict = None
        else:
            # C'est un state_dict
            self._weights_path = None
            self._pending_state_dict = weights_source
        try:
            self.request_queue.put(RELOAD_WEIGHTS_SIGNAL, timeout=1.0)
        except queue.Full:
            pass

    def is_running(self) -> bool:
        """Retourne True si le serveur est en cours d'exécution."""
        return self._running and self._thread is not None and self._thread.is_alive()

    def get_stats(self) -> dict:
        """Retourne les statistiques du serveur."""
        stats = self.stats.copy()
        if stats['total_batches'] > 0:
            stats['avg_batch_size'] = stats['total_requests'] / stats['total_batches']
            stats['avg_inference_time_ms'] = (stats['total_inference_time'] / stats['total_batches']) * 1000
        return stats

    # =========================================================================
    # Boucle principale
    # =========================================================================

    def _run_loop(self):
        """Boucle principale du serveur - collecte, batch, inference, dispatch."""
        batch_timeout_s = self.config.batch_timeout_ms / 1000.0

        while self._running:
            batch: List[InferenceRequest] = []
            deadline = time.time() + batch_timeout_s

            # Phase 1: Collecter les requêtes jusqu'à batch plein ou timeout
            while len(batch) < self.config.max_batch_size:
                try:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        break

                    item = self.request_queue.get(timeout=max(0.001, remaining))

                    # Gérer les signaux de contrôle
                    if item == SHUTDOWN_SIGNAL:
                        self._running = False
                        return
                    elif item == RELOAD_WEIGHTS_SIGNAL:
                        self._do_reload_weights()
                        continue

                    batch.append(item)

                except queue.Empty:
                    break

            # Phase 2: Traiter le batch si non vide
            if batch:
                self._process_batch(batch)

    def _process_batch(self, batch: List[InferenceRequest]):
        """
        Exécute l'inférence sur un batch et distribue les résultats.

        Args:
            batch: Liste de requêtes à traiter
        """
        start_time = time.time()

        # Empiler les états: (N, 21, 4, 4)
        states = np.stack([req.state for req in batch])
        states_tensor = torch.from_numpy(states).to(self.device)

        # Inférence
        with torch.no_grad():
            policies, pieces, values = self.network(states_tensor)

        # Convertir en numpy
        policies_np = policies.cpu().numpy()
        pieces_np = pieces.cpu().numpy()
        values_np = values.cpu().numpy().flatten()

        # Dispatcher les résultats aux workers
        for i, req in enumerate(batch):
            result = InferenceResult(
                worker_id=req.worker_id,
                request_id=req.request_id,
                policy=policies_np[i],
                piece=pieces_np[i],
                value=float(values_np[i])
            )

            # Envoyer au bon worker
            if req.worker_id in self.result_queues:
                try:
                    self.result_queues[req.worker_id].put(result, timeout=1.0)
                except queue.Full:
                    # Le worker ne lit pas assez vite
                    pass

        # Mise à jour des statistiques
        elapsed = time.time() - start_time
        self.stats['total_requests'] += len(batch)
        self.stats['total_batches'] += 1
        self.stats['total_inference_time'] += elapsed

    def _do_reload_weights(self):
        """Recharge les poids du réseau depuis le fichier ou le state_dict."""
        try:
            if self._pending_state_dict is not None:
                # Rechargement depuis un state_dict
                self.network.load_state_dict(self._pending_state_dict)
                self.network.eval()
                self._pending_state_dict = None
            elif self._weights_path and os.path.exists(self._weights_path):
                # Rechargement depuis un fichier
                state_dict = torch.load(self._weights_path, map_location=self.device)
                self.network.load_state_dict(state_dict)
                self.network.eval()
        except Exception as e:
            print(f"[InferenceServer] Erreur rechargement poids: {e}")


# =============================================================================
# Fonctions utilitaires pour création de queues
# =============================================================================

def create_inference_queues(num_workers: int) -> tuple:
    """
    Crée les queues nécessaires pour la communication worker-server.

    Args:
        num_workers: Nombre de workers

    Returns:
        Tuple (request_queue, result_queues)
            - request_queue: Queue partagée pour les requêtes
            - result_queues: Dict worker_id -> Queue de résultats
    """
    request_queue = queue.Queue(maxsize=num_workers * 100)
    result_queues = {i: queue.Queue(maxsize=100) for i in range(num_workers)}
    return request_queue, result_queues


def create_mp_inference_queues(num_workers: int):
    """
    Crée les queues multiprocessing pour la communication inter-process.

    À utiliser quand les workers sont des processus séparés.

    Args:
        num_workers: Nombre de workers

    Returns:
        Tuple (request_queue, result_queues) avec des mp.Queue
    """
    import multiprocessing as mp
    request_queue = mp.Queue(maxsize=num_workers * 100)
    result_queues = {i: mp.Queue(maxsize=100) for i in range(num_workers)}
    return request_queue, result_queues

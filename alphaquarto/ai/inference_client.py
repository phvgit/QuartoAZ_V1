# -*- coding: utf-8 -*-
"""
Client d'inférence pour les workers AlphaZero.

Ce module fournit une interface simple pour les workers afin de
demander des inférences au serveur centralisé.

Usage:
    client = InferenceClient(worker_id=0, request_queue=..., result_queue=...)
    policy, piece, value = client.predict(state)
"""

import queue
from threading import Lock
from typing import Tuple, Union, Any
import numpy as np

from alphaquarto.ai.inference_server import InferenceRequest, InferenceResult


class InferenceClient:
    """
    Client pour demander des inférences au serveur centralisé.

    Chaque worker possède une instance de ce client avec un worker_id unique.
    Le client envoie des requêtes sur la queue partagée et attend les résultats
    sur sa queue de résultats dédiée.

    Attributes:
        worker_id: Identifiant unique du worker
        request_queue: Queue partagée pour envoyer les requêtes
        result_queue: Queue dédiée pour recevoir les résultats
        timeout: Timeout en secondes pour attendre une réponse
    """

    def __init__(
        self,
        worker_id: int,
        request_queue: Any,  # queue.Queue ou multiprocessing.Queue
        result_queue: Any,
        timeout: float = 2.0
    ):
        """
        Initialise le client d'inférence.

        Args:
            worker_id: Identifiant unique du worker (doit correspondre
                      à une clé dans result_queues du serveur)
            request_queue: Queue partagée pour les requêtes
            result_queue: Queue pour recevoir les résultats de ce worker
            timeout: Timeout en secondes pour attendre une réponse
        """
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.result_queue = result_queue
        self.timeout = timeout

        self._request_counter = 0
        self._lock = Lock()

    def predict(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Demande une inférence pour un état encodé.

        Args:
            state: État encodé du jeu (21, 4, 4)

        Returns:
            Tuple (policy, piece, value):
                - policy: Distribution sur les 16 cases
                - piece: Distribution sur les 16 pièces
                - value: Évaluation de la position [-1, 1]

        Raises:
            TimeoutError: Si le serveur ne répond pas dans le délai
            RuntimeError: Si l'ID de requête ne correspond pas
        """
        # Générer un ID de requête unique (thread-safe)
        with self._lock:
            request_id = self._request_counter
            self._request_counter += 1

        # Créer et envoyer la requête
        request = InferenceRequest(
            worker_id=self.worker_id,
            request_id=request_id,
            state=state
        )

        try:
            self.request_queue.put(request, timeout=self.timeout)
        except queue.Full:
            raise TimeoutError(f"Worker {self.worker_id}: Queue de requêtes pleine")

        # Attendre le résultat
        try:
            result: InferenceResult = self.result_queue.get(timeout=self.timeout)
        except queue.Empty:
            raise TimeoutError(
                f"Worker {self.worker_id}: Timeout en attente de résultat "
                f"(request_id={request_id})"
            )

        # Vérifier la correspondance
        if result.request_id != request_id:
            raise RuntimeError(
                f"Worker {self.worker_id}: Désynchronisation - "
                f"attendu request_id={request_id}, reçu {result.request_id}"
            )

        return result.policy, result.piece, result.value

    def predict_with_retry(
        self,
        state: np.ndarray,
        max_retries: int = 3
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Demande une inférence avec réessais en cas de timeout.

        Args:
            state: État encodé du jeu (21, 4, 4)
            max_retries: Nombre maximum de tentatives

        Returns:
            Tuple (policy, piece, value)

        Raises:
            TimeoutError: Si toutes les tentatives échouent
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                return self.predict(state)
            except TimeoutError as e:
                last_error = e
                # Vider la queue de résultats en cas de désynchronisation
                self._drain_result_queue()

        raise TimeoutError(
            f"Worker {self.worker_id}: Échec après {max_retries} tentatives. "
            f"Dernière erreur: {last_error}"
        )

    def _drain_result_queue(self):
        """Vide la queue de résultats (en cas de désynchronisation)."""
        try:
            while True:
                self.result_queue.get_nowait()
        except queue.Empty:
            pass

    def is_server_responsive(self, test_state: np.ndarray = None) -> bool:
        """
        Vérifie si le serveur répond.

        Args:
            test_state: État à utiliser pour le test. Si None, crée un état vide.

        Returns:
            True si le serveur répond dans le délai
        """
        if test_state is None:
            test_state = np.zeros((21, 4, 4), dtype=np.float32)

        try:
            self.predict(test_state)
            return True
        except (TimeoutError, RuntimeError):
            return False


class DummyInferenceClient:
    """
    Client d'inférence factice pour les tests sans serveur.

    Retourne des prédictions aléatoires uniformes.
    """

    def __init__(self, worker_id: int = 0):
        self.worker_id = worker_id

    def predict(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Retourne des prédictions uniformes."""
        policy = np.ones(16, dtype=np.float32) / 16
        piece = np.ones(16, dtype=np.float32) / 16
        value = 0.0
        return policy, piece, value

    def predict_with_retry(
        self,
        state: np.ndarray,
        max_retries: int = 3
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Identique à predict (pas de retry nécessaire)."""
        return self.predict(state)

    def is_server_responsive(self, test_state: np.ndarray = None) -> bool:
        """Toujours responsive."""
        return True

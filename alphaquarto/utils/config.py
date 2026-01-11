# -*- coding: utf-8 -*-
"""
Configuration dataclasses pour AlphaZero Quarto.

Toute la configuration est centralisée ici avec des valeurs par défaut sensées.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NetworkConfig:
    """Configuration du réseau de neurones."""

    # Architecture
    num_filters: int = 64
    num_res_blocks: int = 6
    input_channels: int = 21  # 16 pièces + 4 props pièce courante + 1 indicateur
    board_size: int = 4

    # Régularisation
    l2_reg: float = 1e-4

    # Optimisation (0.0003 avec cosine decay pour éviter l'overfitting)
    learning_rate: float = 0.0003

    @classmethod
    def small(cls) -> 'NetworkConfig':
        """Configuration pour réseau petit (tests rapides)."""
        return cls(num_filters=32, num_res_blocks=2)

    @classmethod
    def medium(cls) -> 'NetworkConfig':
        """Configuration par défaut."""
        return cls(num_filters=64, num_res_blocks=6)

    @classmethod
    def large(cls) -> 'NetworkConfig':
        """Configuration pour réseau large (entraînement intensif)."""
        return cls(num_filters=128, num_res_blocks=10)


@dataclass
class MCTSConfig:
    """Configuration du Monte Carlo Tree Search."""

    # Nombre de simulations par recherche
    num_simulations: int = 100

    # Constante d'exploration UCB (sqrt(2) par défaut)
    c_puct: float = 1.41

    # Bruit Dirichlet pour exploration à la racine
    use_dirichlet: bool = True
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25  # 25% bruit, 75% prior réseau

    # Température pour sélection d'action
    temperature_threshold: int = 10  # Après ce nombre de coups, température basse


@dataclass
class InferenceConfig:
    """Configuration du serveur d'inférence batch."""

    # Taille maximale des batches
    max_batch_size: int = 64

    # Timeout pour collecter un batch (ms)
    batch_timeout_ms: float = 10.0

    # Device PyTorch
    device: str = "cuda"  # "cuda" ou "cpu"

    # Timeout pour les requêtes workers (secondes)
    worker_timeout: float = 2.0


@dataclass
class TrainingConfig:
    """Configuration de l'entraînement."""

    # Boucle principale
    iterations: int = 100
    games_per_iteration: int = 200
    epochs_per_iteration: int = 2  # Réduit de 10 à 2 pour éviter l'overfitting

    # Batch size pour entraînement
    batch_size: int = 64

    # Replay buffer
    # Taille réduite pour éviter le distribution shift (données obsolètes)
    buffer_size: int = 50_000  # ~15-20 itérations de données
    min_buffer_size: int = 500  # Minimum avant de commencer l'entraînement

    # Rafraîchissement du buffer (évite les données obsolètes)
    clear_buffer_after_training: bool = False  # Si True, vide le buffer après chaque training

    # Parallélisation
    num_workers: int = 8

    # Checkpoints
    checkpoint_dir: str = "data/checkpoints"
    model_dir: str = "data/models"
    save_frequency: int = 1  # Sauvegarder tous les N itérations

    # Logging
    verbose: bool = True


@dataclass
class Config:
    """Configuration complète pour AlphaZero Quarto."""

    network: NetworkConfig = field(default_factory=NetworkConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def default(cls) -> 'Config':
        """Configuration par défaut pour entraînement standard."""
        return cls()

    @classmethod
    def quick_test(cls) -> 'Config':
        """Configuration pour test rapide."""
        return cls(
            network=NetworkConfig.small(),
            mcts=MCTSConfig(num_simulations=20),
            inference=InferenceConfig(device="cpu"),
            training=TrainingConfig(
                iterations=1,
                games_per_iteration=4,
                epochs_per_iteration=2,
                num_workers=2,
                min_buffer_size=10,
            )
        )

    @classmethod
    def production(cls, num_workers: int = 10) -> 'Config':
        """Configuration pour entraînement production sur GPU."""
        return cls(
            network=NetworkConfig.medium(),
            mcts=MCTSConfig(num_simulations=400),
            inference=InferenceConfig(device="cuda", max_batch_size=64),
            training=TrainingConfig(
                iterations=100,
                games_per_iteration=200,
                epochs_per_iteration=10,
                num_workers=num_workers,
                batch_size=128,
            )
        )

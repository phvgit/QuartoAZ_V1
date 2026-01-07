# -*- coding: utf-8 -*-
"""
Réseau de neurones AlphaZero pour Quarto.

Architecture:
- Input: État du plateau encodé (4x4x21 canaux)
- Backbone: Blocs résiduels (ResNet)
- Policy head: Distribution sur les 16 cases
- Piece head: Distribution sur les 16 pièces
- Value head: Évaluation de la position (-1 à 1)
"""

# =============================================================================
# CRITICAL: Configuration TensorFlow AVANT tout import
# TF_CPP_MIN_LOG_LEVEL=3 supprime INFO, WARNING et ERROR (C++ layer)
# Doit être défini AVANT l'import de tensorflow
# =============================================================================
import os
if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
if 'TF_ENABLE_ONEDNN_OPTS' not in os.environ:
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
from typing import Tuple, Optional
from pathlib import Path

# Import TensorFlow avec gestion d'erreur
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model, regularizers

    # Configurer le logger Python de TensorFlow
    tf.get_logger().setLevel('ERROR')

    # Désactiver aussi les logs absl
    import logging
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('absl').setLevel(logging.ERROR)

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. Using random network.")

from alphaquarto.game.constants import BOARD_SIZE, NUM_SQUARES, NUM_PIECES, NUM_PROPERTIES


# =============================================================================
# Configuration GPU
# =============================================================================

def configure_gpu(
    use_gpu: bool = True,
    memory_growth: bool = True,
    mixed_precision: bool = False,
    verbose: bool = True
) -> dict:
    """
    Configure TensorFlow pour utiliser le GPU de manière optimale.

    Args:
        use_gpu: Si True, utilise le GPU si disponible. Si False, force CPU.
        memory_growth: Si True, alloue la mémoire GPU progressivement.
        mixed_precision: Si True, utilise la précision mixte (float16/float32)
                        pour accélérer le calcul sur GPU compatible.
        verbose: Afficher les informations de configuration.

    Returns:
        Dictionnaire avec les informations de configuration
    """
    if not TF_AVAILABLE:
        return {'gpu_available': False, 'device': 'cpu', 'error': 'TensorFlow not available'}

    config = {
        'gpu_available': False,
        'gpu_name': None,
        'device': 'cpu',
        'mixed_precision': False,
        'memory_growth': False
    }

    # Lister les GPUs disponibles
    gpus = tf.config.list_physical_devices('GPU')

    if not use_gpu or not gpus:
        # Forcer l'utilisation du CPU
        if not use_gpu:
            tf.config.set_visible_devices([], 'GPU')
            if verbose:
                print("  GPU désactivé, utilisation du CPU")
        elif verbose:
            print("  Aucun GPU détecté, utilisation du CPU")
        return config

    # GPU disponible
    config['gpu_available'] = True
    config['device'] = 'gpu'

    try:
        # Obtenir le nom du GPU
        config['gpu_name'] = gpus[0].name

        # Configuration memory growth (évite d'allouer toute la VRAM)
        if memory_growth:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            config['memory_growth'] = True

        # Mixed precision (float16 pour les calculs, float32 pour les poids)
        # Accélère significativement sur les GPU avec Tensor Cores (RTX, etc.)
        if mixed_precision:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            config['mixed_precision'] = True

        if verbose:
            print(f"  GPU détecté: {gpus[0].name}")
            print(f"  Memory growth: {'Activé' if memory_growth else 'Désactivé'}")
            print(f"  Mixed precision (float16): {'Activé' if mixed_precision else 'Désactivé'}")

            # Afficher la mémoire GPU disponible si possible
            try:
                gpu_details = tf.config.experimental.get_device_details(gpus[0])
                if 'device_name' in gpu_details:
                    print(f"  Nom complet: {gpu_details['device_name']}")
            except Exception:
                pass

    except RuntimeError as e:
        if verbose:
            print(f"  Erreur configuration GPU: {e}")
        config['error'] = str(e)

    return config


def get_gpu_info() -> dict:
    """
    Retourne les informations sur le GPU disponible.

    Returns:
        Dictionnaire avec les informations GPU
    """
    if not TF_AVAILABLE:
        return {'available': False, 'error': 'TensorFlow not available'}

    gpus = tf.config.list_physical_devices('GPU')

    if not gpus:
        return {'available': False, 'count': 0}

    info = {
        'available': True,
        'count': len(gpus),
        'devices': []
    }

    for gpu in gpus:
        device_info = {'name': gpu.name}
        try:
            details = tf.config.experimental.get_device_details(gpu)
            device_info.update(details)
        except Exception:
            pass
        info['devices'].append(device_info)

    return info


# =============================================================================
# Encodeur d'état
# =============================================================================

class StateEncoder:
    """
    Encode l'état du jeu Quarto en tenseur pour le réseau de neurones.

    Canaux d'entrée (21 total):
    - Canaux 0-15: Présence de chaque pièce (one-hot par case)
    - Canaux 16-19: Propriétés de la pièce courante (4 canaux binaires)
    - Canal 20: Masque de la pièce courante (1 si pièce en main)
    """

    NUM_CHANNELS = 21

    @staticmethod
    def encode(game) -> np.ndarray:
        """
        Encode un état de jeu en tenseur.

        Args:
            game: Instance du jeu Quarto

        Returns:
            Tenseur de forme (4, 4, 21)
        """
        state = np.zeros((BOARD_SIZE, BOARD_SIZE, StateEncoder.NUM_CHANNELS), dtype=np.float32)

        # Canaux 0-15: Pièces sur le plateau (one-hot)
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                piece = game.board[i, j]
                if piece > 0:
                    state[i, j, piece - 1] = 1.0

        # Canaux 16-19: Propriétés de la pièce courante
        if game.current_piece is not None:
            piece_bits = game.current_piece - 1
            color = (piece_bits >> 0) & 1
            shape = (piece_bits >> 1) & 1
            size = (piece_bits >> 2) & 1
            hole = (piece_bits >> 3) & 1

            # Remplir tout le plan avec la propriété (broadcast)
            state[:, :, 16] = color
            state[:, :, 17] = shape
            state[:, :, 18] = size
            state[:, :, 19] = hole

            # Canal 20: Indicateur de pièce en main
            state[:, :, 20] = 1.0

        return state

    @staticmethod
    def encode_batch(games: list) -> np.ndarray:
        """Encode un batch d'états de jeu."""
        return np.array([StateEncoder.encode(g) for g in games])


# =============================================================================
# Blocs résiduels
# =============================================================================

def residual_block(x, filters: int, kernel_size: int = 3, l2_reg: float = 1e-4):
    """
    Crée un bloc résiduel avec skip connection.

    Architecture:
    x -> Conv -> BN -> ReLU -> Conv -> BN -> Add(x) -> ReLU
    """
    shortcut = x

    # Première convolution
    x = layers.Conv2D(
        filters, kernel_size,
        padding='same',
        use_bias=False,
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Deuxième convolution
    x = layers.Conv2D(
        filters, kernel_size,
        padding='same',
        use_bias=False,
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    x = layers.BatchNormalization()(x)

    # Skip connection
    x = layers.Add()([shortcut, x])
    x = layers.ReLU()(x)

    return x


# =============================================================================
# Réseau AlphaZero
# =============================================================================

class AlphaZeroNetwork:
    """
    Réseau de neurones AlphaZero pour Quarto.

    Architecture:
    - Input: (4, 4, 21) - état encodé du jeu
    - Backbone: Conv initiale + N blocs résiduels
    - Policy head: Probabilités sur les 16 cases
    - Piece head: Probabilités sur les 16 pièces
    - Value head: Évaluation de la position [-1, 1]

    Attributes:
        model: Modèle Keras compilé
        num_filters: Nombre de filtres dans les convolutions
        num_res_blocks: Nombre de blocs résiduels
        l2_reg: Régularisation L2
    """

    def __init__(
        self,
        num_filters: int = 64,
        num_res_blocks: int = 6,
        l2_reg: float = 1e-4,
        learning_rate: float = 0.001
    ):
        """
        Initialise le réseau AlphaZero.

        Args:
            num_filters: Nombre de filtres par couche conv
            num_res_blocks: Nombre de blocs résiduels
            l2_reg: Coefficient de régularisation L2
            learning_rate: Taux d'apprentissage
        """
        self.num_filters = num_filters
        self.num_res_blocks = num_res_blocks
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.encoder = StateEncoder()

        if TF_AVAILABLE:
            self.model = self._build_model()
        else:
            self.model = None

    def _build_model(self) -> Model:
        """Construit le modèle Keras."""

        # Input
        input_shape = (BOARD_SIZE, BOARD_SIZE, StateEncoder.NUM_CHANNELS)
        inputs = layers.Input(shape=input_shape, name='board_input')

        # Convolution initiale
        x = layers.Conv2D(
            self.num_filters, 3,
            padding='same',
            use_bias=False,
            kernel_regularizer=regularizers.l2(self.l2_reg),
            name='initial_conv'
        )(inputs)
        x = layers.BatchNormalization(name='initial_bn')(x)
        x = layers.ReLU(name='initial_relu')(x)

        # Blocs résiduels
        for i in range(self.num_res_blocks):
            x = residual_block(x, self.num_filters, l2_reg=self.l2_reg)

        # =====================================================================
        # Policy Head (sélection de case)
        # =====================================================================
        policy = layers.Conv2D(
            2, 1,
            padding='same',
            use_bias=False,
            kernel_regularizer=regularizers.l2(self.l2_reg),
            name='policy_conv'
        )(x)
        policy = layers.BatchNormalization(name='policy_bn')(policy)
        policy = layers.ReLU(name='policy_relu')(policy)
        policy = layers.Flatten(name='policy_flatten')(policy)
        policy = layers.Dense(
            NUM_SQUARES,
            activation='softmax',
            kernel_regularizer=regularizers.l2(self.l2_reg),
            name='policy_output'
        )(policy)

        # =====================================================================
        # Piece Head (sélection de pièce à donner)
        # =====================================================================
        piece = layers.Conv2D(
            2, 1,
            padding='same',
            use_bias=False,
            kernel_regularizer=regularizers.l2(self.l2_reg),
            name='piece_conv'
        )(x)
        piece = layers.BatchNormalization(name='piece_bn')(piece)
        piece = layers.ReLU(name='piece_relu')(piece)
        piece = layers.Flatten(name='piece_flatten')(piece)
        piece = layers.Dense(
            NUM_PIECES,
            activation='softmax',
            kernel_regularizer=regularizers.l2(self.l2_reg),
            name='piece_output'
        )(piece)

        # =====================================================================
        # Value Head (évaluation de la position)
        # =====================================================================
        value = layers.Conv2D(
            1, 1,
            padding='same',
            use_bias=False,
            kernel_regularizer=regularizers.l2(self.l2_reg),
            name='value_conv'
        )(x)
        value = layers.BatchNormalization(name='value_bn')(value)
        value = layers.ReLU(name='value_relu')(value)
        value = layers.Flatten(name='value_flatten')(value)
        value = layers.Dense(
            64,
            activation='relu',
            kernel_regularizer=regularizers.l2(self.l2_reg),
            name='value_dense'
        )(value)
        value = layers.Dense(
            1,
            activation='tanh',
            kernel_regularizer=regularizers.l2(self.l2_reg),
            name='value_output'
        )(value)

        # Créer le modèle
        model = Model(
            inputs=inputs,
            outputs=[policy, piece, value],
            name='AlphaZeroQuarto'
        )

        # Compiler
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss={
                'policy_output': 'categorical_crossentropy',
                'piece_output': 'categorical_crossentropy',
                'value_output': 'mse'
            },
            loss_weights={
                'policy_output': 1.0,
                'piece_output': 1.0,
                'value_output': 1.0
            },
            metrics={
                'policy_output': 'accuracy',
                'piece_output': 'accuracy',
                'value_output': 'mae'
            }
        )

        return model

    # =========================================================================
    # Prédiction
    # =========================================================================

    def predict(self, game) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Prédit policy, piece et value pour un état de jeu.

        Args:
            game: Instance du jeu Quarto

        Returns:
            Tuple (policy, piece_probs, value):
                - policy: Distribution sur les 16 cases
                - piece_probs: Distribution sur les 16 pièces
                - value: Évaluation de la position [-1, 1]
        """
        if self.model is None:
            # Fallback aléatoire si TensorFlow non disponible
            return (
                np.ones(NUM_SQUARES) / NUM_SQUARES,
                np.ones(NUM_PIECES) / NUM_PIECES,
                0.0
            )

        # Encoder l'état
        state = self.encoder.encode(game)
        state_batch = np.expand_dims(state, axis=0)

        # Prédiction
        policy, piece, value = self.model.predict(state_batch, verbose=0)

        return policy[0], piece[0], float(value[0, 0])

    def predict_batch(self, games: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prédit pour un batch de jeux.

        Args:
            games: Liste d'instances de jeux Quarto

        Returns:
            Tuple (policies, piece_probs, values)
        """
        if self.model is None:
            n = len(games)
            return (
                np.ones((n, NUM_SQUARES)) / NUM_SQUARES,
                np.ones((n, NUM_PIECES)) / NUM_PIECES,
                np.zeros(n)
            )

        states = self.encoder.encode_batch(games)
        policies, pieces, values = self.model.predict(states, verbose=0)

        return policies, pieces, values.flatten()

    def __call__(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Interface pour MCTS: prédit depuis un état déjà encodé.

        Compatible avec l'interface attendue par MCTS._evaluate()

        Args:
            state: État encodé (4, 4, 16) ou (4, 4, 21)

        Returns:
            Tuple (policy, value) pour compatibilité MCTS
        """
        if self.model is None:
            return np.ones(NUM_SQUARES) / NUM_SQUARES, 0.0

        # Adapter l'état si nécessaire
        if state.shape[-1] == NUM_PIECES:
            # Ancien format (4, 4, 16) - ajouter canaux manquants
            full_state = np.zeros((BOARD_SIZE, BOARD_SIZE, StateEncoder.NUM_CHANNELS), dtype=np.float32)
            full_state[:, :, :NUM_PIECES] = state
        else:
            full_state = state

        state_batch = np.expand_dims(full_state, axis=0)
        policy, _, value = self.model.predict(state_batch, verbose=0)

        return policy[0], float(value[0, 0])

    # =========================================================================
    # Entraînement
    # =========================================================================

    def train_on_batch(
        self,
        states: np.ndarray,
        policy_targets: np.ndarray,
        piece_targets: np.ndarray,
        value_targets: np.ndarray
    ) -> dict:
        """
        Entraîne sur un batch de données.

        Args:
            states: Batch d'états encodés (N, 4, 4, 21)
            policy_targets: Distributions cibles sur cases (N, 16)
            piece_targets: Distributions cibles sur pièces (N, 16)
            value_targets: Valeurs cibles (N,)

        Returns:
            Dictionnaire avec les métriques de loss
        """
        if self.model is None:
            return {'loss': 0.0}

        # Reshape value targets si nécessaire
        if len(value_targets.shape) == 1:
            value_targets = value_targets.reshape(-1, 1)

        # Entraînement
        history = self.model.fit(
            states,
            {
                'policy_output': policy_targets,
                'piece_output': piece_targets,
                'value_output': value_targets
            },
            batch_size=len(states),
            epochs=1,
            verbose=0
        )

        return {
            'loss': history.history['loss'][0],
            'policy_loss': history.history['policy_output_loss'][0],
            'piece_loss': history.history['piece_output_loss'][0],
            'value_loss': history.history['value_output_loss'][0],
            'policy_accuracy': history.history['policy_output_accuracy'][0],
            'piece_accuracy': history.history['piece_output_accuracy'][0],
            'value_mae': history.history['value_output_mae'][0]
        }

    def train_epoch(
        self,
        states: np.ndarray,
        policy_targets: np.ndarray,
        piece_targets: np.ndarray,
        value_targets: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True
    ) -> dict:
        """
        Entraîne une époque complète sur un dataset.

        Args:
            states: Tous les états
            policy_targets: Toutes les cibles policy
            piece_targets: Toutes les cibles piece
            value_targets: Toutes les valeurs cibles
            batch_size: Taille des mini-batches
            shuffle: Mélanger les données

        Returns:
            Métriques moyennes de l'époque
        """
        if self.model is None:
            return {'loss': 0.0}

        # Reshape value targets si nécessaire
        if len(value_targets.shape) == 1:
            value_targets = value_targets.reshape(-1, 1)

        history = self.model.fit(
            states,
            {
                'policy_output': policy_targets,
                'piece_output': piece_targets,
                'value_output': value_targets
            },
            batch_size=batch_size,
            epochs=1,
            shuffle=shuffle,
            verbose=0
        )

        return {
            'loss': history.history['loss'][0],
            'policy_loss': history.history['policy_output_loss'][0],
            'piece_loss': history.history['piece_output_loss'][0],
            'value_loss': history.history['value_output_loss'][0],
            'policy_accuracy': history.history['policy_output_accuracy'][0],
            'piece_accuracy': history.history['piece_output_accuracy'][0]
        }

    # =========================================================================
    # Sauvegarde / Chargement
    # =========================================================================

    def save_weights(self, path: str):
        """Sauvegarde les poids du modèle."""
        if self.model is not None:
            # Créer le répertoire parent si nécessaire
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.model.save_weights(path)

    def load_weights(self, path: str, skip_optimizer: bool = False):
        """
        Charge les poids du modèle.

        Args:
            path: Chemin vers le fichier de poids
            skip_optimizer: Si True, ignore les variables de l'optimizer (évite warnings)
                           Utile pour l'inférence seule.
        """
        if self.model is not None and Path(path).exists():
            if skip_optimizer:
                self.model.load_weights(path, skip_mismatch=True)
            else:
                self.model.load_weights(path)

    def save_model(self, path: str):
        """Sauvegarde le modèle complet."""
        if self.model is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.model.save(path)

    @classmethod
    def load_model(cls, path: str) -> 'AlphaZeroNetwork':
        """Charge un modèle complet."""
        instance = cls()
        if TF_AVAILABLE and Path(path).exists():
            instance.model = keras.models.load_model(path)
        return instance

    # =========================================================================
    # Utilitaires
    # =========================================================================

    def summary(self):
        """Affiche un résumé du modèle."""
        if self.model is not None:
            self.model.summary()

    def get_config(self) -> dict:
        """Retourne la configuration du réseau."""
        return {
            'num_filters': self.num_filters,
            'num_res_blocks': self.num_res_blocks,
            'l2_reg': self.l2_reg,
            'learning_rate': self.learning_rate,
            'input_channels': StateEncoder.NUM_CHANNELS
        }

    def count_parameters(self) -> int:
        """Compte le nombre de paramètres du modèle."""
        if self.model is not None:
            return self.model.count_params()
        return 0


# =============================================================================
# Fonction utilitaire pour créer un réseau pré-configuré
# =============================================================================

def create_network(
    size: str = 'medium',
    learning_rate: float = 0.001
) -> AlphaZeroNetwork:
    """
    Crée un réseau avec une configuration prédéfinie.

    Args:
        size: 'small', 'medium', ou 'large'
        learning_rate: Taux d'apprentissage

    Returns:
        Instance d'AlphaZeroNetwork configurée
    """
    configs = {
        'small': {'num_filters': 32, 'num_res_blocks': 4},
        'medium': {'num_filters': 64, 'num_res_blocks': 6},
        'large': {'num_filters': 128, 'num_res_blocks': 10}
    }

    config = configs.get(size, configs['medium'])
    return AlphaZeroNetwork(
        num_filters=config['num_filters'],
        num_res_blocks=config['num_res_blocks'],
        learning_rate=learning_rate
    )

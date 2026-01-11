# -*- coding: utf-8 -*-
"""
Réseau de neurones AlphaZero pour Quarto (PyTorch).

Architecture:
- Input: État du plateau encodé (batch, 21, 4, 4) - channels first
- Backbone: Blocs résiduels (ResNet)
- Policy head: Distribution sur les 16 cases
- Piece head: Distribution sur les 16 pièces
- Value head: Évaluation de la position (-1 à 1)
"""

import numpy as np
from typing import Tuple, Optional, Union
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from alphaquarto.game.constants import BOARD_SIZE, NUM_SQUARES, NUM_PIECES
from alphaquarto.utils.config import NetworkConfig


# =============================================================================
# Encodeur d'état
# =============================================================================

class StateEncoder:
    """
    Encode l'état du jeu Quarto en tenseur pour le réseau de neurones.

    Canaux d'entrée (21 total) - Format channels-first (C, H, W):
    - Canaux 0-15: Présence de chaque pièce (one-hot par case)
    - Canaux 16-19: Propriétés de la pièce courante (4 canaux binaires)
    - Canal 20: Masque de la pièce courante (1 si pièce en main)
    """

    NUM_CHANNELS = 21

    @staticmethod
    def encode(game) -> np.ndarray:
        """
        Encode un état de jeu en tenseur channels-first.

        Args:
            game: Instance du jeu Quarto

        Returns:
            Tenseur de forme (21, 4, 4) - channels first pour PyTorch
        """
        state = np.zeros((StateEncoder.NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

        # Canaux 0-15: Pièces sur le plateau (one-hot)
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                piece = game.board[i, j]
                if piece > 0:
                    state[piece - 1, i, j] = 1.0

        # Canaux 16-19: Propriétés de la pièce courante
        if game.current_piece is not None:
            piece_bits = game.current_piece - 1
            color = (piece_bits >> 0) & 1
            shape = (piece_bits >> 1) & 1
            size = (piece_bits >> 2) & 1
            hole = (piece_bits >> 3) & 1

            # Remplir tout le plan avec la propriété (broadcast)
            state[16, :, :] = color
            state[17, :, :] = shape
            state[18, :, :] = size
            state[19, :, :] = hole

            # Canal 20: Indicateur de pièce en main
            state[20, :, :] = 1.0

        return state

    @staticmethod
    def encode_batch(games: list) -> np.ndarray:
        """
        Encode un batch d'états de jeu.

        Returns:
            Tenseur de forme (N, 21, 4, 4)
        """
        return np.array([StateEncoder.encode(g) for g in games])


# =============================================================================
# Blocs résiduels PyTorch
# =============================================================================

class ResidualBlock(nn.Module):
    """
    Bloc résiduel avec skip connection.

    Architecture:
    x -> Conv -> BN -> ReLU -> Conv -> BN -> Add(x) -> ReLU
    """

    def __init__(self, num_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return F.relu(out)


# =============================================================================
# Réseau AlphaZero PyTorch
# =============================================================================

class AlphaZeroNetwork(nn.Module):
    """
    Réseau de neurones AlphaZero pour Quarto (PyTorch).

    Architecture:
    - Input: (batch, 21, 4, 4) - état encodé du jeu (channels first)
    - Backbone: Conv initiale + N blocs résiduels
    - Policy head: Probabilités sur les 16 cases
    - Piece head: Probabilités sur les 16 pièces
    - Value head: Évaluation de la position [-1, 1]
    """

    def __init__(self, config: Optional[NetworkConfig] = None):
        """
        Initialise le réseau AlphaZero.

        Args:
            config: Configuration du réseau. Si None, utilise les valeurs par défaut.
        """
        super().__init__()

        if config is None:
            config = NetworkConfig()

        self.config = config
        self.num_filters = config.num_filters
        self.num_res_blocks = config.num_res_blocks

        # Convolution initiale
        self.conv_initial = nn.Conv2d(
            config.input_channels, config.num_filters, 3, padding=1, bias=False
        )
        self.bn_initial = nn.BatchNorm2d(config.num_filters)

        # Tour de blocs résiduels
        self.res_blocks = nn.ModuleList([
            ResidualBlock(config.num_filters)
            for _ in range(config.num_res_blocks)
        ])

        # Policy head (sélection de case)
        self.policy_conv = nn.Conv2d(config.num_filters, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, NUM_SQUARES)

        # Piece head (sélection de pièce à donner)
        self.piece_conv = nn.Conv2d(config.num_filters, 2, 1, bias=False)
        self.piece_bn = nn.BatchNorm2d(2)
        self.piece_fc = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, NUM_PIECES)

        # Value head (évaluation de la position)
        self.value_conv = nn.Conv2d(config.num_filters, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * BOARD_SIZE * BOARD_SIZE, 64)
        self.value_fc2 = nn.Linear(64, 1)

        # Initialisation des poids
        self._init_weights()

    def _init_weights(self):
        """Initialise les poids selon les bonnes pratiques."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass du réseau.

        Args:
            x: Tensor de forme (batch, 21, 4, 4)

        Returns:
            Tuple (policy, piece, value):
                - policy: (batch, 16) softmax sur les cases
                - piece: (batch, 16) softmax sur les pièces
                - value: (batch, 1) tanh entre -1 et 1
        """
        # Backbone
        x = F.relu(self.bn_initial(self.conv_initial(x)))
        for block in self.res_blocks:
            x = block(x)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = F.softmax(self.policy_fc(p), dim=1)

        # Piece head
        pi = F.relu(self.piece_bn(self.piece_conv(x)))
        pi = pi.view(pi.size(0), -1)
        pi = F.softmax(self.piece_fc(pi), dim=1)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, pi, v

    # =========================================================================
    # Prédiction (interface numpy)
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
        self.eval()
        state = StateEncoder.encode(game)
        state_tensor = torch.from_numpy(state).unsqueeze(0)

        # Déplacer vers le même device que le modèle
        device = next(self.parameters()).device
        state_tensor = state_tensor.to(device)

        with torch.no_grad():
            policy, piece, value = self.forward(state_tensor)

        return (
            policy[0].cpu().numpy(),
            piece[0].cpu().numpy(),
            float(value[0, 0].cpu())
        )

    def predict_batch(self, games: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prédit pour un batch de jeux.

        Args:
            games: Liste d'instances de jeux Quarto

        Returns:
            Tuple (policies, piece_probs, values)
        """
        self.eval()
        states = StateEncoder.encode_batch(games)
        states_tensor = torch.from_numpy(states)

        device = next(self.parameters()).device
        states_tensor = states_tensor.to(device)

        with torch.no_grad():
            policies, pieces, values = self.forward(states_tensor)

        return (
            policies.cpu().numpy(),
            pieces.cpu().numpy(),
            values.cpu().numpy().flatten()
        )

    def predict_from_encoded(
        self,
        states: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prédit depuis des états déjà encodés.

        Args:
            states: États encodés (N, 21, 4, 4)

        Returns:
            Tuple (policies, pieces, values)
        """
        self.eval()
        states_tensor = torch.from_numpy(states)

        device = next(self.parameters()).device
        states_tensor = states_tensor.to(device)

        with torch.no_grad():
            policies, pieces, values = self.forward(states_tensor)

        return (
            policies.cpu().numpy(),
            pieces.cpu().numpy(),
            values.cpu().numpy().flatten()
        )

    # =========================================================================
    # Sauvegarde / Chargement
    # =========================================================================

    def save_weights(self, path: str):
        """Sauvegarde les poids du modèle."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str, map_location: Optional[str] = None):
        """
        Charge les poids du modèle.

        Args:
            path: Chemin vers le fichier de poids
            map_location: Device de destination ('cpu', 'cuda', etc.)
        """
        if Path(path).exists():
            state_dict = torch.load(path, map_location=map_location)
            self.load_state_dict(state_dict)

    # =========================================================================
    # Utilitaires
    # =========================================================================

    def get_config(self) -> dict:
        """Retourne la configuration du réseau."""
        return {
            'num_filters': self.config.num_filters,
            'num_res_blocks': self.config.num_res_blocks,
            'l2_reg': self.config.l2_reg,
            'learning_rate': self.config.learning_rate,
            'input_channels': self.config.input_channels
        }

    def count_parameters(self) -> int:
        """Compte le nombre de paramètres du modèle."""
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self) -> int:
        """Compte le nombre de paramètres entraînables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Configuration GPU
# =============================================================================

def configure_gpu(
    use_gpu: bool = True,
    verbose: bool = True
) -> dict:
    """
    Configure PyTorch pour utiliser le GPU.

    Args:
        use_gpu: Si True, utilise le GPU si disponible
        verbose: Afficher les informations de configuration

    Returns:
        Dictionnaire avec les informations de configuration
    """
    config = {
        'gpu_available': torch.cuda.is_available(),
        'device': 'cpu',
        'gpu_name': None
    }

    if use_gpu and torch.cuda.is_available():
        config['device'] = 'cuda'
        config['gpu_name'] = torch.cuda.get_device_name(0)

        if verbose:
            print(f"  GPU détecté: {config['gpu_name']}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Mémoire GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif verbose:
        if not torch.cuda.is_available():
            print("  Aucun GPU détecté, utilisation du CPU")
        else:
            print("  GPU désactivé, utilisation du CPU")

    return config


def get_gpu_info() -> dict:
    """
    Retourne les informations sur le GPU disponible.

    Returns:
        Dictionnaire avec les informations GPU
    """
    if not torch.cuda.is_available():
        return {'available': False, 'count': 0}

    return {
        'available': True,
        'count': torch.cuda.device_count(),
        'devices': [
            {
                'name': torch.cuda.get_device_name(i),
                'memory': torch.cuda.get_device_properties(i).total_memory,
                'capability': torch.cuda.get_device_capability(i)
            }
            for i in range(torch.cuda.device_count())
        ]
    }


# =============================================================================
# Fonction utilitaire pour créer un réseau
# =============================================================================

def create_network(
    size: str = 'medium',
    learning_rate: float = 0.001
) -> AlphaZeroNetwork:
    """
    Crée un réseau avec une configuration prédéfinie.

    Args:
        size: 'small', 'medium', ou 'large'
        learning_rate: Taux d'apprentissage (pour info, utilisé par trainer)

    Returns:
        Instance d'AlphaZeroNetwork configurée
    """
    if size == 'small':
        config = NetworkConfig.small()
    elif size == 'large':
        config = NetworkConfig.large()
    else:
        config = NetworkConfig.medium()

    config.learning_rate = learning_rate
    return AlphaZeroNetwork(config)

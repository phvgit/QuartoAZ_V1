# -*- coding: utf-8 -*-
"""
Fast MCTS wrapper using C++ implementation.

This module provides a Python interface to the C++ MCTS implementation
with batched neural network evaluation.

Usage:
    from alphaquarto.ai.mcts_fast import FastMCTS, MCTS_CPP_AVAILABLE

    if MCTS_CPP_AVAILABLE:
        mcts = FastMCTS(config, network)
    else:
        # Fall back to Python MCTS
        mcts = MCTS(config, network=network)
"""

import numpy as np
from typing import Tuple, Optional, Callable
import torch

from alphaquarto.game.quarto import Quarto
from alphaquarto.game.constants import NUM_SQUARES, NUM_PIECES
from alphaquarto.utils.config import MCTSConfig

# Try to import C++ module
try:
    from alphaquarto.ai.mcts_cpp import mcts_cpp
    MCTS_CPP_AVAILABLE = True
except ImportError:
    MCTS_CPP_AVAILABLE = False
    mcts_cpp = None


class FastMCTS:
    """
    Fast MCTS using C++ implementation with batched GPU inference.

    The C++ code handles tree operations (selection, expansion, backprop)
    while Python handles neural network evaluation on GPU.
    """

    def __init__(
        self,
        config: MCTSConfig,
        network = None,
        inference_client = None,
        batch_size: int = 8
    ):
        """
        Initialize FastMCTS.

        Args:
            config: MCTS configuration
            network: PyTorch network for evaluation (used if no inference_client)
            inference_client: Optional inference client for distributed evaluation
            batch_size: Number of leaves to evaluate in one batch
        """
        if not MCTS_CPP_AVAILABLE:
            raise ImportError(
                "C++ MCTS module not available. "
                "Compile it with: cd alphaquarto/ai/mcts_cpp && python setup.py build_ext --inplace"
            )

        self.config = config
        self.network = network
        self.inference_client = inference_client
        self.batch_size = batch_size

        # Create the C++ MCTS instance with our evaluation callback
        self._mcts = mcts_cpp.MCTS(
            num_simulations=config.num_simulations,
            c_puct=config.c_puct,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_epsilon=config.dirichlet_epsilon,
            batch_size=batch_size,
            eval_callback=self._eval_batch
        )

    def _eval_batch(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate a batch of states using the neural network.

        Args:
            states: Batch of encoded states (N, 21, 4, 4)

        Returns:
            Tuple of (policies, piece_probs, values)
        """
        if self.inference_client is not None:
            # Use distributed inference
            policies = []
            pieces = []
            values = []
            for state in states:
                p, pc, v = self.inference_client.predict(state)
                policies.append(p)
                pieces.append(pc)
                values.append(v)
            return np.array(policies), np.array(pieces), np.array(values)

        elif self.network is not None:
            # Direct network evaluation
            with torch.no_grad():
                # Convert to tensor
                states_t = torch.from_numpy(states).float()
                if next(self.network.parameters()).is_cuda:
                    states_t = states_t.cuda()

                # Forward pass
                policy, piece, value = self.network(states_t)

                # Convert back to numpy
                policies = policy.cpu().numpy()
                pieces = piece.cpu().numpy()
                values = value.cpu().numpy().flatten()

                return policies, pieces, values
        else:
            # Fallback: uniform policy
            n = len(states)
            policies = np.ones((n, NUM_SQUARES)) / NUM_SQUARES
            pieces = np.ones((n, NUM_PIECES)) / NUM_PIECES
            values = np.zeros(n)
            return policies, pieces, values

    def search(
        self,
        game: Quarto,
        temperature: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run MCTS search from the current game state.

        Args:
            game: Current game state
            temperature: Temperature for move selection

        Returns:
            Tuple of (move_probabilities, piece_probabilities)
        """
        # Convert game state to arrays for C++
        board = np.array(game.board.flatten(), dtype=np.int32)
        current_piece = game.current_piece if game.current_piece else 0
        current_player = game.current_player
        game_over = game.game_over
        winner = game.winner if game.winner is not None else -1

        # Run C++ MCTS
        move_probs, piece_probs = self._mcts.search(
            board, current_piece, current_player,
            game_over, winner, temperature
        )

        return np.array(move_probs), np.array(piece_probs)

    def get_best_move(self, game: Quarto) -> int:
        """Get the best move with temperature=0."""
        move_probs, _ = self.search(game, temperature=0.01)

        # Mask illegal moves
        legal = game.get_legal_moves()
        masked = np.zeros_like(move_probs)
        for m in legal:
            masked[m] = move_probs[m]

        return int(np.argmax(masked))

    def get_best_piece(self, game: Quarto) -> int:
        """Get the best piece to give to opponent."""
        _, piece_probs = self.search(game, temperature=0.01)

        # Mask unavailable pieces
        available = game.get_available_pieces()
        masked = np.zeros_like(piece_probs)
        for p in available:
            masked[p - 1] = piece_probs[p - 1]

        return int(np.argmax(masked)) + 1


def create_mcts(config: MCTSConfig, network=None, inference_client=None):
    """
    Factory function to create the best available MCTS implementation.

    Returns FastMCTS if C++ is available, otherwise falls back to Python MCTS.
    """
    if MCTS_CPP_AVAILABLE:
        return FastMCTS(config, network=network, inference_client=inference_client)
    else:
        # Fallback to Python implementation
        from alphaquarto.ai.mcts import MCTS
        return MCTS(config, network=network, inference_client=inference_client)

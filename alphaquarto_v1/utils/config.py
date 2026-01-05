# -*- coding: utf-8 -*-
"""Configuration"""

from dataclasses import dataclass

@dataclass
class TrainingConfig:
    iterations: int = 10
    games_per_iteration: int = 10
    mcts_simulations: int = 100

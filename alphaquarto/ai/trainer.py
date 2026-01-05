# -*- coding: utf-8 -*-
"""Entraneur AlphaZero"""

class AlphaZeroTrainer:
    """Entraneur avec selfplay"""
    
    def __init__(self, network, mcts_sims=100):
        self.network = network
        self.mcts_sims = mcts_sims
        self.replay_buffer = []
    
    def play_game(self):
        return []
    
    def train_iteration(self, num_games=10, batch_size=32):
        return 0.0

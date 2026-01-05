# -*- coding: utf-8 -*-
"""Rseau de neurones AlphaZero"""

import numpy as np
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:
    pass

class AlphaZeroNetwork:
    """Rseau AlphaZero (placeholder)"""
    
    def __init__(self):
        self.model = None
    
    def predict(self, state):
        return np.random.randn(16), 0.0
    
    def train_on_batch(self, states, policies, values):
        return 0.0
    
    def save_weights(self, path):
        pass
    
    def load_weights(self, path):
        pass

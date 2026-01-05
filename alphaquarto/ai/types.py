# -*- coding: utf-8 -*-
"""Types pour IA - MCTS Nodes et structures de données"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np


@dataclass
class MCTSNode:
    """
    Noeud de l'arbre MCTS pour AlphaZero.

    Dans Quarto, chaque noeud représente un état du jeu après:
    - Un placement de pièce (move)
    - Un choix de pièce pour l'adversaire (piece_choice)

    Attributes:
        visit_count: Nombre de fois que ce noeud a été visité
        value_sum: Somme des valeurs retournées par les simulations
        prior: Probabilité a priori (du réseau de neurones ou uniforme)
        children: Dictionnaire action -> noeud enfant
        parent: Noeud parent (None pour la racine)
        action: Action qui a mené à ce noeud
        is_expanded: True si le noeud a été développé
    """
    visit_count: int = 0
    value_sum: float = 0.0
    prior: float = 0.0
    children: Dict[int, 'MCTSNode'] = field(default_factory=dict)
    parent: Optional['MCTSNode'] = None
    action: Optional[int] = None
    is_expanded: bool = False

    def get_value(self) -> float:
        """Retourne la valeur moyenne du noeud (Q-value)."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, c_puct: float = 1.41) -> float:
        """
        Calcule le score UCB (Upper Confidence Bound) pour la sélection.

        Formule AlphaZero: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

        Args:
            c_puct: Constante d'exploration (défaut: sqrt(2) ≈ 1.41)

        Returns:
            Score UCB pour ce noeud
        """
        if self.parent is None:
            return 0.0

        # Terme d'exploration: encourage les noeuds peu visités
        exploration = c_puct * self.prior * np.sqrt(self.parent.visit_count) / (1 + self.visit_count)

        # Q-value (exploitation) + exploration
        return self.get_value() + exploration

    def select_child(self, c_puct: float = 1.41) -> 'MCTSNode':
        """
        Sélectionne le meilleur enfant selon UCB.

        Args:
            c_puct: Constante d'exploration

        Returns:
            Noeud enfant avec le meilleur score UCB
        """
        best_score = float('-inf')
        best_child = None

        for child in self.children.values():
            score = child.ucb_score(c_puct)
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def expand(self, actions: list, priors: np.ndarray) -> None:
        """
        Développe le noeud en créant des enfants pour chaque action légale.

        Args:
            actions: Liste des actions légales
            priors: Probabilités a priori pour chaque action
        """
        self.is_expanded = True

        for action in actions:
            if action not in self.children:
                child = MCTSNode(
                    prior=priors[action] if action < len(priors) else 1.0 / len(actions),
                    parent=self,
                    action=action
                )
                self.children[action] = child

    def backpropagate(self, value: float) -> None:
        """
        Propage la valeur vers la racine.

        Args:
            value: Valeur à propager (1 = victoire, -1 = défaite, 0 = nul)
        """
        node = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            # Alterner le signe car les joueurs alternent
            value = -value
            node = node.parent

    def get_visit_distribution(self, temperature: float = 1.0) -> np.ndarray:
        """
        Retourne la distribution des visites pour la sélection d'action.

        Args:
            temperature: Contrôle l'exploration (0 = glouton, 1 = proportionnel)

        Returns:
            Distribution de probabilité sur les actions
        """
        if not self.children:
            return np.array([])

        visits = np.array([child.visit_count for child in self.children.values()])
        actions = list(self.children.keys())

        if temperature == 0:
            # Mode glouton: choisir l'action la plus visitée
            probs = np.zeros(len(visits))
            probs[np.argmax(visits)] = 1.0
        else:
            # Appliquer la température
            visits_temp = np.power(visits.astype(float), 1.0 / temperature)
            probs = visits_temp / visits_temp.sum()

        return actions, probs

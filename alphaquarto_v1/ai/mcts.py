# -*- coding: utf-8 -*-
"""
MCTS (Monte Carlo Tree Search) pour AlphaQuarto.

Implémentation suivant l'approche AlphaZero:
- Utilise un réseau de neurones pour l'évaluation et les priors
- Pas de rollouts aléatoires (contrairement au MCTS classique)
- Exploration guidée par UCB (Upper Confidence Bound)
- Détection intelligente des pièces dangereuses (qui permettent un Quarto)
"""

import numpy as np
from typing import Optional, Tuple, Callable, List
from alphaquarto_v1.ai.types import MCTSNode
from alphaquarto_v1.game.constants import NUM_SQUARES, NUM_PIECES


class MCTS:
    """
    Monte Carlo Tree Search pour Quarto.

    Gère la recherche en deux phases:
    1. Sélection de case (où placer la pièce courante)
    2. Sélection de pièce (quelle pièce donner à l'adversaire)

    Attributes:
        num_simulations: Nombre de simulations par recherche
        c_puct: Constante d'exploration pour UCB
        network: Réseau de neurones pour évaluation (optionnel)
        use_dirichlet: Ajouter du bruit Dirichlet à la racine
        dirichlet_alpha: Paramètre alpha pour le bruit Dirichlet
        dirichlet_epsilon: Poids du bruit Dirichlet
    """

    def __init__(
        self,
        num_simulations: int = 100,
        c_puct: float = 1.41,
        network: Optional[Callable] = None,
        use_dirichlet: bool = True,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
    ):
        """
        Initialise le MCTS.

        Args:
            num_simulations: Nombre de simulations par recherche
            c_puct: Constante d'exploration pour UCB
            network: Fonction (state) -> (policy, value) ou None pour random
            use_dirichlet: Ajouter du bruit à la racine pour exploration
            dirichlet_alpha: Paramètre du bruit Dirichlet
            dirichlet_epsilon: Poids du bruit (0 = pas de bruit, 1 = tout bruit)
        """
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.network = network
        self.use_dirichlet = use_dirichlet
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    # =========================================================================
    # Méthodes de détection de coups gagnants / pièces dangereuses
    # =========================================================================

    def _find_winning_move(self, game) -> Optional[int]:
        """
        Trouve un coup gagnant immédiat s'il existe.

        Args:
            game: Instance du jeu avec une pièce en main

        Returns:
            Index de la case gagnante, ou None si aucun coup gagnant
        """
        if game.current_piece is None:
            return None

        legal_moves = game.get_legal_moves()
        for move in legal_moves:
            # Simuler le coup
            test_game = game.clone()
            test_game.play_move(move)
            if test_game.game_over and test_game.winner is not None:
                return move
        return None

    def _piece_enables_quarto(self, game, piece_id: int) -> bool:
        """
        Vérifie si donner cette pièce à l'adversaire lui permet de gagner.

        Args:
            game: Instance du jeu (après notre coup)
            piece_id: ID de la pièce à évaluer (1-16)

        Returns:
            True si l'adversaire peut faire Quarto avec cette pièce
        """
        test_game = game.clone()
        test_game.choose_piece(piece_id)

        # Vérifier si l'adversaire a un coup gagnant
        return self._find_winning_move(test_game) is not None

    def _get_safe_pieces(self, game) -> Tuple[List[int], List[int]]:
        """
        Sépare les pièces en pièces sûres et dangereuses.

        Args:
            game: Instance du jeu

        Returns:
            Tuple (safe_pieces, dangerous_pieces) - listes d'IDs (1-16)
        """
        available = game.get_available_pieces()
        safe_pieces = []
        dangerous_pieces = []

        for piece_id in available:
            if self._piece_enables_quarto(game, piece_id):
                dangerous_pieces.append(piece_id)
            else:
                safe_pieces.append(piece_id)

        return safe_pieces, dangerous_pieces

    # =========================================================================
    # Recherche MCTS principale
    # =========================================================================

    def search(self, game, temperature: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Effectue une recherche MCTS pour sélectionner une case.

        Vérifie d'abord s'il existe un coup gagnant immédiat.

        Args:
            game: Instance du jeu Quarto avec une pièce en main
            temperature: Contrôle l'exploration (0 = glouton, 1 = proportionnel)

        Returns:
            Tuple (move_probs, piece_probs):
                - move_probs: Distribution sur les 16 cases
                - piece_probs: Distribution sur les 16 pièces (pour après le coup)
        """
        if game.current_piece is None:
            raise ValueError("Aucune pièce en main - utilisez search_piece d'abord")

        # Vérifier s'il y a un coup gagnant immédiat
        winning_move = self._find_winning_move(game)
        if winning_move is not None:
            move_probs = np.zeros(NUM_SQUARES)
            move_probs[winning_move] = 1.0
            return move_probs, np.zeros(NUM_PIECES)

        root = MCTSNode()
        legal_moves = game.get_legal_moves()

        if not legal_moves:
            return np.zeros(NUM_SQUARES), np.zeros(NUM_PIECES)

        # Évaluer l'état racine et obtenir les priors
        move_priors, value = self._evaluate(game)
        move_priors = self._mask_illegal_moves(move_priors, legal_moves)

        # Ajouter du bruit Dirichlet à la racine pour l'exploration
        if self.use_dirichlet and len(legal_moves) > 0:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_moves))
            noise_full = np.zeros(NUM_SQUARES)
            for i, move in enumerate(legal_moves):
                noise_full[move] = noise[i]
            move_priors = (1 - self.dirichlet_epsilon) * move_priors + self.dirichlet_epsilon * noise_full

        # Développer la racine
        root.expand(legal_moves, move_priors)

        # Effectuer les simulations
        for _ in range(self.num_simulations):
            self._simulate(game.clone(), root)

        # Extraire la distribution des visites
        move_probs = np.zeros(NUM_SQUARES)
        for action, child in root.children.items():
            move_probs[action] = child.visit_count

        # Appliquer la température
        if temperature == 0:
            best_move = max(root.children.keys(), key=lambda a: root.children[a].visit_count)
            move_probs = np.zeros(NUM_SQUARES)
            move_probs[best_move] = 1.0
        elif move_probs.sum() > 0:
            move_probs = np.power(move_probs, 1.0 / temperature)
            move_probs /= move_probs.sum()

        piece_probs = self._get_piece_probs(game, root)

        return move_probs, piece_probs

    def search_piece(self, game, temperature: float = 1.0) -> np.ndarray:
        """
        Recherche MCTS pour sélectionner une pièce à donner à l'adversaire.

        Stratégie:
        1. D'abord, identifier les pièces sûres (qui ne permettent pas un Quarto)
        2. Si des pièces sûres existent, choisir parmi elles
        3. Sinon, toutes les pièces sont dangereuses - l'adversaire gagnera

        Args:
            game: Instance du jeu Quarto (après avoir placé une pièce)
            temperature: Contrôle l'exploration

        Returns:
            Distribution de probabilité sur les 16 pièces
        """
        available_pieces = game.get_available_pieces()

        if not available_pieces:
            return np.zeros(NUM_PIECES)

        # Identifier les pièces sûres vs dangereuses
        safe_pieces, dangerous_pieces = self._get_safe_pieces(game)

        # Si des pièces sûres existent, privilégier fortement ces pièces
        if safe_pieces:
            # Créer des priors qui favorisent les pièces sûres
            piece_priors = np.zeros(NUM_PIECES)
            for piece in safe_pieces:
                piece_priors[piece - 1] = 1.0
            for piece in dangerous_pieces:
                piece_priors[piece - 1] = 0.01  # Très faible mais non nul
            piece_priors /= piece_priors.sum()
        else:
            # Toutes les pièces sont dangereuses - distribution uniforme
            piece_priors = np.zeros(NUM_PIECES)
            for piece in available_pieces:
                piece_priors[piece - 1] = 1.0 / len(available_pieces)

        root = MCTSNode()

        # Ajouter du bruit Dirichlet (seulement sur les pièces sûres si possible)
        if self.use_dirichlet:
            target_pieces = safe_pieces if safe_pieces else available_pieces
            if len(target_pieces) > 0:
                noise = np.random.dirichlet([self.dirichlet_alpha] * len(target_pieces))
                noise_full = np.zeros(NUM_PIECES)
                for i, piece in enumerate(target_pieces):
                    noise_full[piece - 1] = noise[i]
                # Normaliser le bruit sur toutes les pièces disponibles
                noise_full = self._mask_illegal_pieces(noise_full, available_pieces)
                piece_priors = (1 - self.dirichlet_epsilon) * piece_priors + self.dirichlet_epsilon * noise_full

        # Développer la racine
        root.expand([p - 1 for p in available_pieces], piece_priors)

        # Simulations
        for _ in range(self.num_simulations):
            self._simulate_piece_selection(game.clone(), root, available_pieces)

        # Extraire la distribution
        piece_probs = np.zeros(NUM_PIECES)
        for action, child in root.children.items():
            piece_probs[action] = child.visit_count

        # Appliquer la température
        if temperature == 0 and piece_probs.sum() > 0:
            best_piece = max(root.children.keys(), key=lambda a: root.children[a].visit_count)
            piece_probs = np.zeros(NUM_PIECES)
            piece_probs[best_piece] = 1.0
        elif piece_probs.sum() > 0:
            piece_probs = np.power(piece_probs, 1.0 / temperature)
            piece_probs /= piece_probs.sum()

        return piece_probs

    # =========================================================================
    # Simulations MCTS
    # =========================================================================

    def _simulate(self, game, node: MCTSNode) -> float:
        """
        Effectue une simulation depuis le noeud donné.

        L'adversaire joue de manière intelligente:
        - S'il a un coup gagnant, il le joue
        - Sinon, il joue aléatoirement

        Args:
            game: Clone du jeu pour simulation
            node: Noeud de départ

        Returns:
            Valeur de la simulation
        """
        path = [node]

        # Phase 1: Sélection - descendre l'arbre
        while node.is_expanded and node.children:
            node = node.select_child(self.c_puct)
            path.append(node)

            if node.action is not None:
                game.play_move(node.action)

                if not game.game_over:
                    available = game.get_available_pieces()
                    if available:
                        # Choisir une pièce intelligemment (éviter de donner un Quarto)
                        piece = self._choose_piece_for_simulation(game, available)
                        game.choose_piece(piece)

        # Phase 2 & 3: Expansion et Évaluation
        if game.game_over:
            if game.winner is not None:
                value = 1.0 if game.winner == game.current_player else -1.0
            else:
                value = 0.0
        else:
            legal_moves = game.get_legal_moves()
            if legal_moves and game.current_piece is not None:
                priors, value = self._evaluate(game)
                priors = self._mask_illegal_moves(priors, legal_moves)
                node.expand(legal_moves, priors)
            else:
                value = 0.0

        # Phase 4: Backpropagation
        for n in reversed(path):
            n.visit_count += 1
            n.value_sum += value
            value = -value

        return value

    def _simulate_piece_selection(self, game, node: MCTSNode, available_pieces: list) -> float:
        """
        Simulation pour la sélection de pièce.

        IMPORTANT: L'adversaire joue de manière optimale - s'il a un coup
        gagnant, il le joue systématiquement.

        Args:
            game: Clone du jeu
            node: Noeud racine
            available_pieces: Pièces disponibles

        Returns:
            Valeur de la simulation
        """
        path = [node]

        # Sélection
        while node.is_expanded and node.children:
            node = node.select_child(self.c_puct)
            path.append(node)

            if node.action is not None:
                piece_id = node.action + 1
                game.choose_piece(piece_id)

                if not game.game_over:
                    # L'adversaire joue - VÉRIFIER S'IL A UN COUP GAGNANT
                    winning_move = self._find_winning_move(game)
                    if winning_move is not None:
                        # L'adversaire joue le coup gagnant!
                        game.play_move(winning_move)
                    else:
                        # Pas de coup gagnant - jouer aléatoirement
                        legal_moves = game.get_legal_moves()
                        if legal_moves:
                            move = np.random.choice(legal_moves)
                            game.play_move(move)

                    # Après le coup de l'adversaire, choisir une pièce pour nous
                    if not game.game_over:
                        remaining = game.get_available_pieces()
                        if remaining:
                            piece = self._choose_piece_for_simulation(game, remaining)
                            game.choose_piece(piece)

        # Évaluation
        if game.game_over:
            if game.winner is not None:
                # Du point de vue de celui qui a choisi la pièce:
                # Si l'adversaire gagne, c'est mauvais pour nous
                value = -1.0
            else:
                value = 0.0
        else:
            value = self._smart_rollout(game)

        # Backpropagation
        for n in reversed(path):
            n.visit_count += 1
            n.value_sum += value
            value = -value

        return value

    def _choose_piece_for_simulation(self, game, available_pieces: list) -> int:
        """
        Choisit une pièce pour la simulation, en évitant les pièces dangereuses si possible.

        Args:
            game: État actuel du jeu
            available_pieces: Pièces disponibles

        Returns:
            ID de la pièce choisie
        """
        safe_pieces, dangerous_pieces = self._get_safe_pieces(game)

        if safe_pieces:
            return np.random.choice(safe_pieces)
        else:
            # Toutes les pièces sont dangereuses
            return np.random.choice(available_pieces)

    # =========================================================================
    # Évaluation et Rollouts
    # =========================================================================

    def _evaluate(self, game) -> Tuple[np.ndarray, float]:
        """
        Évalue un état de jeu.

        Args:
            game: Instance du jeu

        Returns:
            Tuple (policy, value) où policy est sur les cases
        """
        if self.network is not None:
            state = game.get_state()
            policy, value = self.network(state)
            return policy, value
        else:
            policy = np.ones(NUM_SQUARES) / NUM_SQUARES
            value = self._smart_rollout(game.clone())
            return policy, value

    def _evaluate_pieces(self, game) -> np.ndarray:
        """
        Obtient les priors pour la sélection de pièce.

        Pénalise les pièces qui permettent un Quarto à l'adversaire.

        Args:
            game: Instance du jeu

        Returns:
            Distribution sur les 16 pièces
        """
        available = game.get_available_pieces()
        if not available:
            return np.zeros(NUM_PIECES)

        safe_pieces, dangerous_pieces = self._get_safe_pieces(game)

        priors = np.zeros(NUM_PIECES)
        if safe_pieces:
            # Pièces sûres ont un prior élevé
            for piece in safe_pieces:
                priors[piece - 1] = 1.0
            # Pièces dangereuses ont un prior très faible
            for piece in dangerous_pieces:
                priors[piece - 1] = 0.01
        else:
            # Toutes dangereuses - uniforme
            for piece in available:
                priors[piece - 1] = 1.0

        if priors.sum() > 0:
            priors /= priors.sum()

        return priors

    def _smart_rollout(self, game, max_moves: int = 50) -> float:
        """
        Effectue un rollout intelligent jusqu'à la fin du jeu.

        Les joueurs:
        - Jouent les coups gagnants s'ils existent
        - Évitent de donner des pièces gagnantes si possible
        - Sinon jouent aléatoirement

        Args:
            game: Clone du jeu
            max_moves: Nombre maximum de coups

        Returns:
            Valeur: 1 si victoire joueur actuel, -1 si défaite, 0 si nul
        """
        original_player = game.current_player

        for _ in range(max_moves):
            if game.game_over:
                break

            # Choisir une pièce si nécessaire
            if game.current_piece is None:
                available = game.get_available_pieces()
                if available:
                    piece = self._choose_piece_for_simulation(game, available)
                    game.choose_piece(piece)
                else:
                    break

            # Vérifier s'il y a un coup gagnant
            winning_move = self._find_winning_move(game)
            if winning_move is not None:
                game.play_move(winning_move)
            else:
                # Jouer aléatoirement
                legal_moves = game.get_legal_moves()
                if legal_moves:
                    move = np.random.choice(legal_moves)
                    game.play_move(move)
                else:
                    break

        if game.game_over:
            if game.winner is None:
                return 0.0
            elif game.winner == original_player:
                return 1.0
            else:
                return -1.0
        else:
            return 0.0

    # =========================================================================
    # Méthodes utilitaires
    # =========================================================================

    def _mask_illegal_moves(self, priors: np.ndarray, legal_moves: list) -> np.ndarray:
        """Masque les coups illégaux et renormalise."""
        mask = np.zeros(NUM_SQUARES)
        for move in legal_moves:
            mask[move] = 1.0

        masked = priors * mask
        total = masked.sum()
        if total > 0:
            return masked / total
        else:
            return mask / mask.sum() if mask.sum() > 0 else mask

    def _mask_illegal_pieces(self, priors: np.ndarray, available_pieces: list) -> np.ndarray:
        """Masque les pièces indisponibles et renormalise."""
        mask = np.zeros(NUM_PIECES)
        for piece in available_pieces:
            mask[piece - 1] = 1.0

        masked = priors * mask
        total = masked.sum()
        if total > 0:
            return masked / total
        else:
            return mask / mask.sum() if mask.sum() > 0 else np.zeros(NUM_PIECES)

    def _get_piece_probs(self, game, root: MCTSNode) -> np.ndarray:
        """Estime les probabilités de choix de pièce."""
        available = game.get_available_pieces()
        if not available:
            return np.zeros(NUM_PIECES)

        # Utiliser l'évaluation intelligente des pièces
        return self._evaluate_pieces(game)

    # =========================================================================
    # Interface publique simplifiée
    # =========================================================================

    def get_best_move(self, game) -> int:
        """
        Retourne le meilleur coup.

        Vérifie d'abord s'il y a un coup gagnant immédiat.

        Args:
            game: Instance du jeu avec pièce en main

        Returns:
            Index de la case (0-15)
        """
        # Vérifier coup gagnant immédiat
        winning_move = self._find_winning_move(game)
        if winning_move is not None:
            return winning_move

        move_probs, _ = self.search(game, temperature=0)
        return int(np.argmax(move_probs))

    def get_best_piece(self, game) -> int:
        """
        Retourne la meilleure pièce à donner.

        Évite les pièces qui permettent un Quarto à l'adversaire.

        Args:
            game: Instance du jeu après avoir placé une pièce

        Returns:
            ID de la pièce (1-16)
        """
        available = game.get_available_pieces()
        if not available:
            raise ValueError("Aucune pièce disponible")

        # D'abord, vérifier s'il y a des pièces sûres
        safe_pieces, dangerous_pieces = self._get_safe_pieces(game)

        if safe_pieces:
            # Si peu de pièces sûres, faire une recherche MCTS parmi elles
            if len(safe_pieces) <= 3:
                # Recherche complète pour choisir la meilleure pièce sûre
                piece_probs = self.search_piece(game, temperature=0)
                # Ne considérer que les pièces sûres
                for i in range(NUM_PIECES):
                    if (i + 1) not in safe_pieces:
                        piece_probs[i] = 0
                if piece_probs.sum() > 0:
                    return int(np.argmax(piece_probs)) + 1
                else:
                    return np.random.choice(safe_pieces)
            else:
                piece_probs = self.search_piece(game, temperature=0)
                return int(np.argmax(piece_probs)) + 1
        else:
            # Toutes les pièces sont dangereuses - l'adversaire gagnera
            # Retourner n'importe quelle pièce
            return np.random.choice(available)

# -*- coding: utf-8 -*-
"""Classe Quarto - Moteur du jeu"""

import numpy as np
from alphaquarto.game.constants import (
    BOARD_SIZE, NUM_SQUARES, NUM_PIECES, NUM_PROPERTIES,
    COLOR_BIT, SHAPE_BIT, SIZE_BIT, HOLE_BIT
)


def get_piece_properties(piece_id: int) -> tuple[int, int, int, int]:
    """
    Extrait les 4 propriétés binaires d'une pièce.

    Les pièces sont numérotées 1-16, encodant 4 bits (0-15):
    - Bit 0: Couleur (0=clair, 1=foncé)
    - Bit 1: Forme (0=rond, 1=carré)
    - Bit 2: Taille (0=petit, 1=grand)
    - Bit 3: Trou (0=plein, 1=troué)

    Args:
        piece_id: ID de la pièce (1-16)

    Returns:
        Tuple (couleur, forme, taille, trou) - chaque valeur est 0 ou 1
    """
    if piece_id < 1 or piece_id > NUM_PIECES:
        raise ValueError(f"piece_id doit être entre 1 et {NUM_PIECES}")

    bits = piece_id - 1  # Convertir en 0-15
    return (
        (bits >> COLOR_BIT) & 1,
        (bits >> SHAPE_BIT) & 1,
        (bits >> SIZE_BIT) & 1,
        (bits >> HOLE_BIT) & 1,
    )


def get_property(piece_id: int, property_bit: int) -> int:
    """
    Extrait une propriété spécifique d'une pièce.

    Args:
        piece_id: ID de la pièce (1-16)
        property_bit: Index du bit (COLOR_BIT, SHAPE_BIT, SIZE_BIT, HOLE_BIT)

    Returns:
        0 ou 1
    """
    if piece_id < 1 or piece_id > NUM_PIECES:
        raise ValueError(f"piece_id doit être entre 1 et {NUM_PIECES}")
    return ((piece_id - 1) >> property_bit) & 1


def check_line_quarto(pieces: list[int]) -> bool:
    """
    Vérifie si 4 pièces forment un Quarto (propriété commune).

    Un Quarto est formé quand 4 pièces partagent AU MOINS une propriété:
    toutes claires OU toutes foncées, toutes rondes OU toutes carrées, etc.

    Args:
        pieces: Liste de 4 IDs de pièces (1-16)

    Returns:
        True si les 4 pièces forment un Quarto
    """
    if len(pieces) != 4:
        return False

    # Vérifier que toutes les pièces sont valides (non nulles)
    if any(p < 1 or p > NUM_PIECES for p in pieces):
        return False

    # Pour chaque propriété, vérifier si toutes les pièces ont la même valeur
    for prop_bit in range(NUM_PROPERTIES):
        values = [get_property(p, prop_bit) for p in pieces]
        # Toutes à 0 ou toutes à 1 = Quarto sur cette propriété
        if all(v == 0 for v in values) or all(v == 1 for v in values):
            return True

    return False

class Quarto:
    """Jeu Quarto complet"""
    
    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.available_pieces = set(range(1, NUM_PIECES + 1))
        self.current_piece = None
        self.current_player = 0  # Joueur 0 commence
        self.game_over = False
        self.winner = None
    
    def get_legal_moves(self):
        """Retourne les coups legaux"""
        return [i for i in range(NUM_SQUARES) if self.board.flat[i] == 0]
    
    def get_available_pieces(self):
        """Retourne les pieces disponibles"""
        return sorted(list(self.available_pieces))

    def check_quarto(self) -> bool:
        """
        Vérifie si un Quarto a été formé sur le plateau.

        Vérifie les 10 lignes possibles:
        - 4 lignes horizontales
        - 4 lignes verticales
        - 2 diagonales

        Returns:
            True si un Quarto est détecté
        """
        # Lignes horizontales
        for row in range(BOARD_SIZE):
            pieces = [self.board[row, col] for col in range(BOARD_SIZE)]
            if all(p > 0 for p in pieces) and check_line_quarto(pieces):
                return True

        # Lignes verticales
        for col in range(BOARD_SIZE):
            pieces = [self.board[row, col] for row in range(BOARD_SIZE)]
            if all(p > 0 for p in pieces) and check_line_quarto(pieces):
                return True

        # Diagonale principale (haut-gauche vers bas-droite)
        pieces = [self.board[i, i] for i in range(BOARD_SIZE)]
        if all(p > 0 for p in pieces) and check_line_quarto(pieces):
            return True

        # Diagonale secondaire (haut-droite vers bas-gauche)
        pieces = [self.board[i, BOARD_SIZE - 1 - i] for i in range(BOARD_SIZE)]
        if all(p > 0 for p in pieces) and check_line_quarto(pieces):
            return True

        return False
    
    def play_move(self, square):
        """
        Joue un coup: place la pièce courante sur la case spécifiée.

        Args:
            square: Index de la case (0-15)

        Returns:
            True si le coup a été joué, False sinon
        """
        if self.game_over or square < 0 or square >= NUM_SQUARES:
            return False
        if self.board.flat[square] != 0 or self.current_piece is None:
            return False

        # Placer la pièce (déjà retirée de available_pieces dans choose_piece)
        self.board.flat[square] = self.current_piece
        self.current_piece = None

        # Vérifier si un Quarto a été formé
        if self.check_quarto():
            self.game_over = True
            self.winner = self.current_player
        # Sinon, vérifier si le plateau est plein (match nul)
        elif not self.get_legal_moves():
            self.game_over = True
            self.winner = None  # Match nul
        else:
            # Alterner le joueur pour le prochain tour
            self.current_player = 1 - self.current_player

        return True
    
    def choose_piece(self, piece):
        """
        Choisit une pièce pour l'adversaire.

        La pièce est immédiatement retirée des pièces disponibles
        car elle est maintenant assignée.

        Args:
            piece: ID de la pièce (1-16)

        Returns:
            True si la pièce a été choisie, False sinon
        """
        if piece not in self.available_pieces:
            return False
        self.current_piece = piece
        self.available_pieces.discard(piece)
        return True
    
    def clone(self):
        """Clone le jeu"""
        game = Quarto()
        game.board = self.board.copy()
        game.available_pieces = self.available_pieces.copy()
        game.current_piece = self.current_piece
        game.current_player = self.current_player
        game.game_over = self.game_over
        game.winner = self.winner
        return game
    
    def get_state(self):
        """Retourne l'etat en one-hot"""
        state = np.zeros((BOARD_SIZE, BOARD_SIZE, NUM_PIECES), dtype=np.float32)
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                piece = self.board[i, j]
                if piece > 0:
                    state[i, j, piece - 1] = 1.0
        return state

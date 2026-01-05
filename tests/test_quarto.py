"""Tests Quarto"""

import pytest
import numpy as np
from alphaquarto_v1.game import Quarto
from alphaquarto_v1.game.quarto import (
    get_piece_properties,
    get_property,
    check_line_quarto,
)
from alphaquarto_v1.game.constants import COLOR_BIT, SHAPE_BIT, SIZE_BIT, HOLE_BIT


# =============================================================================
# Tests d'initialisation
# =============================================================================

def test_quarto_init():
    game = Quarto()
    assert game.current_piece is None
    assert len(game.available_pieces) == 16
    assert game.current_player == 0
    assert game.game_over is False
    assert game.winner is None


def test_quarto_legal_moves():
    game = Quarto()
    moves = game.get_legal_moves()
    assert len(moves) == 16


# =============================================================================
# Tests des propriétés des pièces
# =============================================================================

def test_get_piece_properties_piece_1():
    """Pièce 1 = bits 0000 = toutes propriétés à 0"""
    props = get_piece_properties(1)
    assert props == (0, 0, 0, 0)


def test_get_piece_properties_piece_16():
    """Pièce 16 = bits 1111 = toutes propriétés à 1"""
    props = get_piece_properties(16)
    assert props == (1, 1, 1, 1)


def test_get_piece_properties_piece_5():
    """Pièce 5 = bits 0100 = forme=1, reste=0"""
    props = get_piece_properties(5)
    # bits = 4 = 0100
    # COLOR_BIT=0: (4 >> 0) & 1 = 0
    # SHAPE_BIT=1: (4 >> 1) & 1 = 0
    # SIZE_BIT=2: (4 >> 2) & 1 = 1
    # HOLE_BIT=3: (4 >> 3) & 1 = 0
    assert props == (0, 0, 1, 0)


def test_get_piece_properties_all_pieces():
    """Vérifie que toutes les pièces ont des propriétés valides"""
    for piece_id in range(1, 17):
        props = get_piece_properties(piece_id)
        assert len(props) == 4
        assert all(p in (0, 1) for p in props)


def test_get_piece_properties_invalid():
    """Pièce invalide doit lever une exception"""
    with pytest.raises(ValueError):
        get_piece_properties(0)
    with pytest.raises(ValueError):
        get_piece_properties(17)


def test_get_property():
    """Test extraction d'une propriété individuelle"""
    # Pièce 9 = bits 1000 = HOLE=1, reste=0
    assert get_property(9, COLOR_BIT) == 0
    assert get_property(9, SHAPE_BIT) == 0
    assert get_property(9, SIZE_BIT) == 0
    assert get_property(9, HOLE_BIT) == 1


# =============================================================================
# Tests de check_line_quarto
# =============================================================================

def test_check_line_quarto_same_color():
    """4 pièces de même couleur (bit 0 identique)"""
    # Pièces 1, 3, 5, 7 ont toutes COLOR_BIT = 0
    # 1-1=0=0000, 3-1=2=0010, 5-1=4=0100, 7-1=6=0110
    pieces = [1, 3, 5, 7]
    assert check_line_quarto(pieces) is True


def test_check_line_quarto_same_size():
    """4 pièces de même taille (bit 2 identique)"""
    # Pièces 5, 6, 7, 8 ont toutes SIZE_BIT = 1
    # 5-1=4=0100, 6-1=5=0101, 7-1=6=0110, 8-1=7=0111
    pieces = [5, 6, 7, 8]
    assert check_line_quarto(pieces) is True


def test_check_line_quarto_no_common_property():
    """4 pièces sans propriété commune"""
    # Construction de 4 pièces qui n'ont aucune propriété commune
    # 1=0000, 2=0001, 4=0011, 8=0111 - vérifier manuellement
    # En fait, trouvons un vrai exemple:
    # 1=0000, 6=0101, 11=1010, 16=1111
    # COLOR: 0,1,0,1 - pas commun
    # SHAPE: 0,0,1,1 - pas commun
    # SIZE: 0,1,0,1 - pas commun
    # HOLE: 0,0,1,1 - pas commun
    pieces = [1, 6, 11, 16]
    assert check_line_quarto(pieces) is False


def test_check_line_quarto_all_same():
    """4 pièces identiques (impossible en vrai jeu mais bon test)"""
    # En pratique c'est impossible, mais la fonction doit retourner True
    pieces = [1, 1, 1, 1]
    assert check_line_quarto(pieces) is True


def test_check_line_quarto_incomplete_line():
    """Moins de 4 pièces"""
    assert check_line_quarto([1, 2, 3]) is False
    assert check_line_quarto([]) is False


def test_check_line_quarto_with_zero():
    """Ligne avec case vide (0)"""
    pieces = [1, 2, 0, 4]
    assert check_line_quarto(pieces) is False


# =============================================================================
# Tests de détection de Quarto sur le plateau
# =============================================================================

def test_check_quarto_horizontal():
    """Quarto sur une ligne horizontale"""
    game = Quarto()
    # Placer pièces 1, 3, 5, 7 sur la première ligne (même couleur)
    game.board[0, 0] = 1
    game.board[0, 1] = 3
    game.board[0, 2] = 5
    game.board[0, 3] = 7
    assert game.check_quarto() is True


def test_check_quarto_vertical():
    """Quarto sur une colonne verticale"""
    game = Quarto()
    # Placer pièces 1, 3, 5, 7 sur la première colonne
    game.board[0, 0] = 1
    game.board[1, 0] = 3
    game.board[2, 0] = 5
    game.board[3, 0] = 7
    assert game.check_quarto() is True


def test_check_quarto_diagonal_main():
    """Quarto sur la diagonale principale"""
    game = Quarto()
    # Placer pièces 1, 3, 5, 7 sur la diagonale
    game.board[0, 0] = 1
    game.board[1, 1] = 3
    game.board[2, 2] = 5
    game.board[3, 3] = 7
    assert game.check_quarto() is True


def test_check_quarto_diagonal_anti():
    """Quarto sur la diagonale secondaire"""
    game = Quarto()
    # Placer pièces 1, 3, 5, 7 sur l'anti-diagonale
    game.board[0, 3] = 1
    game.board[1, 2] = 3
    game.board[2, 1] = 5
    game.board[3, 0] = 7
    assert game.check_quarto() is True


def test_check_quarto_no_quarto():
    """Pas de Quarto formé"""
    game = Quarto()
    # Placer des pièces sans former de Quarto
    game.board[0, 0] = 1
    game.board[0, 1] = 6
    game.board[0, 2] = 11
    game.board[0, 3] = 16
    assert game.check_quarto() is False


def test_check_quarto_incomplete_line():
    """Ligne incomplète (3 pièces seulement)"""
    game = Quarto()
    game.board[0, 0] = 1
    game.board[0, 1] = 3
    game.board[0, 2] = 5
    # game.board[0, 3] reste à 0
    assert game.check_quarto() is False


# =============================================================================
# Tests d'intégration play_move + détection
# =============================================================================

def test_play_move_triggers_quarto_detection():
    """play_move détecte un Quarto et termine la partie"""
    game = Quarto()

    # Préparer le plateau avec 3 pièces alignées (même couleur)
    game.board[0, 0] = 1
    game.board[0, 1] = 3
    game.board[0, 2] = 5
    game.available_pieces -= {1, 3, 5}

    # Le joueur 0 va placer la 4ème pièce
    game.current_player = 0
    game.choose_piece(7)
    result = game.play_move(3)  # Case 3 = (0, 3)

    assert result is True
    assert game.game_over is True
    assert game.winner == 0


def test_play_move_no_quarto_alternates_player():
    """play_move alterne le joueur si pas de Quarto"""
    game = Quarto()
    assert game.current_player == 0

    game.choose_piece(1)
    game.play_move(0)
    assert game.current_player == 1

    game.choose_piece(2)
    game.play_move(1)
    assert game.current_player == 0


def test_game_over_when_no_moves_left():
    """Test que game_over est True quand il n'y a plus de coups possibles"""
    game = Quarto()

    # Remplir tout le plateau sauf une case
    for i in range(15):
        game.board.flat[i] = i + 1
        game.available_pieces.discard(i + 1)

    # Il reste la case 15 vide et la pièce 16
    assert len(game.get_legal_moves()) == 1
    assert 16 in game.available_pieces

    # Jouer le dernier coup
    game.choose_piece(16)
    result = game.play_move(15)

    assert result is True
    assert game.game_over is True
    # winner est défini si Quarto, sinon None (match nul)


def test_winner_set_on_quarto():
    """Test que winner est correctement défini lors d'un Quarto"""
    game = Quarto()

    # Joueur 0 place des pièces sans former de Quarto
    game.choose_piece(1)
    game.play_move(0)
    assert game.winner is None

    # Joueur 1 continue
    game.choose_piece(3)
    game.play_move(1)
    assert game.winner is None

    # Joueur 0 continue
    game.choose_piece(5)
    game.play_move(2)
    assert game.winner is None

    # Joueur 1 va former un Quarto en plaçant la pièce 7 sur la case 3
    # Pièces 1, 3, 5, 7 ont toutes COLOR_BIT = 0
    game.choose_piece(7)
    game.play_move(3)

    assert game.game_over is True
    assert game.winner == 1  # Joueur 1 a gagné


def test_clone_preserves_current_player():
    """clone() copie bien current_player"""
    game = Quarto()
    game.current_player = 1

    cloned = game.clone()
    assert cloned.current_player == 1

    # Modifier l'original ne doit pas affecter le clone
    game.current_player = 0
    assert cloned.current_player == 1

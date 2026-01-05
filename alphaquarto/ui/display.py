# -*- coding: utf-8 -*-
"""Affichage du jeu Quarto"""

from alphaquarto.game.constants import BOARD_SIZE
from alphaquarto.game.quarto import get_piece_properties
from alphaquarto.ui.colors import Colors


def encode_piece_code(piece_id: int) -> str:
    """
    Encode une pièce en un code visuel de 4 caractères.

    Format: [forme_ouvrante][taille+couleur][trou][forme_fermante]

    Codes:
    - () = Forme ronde, [] = Forme carrée
    - W = Grande + Claire, B = Grande + Foncée
    - w = Petite + Claire, b = Petite + Foncée
    - * = Pleine, o = Creuse

    Exemples:
    - [W*] = Carrée, Grande, Claire, Pleine
    - (bo) = Ronde, Petite, Foncée, Creuse

    Args:
        piece_id: ID de la pièce (1-16)

    Returns:
        Code de 4 caractères représentant la pièce
    """
    color, shape, size, hole = get_piece_properties(piece_id)

    # Forme: () pour rond (shape=0), [] pour carré (shape=1)
    if shape == 0:
        open_char, close_char = "(", ")"
    else:
        open_char, close_char = "[", "]"

    # Taille + Couleur combinées
    if size == 1 and color == 0:
        size_color = "W"  # Grande + Claire
    elif size == 1 and color == 1:
        size_color = "B"  # Grande + Foncée
    elif size == 0 and color == 0:
        size_color = "w"  # Petite + Claire
    else:
        size_color = "b"  # Petite + Foncée

    # Trou: * pour pleine (hole=0), o pour creuse (hole=1)
    hole_char = "o" if hole == 1 else "*"

    return f"{open_char}{size_color}{hole_char}{close_char}"


def index_to_coord(index: int) -> str:
    """
    Convertit un index de case (0-15) en coordonnées (a1, b2, etc.).

    Args:
        index: Index de la case (0-15)

    Returns:
        Coordonnées sous forme de chaîne (ex: "a1", "d4")
    """
    row = index // BOARD_SIZE
    col = index % BOARD_SIZE
    col_labels = "abcd"
    return f"{col_labels[col]}{row + 1}"


class QuartoDisplay:
    """Gère l'affichage du jeu Quarto"""

    def __init__(self):
        self.last_placed_position: tuple[int, int] | None = None

    def render(self, game, show_available_pieces: bool = True, show_piece_in_hand: bool = True) -> None:
        """
        Affiche un rendu texte de l'état courant, avec :
        - Coordonnées de colonnes a..d en haut
        - Coordonnées de lignes 1..4 à gauche
        - Codes de pièces dans chaque case

        Args:
            game: Instance du jeu Quarto
            show_available_pieces: Afficher la liste des pièces disponibles
            show_piece_in_hand: Afficher la pièce en main
        """
        # Entête colonnes
        col_labels = ["a", "b", "c", "d"]
        header = "   " + "".join(f"{c:^6}" for c in col_labels)
        print(header)
        print("   " + "-" * (len(col_labels) * 6))

        for row in range(BOARD_SIZE):
            row_cells = []
            for col in range(BOARD_SIZE):
                piece = game.board[row, col]
                if piece == 0:
                    # Case vide
                    cell_str = " .... "
                else:
                    # Case occupée
                    code = encode_piece_code(piece)
                    # Dernière pièce placée en jaune
                    if self.last_placed_position == (row, col):
                        cell_str = f" {Colors.YELLOW}{code}{Colors.RESET} "
                    else:
                        cell_str = f" {code} "
                row_cells.append(cell_str)

            # Label de ligne (1..4) au début de chaque ligne
            line_label = f"{row + 1} |"
            print(line_label + "".join(row_cells))

        print()

        # Affiche la pièce en main (seulement si une pièce est en main)
        if show_piece_in_hand and game.current_piece is not None:
            code = encode_piece_code(game.current_piece)
            print(f"Piece en main : {game.current_piece} -> {Colors.LIGHT_BROWN}{code}{Colors.RESET}")

        # Affiche les pièces disponibles (seulement si une pièce est en main)
        if show_available_pieces and game.current_piece is not None:
            available = game.get_available_pieces()
            if available:
                pieces_str = " ".join(f"{p}:{encode_piece_code(p)}" for p in available)
                print(f"Pièces disponibles : {pieces_str}")
            else:
                print("Pièces disponibles : aucune")

    def render_board(self, game) -> str:
        """
        Retourne une représentation textuelle du plateau.
        (Méthode conservée pour compatibilité)
        """
        lines = []
        col_labels = ["a", "b", "c", "d"]
        lines.append("   " + "".join(f"{c:^6}" for c in col_labels))
        lines.append("   " + "-" * 24)

        for row in range(BOARD_SIZE):
            row_cells = []
            for col in range(BOARD_SIZE):
                piece = game.board[row, col]
                if piece == 0:
                    row_cells.append(" .... ")
                else:
                    code = encode_piece_code(piece)
                    row_cells.append(f" {code} ")
            lines.append(f"{row + 1} |" + "".join(row_cells))

        return "\n".join(lines)

    def set_last_placed(self, row: int, col: int) -> None:
        """Définit la dernière position jouée pour le highlighting"""
        self.last_placed_position = (row, col)

    def clear_last_placed(self) -> None:
        """Efface le highlighting de la dernière pièce"""
        self.last_placed_position = None

    def render_piece_legend(self) -> None:
        """Affiche la légende des codes de pièces"""
        print("\n=== légende des pièces ===")
        print("Forme   : () = Ronde    [] = Carrée")
        print("Taille  : Majuscule = Grande, Minuscule = Petite")
        print("Couleur : W/w = Claire   B/b = Foncée")
        print("Trou    : * = Pleine    o = Creuse")
        print()
        print("Exemples:")
        print("  [W*] = Carrée, Grande, Claire, Pleine")
        print("  (bo) = Ronde, Petite, Foncée, Creuse")
        print()

    def render_board_mapping(self, game) -> None:
        """
        Affiche un schéma indiquant comment les indices des cases libres
        correspondent aux cases du plateau 4x4, en coordonnées lettre+chiffre.
        Seules les cases libres sont affichées avec leur index.
        Les cases occupées sont représentées par des points.
        """
        print("\nCases libres:")
        col_labels = ["a", "b", "c", "d"]
        idx = 0
        for row in range(BOARD_SIZE):
            row_str = []
            for col in range(BOARD_SIZE):
                coord = f"{col_labels[col]}{row + 1}"  # ex: a1, b1, ...
                if game.board[row, col] == 0:
                    # Case libre: afficher index -> coordonnée
                    row_str.append(f"{idx:2d} -> {coord}")
                else:
                    # Case occupée
                    row_str.append("  ....  ")
                idx += 1
            print("  " + " | ".join(row_str))
        print()

    def render_available_pieces(self, game) -> None:
        """Affiche les pièces disponibles avec leur code visuel"""
        pieces = game.get_available_pieces()
        if pieces:
            pieces_str = " ".join(f"{p}:{encode_piece_code(p)}" for p in pieces)
            print(f"Pieces disponibles: {pieces_str}")
        else:
            print("Pieces disponibles: aucune")

    def render_game_header(self, title: str, show_rules: bool = True) -> None:
        """
        Affiche l'en-tête d'une partie avec la légende.

        Args:
            title: Titre de la partie (ex: "Quarto - Humain vs IA")
            show_rules: Afficher les règles de base
        """
        print("\n" + "=" * 50)
        print(title)
        print("=" * 50)
        self.render_piece_legend()
        if show_rules:
            print("Regles:")
            print("1. Choisissez une pièce pour votre adversaire")
            print("2. Placez votre pièce sur le plateau (index 0-15)")

    def render_turn_header(self, turn: int, player: int | None = None) -> None:
        """
        Affiche l'en-tête d'un tour.

        Args:
            turn: Numéro du tour
            player: Numéro du joueur (optionnel)
        """
        if player is not None:
            print(f"\n--- Tour {turn} (Joueur {player}) ---")
        else:
            print(f"\n--- Tour {turn} ---")

    def render_piece_choice(self, piece_id: int, prefix: str = "", suffix: str = "") -> None:
        """
        Affiche une pièce choisie avec son code visuel.

        Args:
            piece_id: ID de la pièce
            prefix: Texte à afficher avant (ex: "Piece choisie: ")
            suffix: Texte à afficher après le code (ex: " pour vous")
        """
        code = encode_piece_code(piece_id)
        print(f"{prefix}{Colors.LIGHT_BROWN}{code}{Colors.RESET}{suffix}")

    def render_game_over(self, game, player_names: tuple[str, str] = ("Joueur 0", "Joueur 1")) -> None:
        """
        Affiche le message de fin de partie.

        Args:
            game: Instance du jeu Quarto
            player_names: Tuple des noms des joueurs (joueur 0, joueur 1)
        """
        print("\n" + "=" * 50)
        print("FIN DU JEU")
        print("=" * 50)
        self.render(game)

        if game.winner is None:
            print(f"{Colors.YELLOW}Egalite ! Plateau plein.{Colors.RESET}")
        else:
            winner_name = player_names[game.winner]
            print(f"{Colors.GREEN}{winner_name} gagne !{Colors.RESET}")

    def render_message(self, message: str, color: str | None = None) -> None:
        """
        Affiche un message avec une couleur optionnelle.

        Args:
            message: Message à afficher
            color: Couleur ANSI (Colors.GREEN, Colors.BLUE, etc.)
        """
        if color:
            print(f"{color}{message}{Colors.RESET}")
        else:
            print(message)

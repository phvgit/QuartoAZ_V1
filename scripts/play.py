#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script pour jouer au Quarto"""

import sys
from pathlib import Path

# Ajouter le répertoire parent au path
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

import argparse
import numpy as np

from alphaquarto.game import Quarto
from alphaquarto.ai import MCTS
from alphaquarto.ui.colors import Colors
from alphaquarto.ui.input_handler import InputHandler
from alphaquarto.ui.display import QuartoDisplay, encode_piece_code, index_to_coord


# Configuration des niveaux de difficulte
DIFFICULTY_CONFIG = {
    'easy': {'num_simulations': 0, 'use_mcts': False},      # Aleatoire
    'medium': {'num_simulations': 50, 'use_mcts': True},    # MCTS leger
    'hard': {'num_simulations': 200, 'use_mcts': True},     # MCTS intensif
}


class SimpleQuartoGame:
    """Jeu Quarto avec IA configurable"""

    def __init__(self, difficulty: str = 'easy'):
        self.game = Quarto()
        self.input_handler = InputHandler()
        self.display = QuartoDisplay()
        self.difficulty = difficulty

        # Configurer l'IA selon la difficulte
        config = DIFFICULTY_CONFIG[difficulty]
        self.use_mcts = config['use_mcts']
        if self.use_mcts:
            self.mcts = MCTS(
                num_simulations=config['num_simulations'],
                use_dirichlet=False  # Pas de bruit pour le jeu
            )
        else:
            self.mcts = None

    def _get_ai_move(self) -> int:
        """Retourne le coup de l'IA selon la difficulte"""
        if self.use_mcts and self.mcts:
            print(f"{Colors.BLUE}(L'IA reflechit...){Colors.RESET}")
            return self.mcts.get_best_move(self.game)
        else:
            legal_moves = self.game.get_legal_moves()
            return np.random.choice(legal_moves)

    def _get_ai_piece(self) -> int:
        """Retourne la piece choisie par l'IA selon la difficulte"""
        if self.use_mcts and self.mcts:
            return self.mcts.get_best_piece(self.game)
        else:
            pieces = self.game.get_available_pieces()
            return np.random.choice(pieces)

    def play_human_vs_ai(self):
        """
        Lance une partie Humain vs IA.

        Flux correct du Quarto:
        1. Humain choisit la premiere piece (pour lui-meme)
        2. Humain PLACE la piece
        3. Humain CHOISIT une piece pour l'IA
        4. IA PLACE la piece que l'humain lui a donnee
        5. IA CHOISIT une piece pour l'Humain
        6. Retour a l'etape 2
        """
        difficulty_names = {'easy': 'Facile', 'medium': 'Moyen', 'hard': 'Difficile'}
        title = f"Quarto - Humain vs IA ({difficulty_names[self.difficulty]})"
        self.display.render_game_header(title)

        # Premiere piece - l'humain choisit pour lui-meme (simplification)
        print("\n" + "=" * 50)
        print("Choisissez la premiere piece (que vous allez placer):")
        self.display.render_available_pieces(self.game)
        first_piece = self.input_handler.get_piece_input(self.game.get_available_pieces())
        self.game.choose_piece(first_piece)
        self.display.render_piece_choice(first_piece, "Vous commencez avec la piece ")
        print()

        turn = 1

        while not self.game.game_over:
            # =====================
            # TOUR DE L'HUMAIN
            # =====================
            self.display.render_turn_header(turn)
            self.display.render(self.game)
            self.display.render_message("Votre tour - PLACEZ votre piece", Colors.GREEN)
            self.display.render_board_mapping(self.game)

            # Humain PLACE la piece courante
            legal_moves = self.game.get_legal_moves()
            move = self.input_handler.get_square_input(legal_moves)
            piece_code = encode_piece_code(self.game.current_piece)
            self.game.play_move(move)
            print(f"Vous avez place la piece {piece_code} sur la case {index_to_coord(move)}")

            if self.game.game_over:
                break

            # Humain CHOISIT une piece pour l'IA
            pieces = self.game.get_available_pieces()
            if not pieces:
                break
            print(f"\n{Colors.GREEN}Choisissez une piece pour l'IA:{Colors.RESET}")
            self.display.render_available_pieces(self.game)
            human_chosen_piece = self.input_handler.get_piece_input(pieces)
            self.game.choose_piece(human_chosen_piece)
            self.display.render_piece_choice(human_chosen_piece, "L'IA devra jouer avec la piece ")

            # =====================
            # TOUR DE L'IA
            # =====================
            print(f"\n{Colors.BLUE}--- Tour de l'IA ---{Colors.RESET}")
            self.display.render(self.game)

            # IA PLACE la piece que l'humain a choisie pour elle
            ai_move = self._get_ai_move()
            self.game.play_move(ai_move)
            print(f"L'IA a place la piece sur la case {ai_move}")
            self.display.render(self.game)

            if self.game.game_over:
                break

            # IA CHOISIT une piece pour l'Humain
            pieces = self.game.get_available_pieces()
            if not pieces:
                break
            ai_piece = self._get_ai_piece()
            self.game.choose_piece(ai_piece)
            self.display.render_piece_choice(ai_piece, f"{Colors.BLUE}L'IA choisit la piece ")
            print(f" pour vous{Colors.RESET}")

            turn += 1

        # Fin du jeu
        self.display.render_game_over(self.game, ("Humain", "IA"))

    def play_ai_vs_ai(self):
        """
        Lance une partie IA vs IA.

        Flux:
        1. Premiere piece choisie aleatoirement
        2. IA 0 PLACE, puis CHOISIT pour IA 1
        3. IA 1 PLACE, puis CHOISIT pour IA 0
        4. Alternance jusqu'a fin de partie
        """
        difficulty_names = {'easy': 'Facile', 'medium': 'Moyen', 'hard': 'Difficile'}
        title = f"Quarto - IA vs IA ({difficulty_names[self.difficulty]})"
        self.display.render_game_header(title, show_rules=False)

        # Premiere piece aleatoire
        first_piece = np.random.randint(1, 17)
        self.game.choose_piece(first_piece)
        self.display.render_piece_choice(first_piece, "Premiere piece: ")
        print()

        turn = 1
        current_ai = 0  # Alterne entre 0 et 1

        while not self.game.game_over:
            self.display.render_turn_header(turn)
            print(f"{Colors.BLUE}IA {current_ai} joue{Colors.RESET}")
            self.display.render(self.game)

            # IA courante PLACE la piece
            move = self._get_ai_move()
            self.game.play_move(move)
            print(f"IA {current_ai} a place la piece sur la case {move}")

            if self.game.game_over:
                break

            # IA courante CHOISIT une piece pour l'autre IA
            pieces = self.game.get_available_pieces()
            if not pieces:
                break
            piece = self._get_ai_piece()
            self.game.choose_piece(piece)
            next_ai = 1 - current_ai
            self.display.render_piece_choice(piece, f"IA {current_ai} choisit la piece ")
            print(f" pour IA {next_ai}")

            current_ai = next_ai
            turn += 1

        # Fin du jeu
        self.display.render_game_over(self.game, ("IA 0", "IA 1"))

    def play_human_vs_human(self):
        """
        Lance une partie Humain vs Humain.

        Flux:
        1. Joueur 1 choisit la premiere piece (pour lui-meme)
        2. Joueur 1 PLACE, puis CHOISIT pour Joueur 2
        3. Joueur 2 PLACE, puis CHOISIT pour Joueur 1
        4. Alternance jusqu'a fin de partie
        """
        self.display.render_game_header("Quarto - Humain vs Humain")

        # Premiere piece
        print("Joueur 1, choisissez la premiere piece (que vous allez placer):")
        self.display.render_available_pieces(self.game)
        first_piece = self.input_handler.get_piece_input(self.game.get_available_pieces())
        self.game.choose_piece(first_piece)
        self.display.render_piece_choice(first_piece, "Joueur 1 commence avec la piece ")

        player = 1
        turn = 1

        while not self.game.game_over:
            self.display.render_turn_header(turn, player)
            self.display.render(self.game)
            self.display.render_board_mapping(self.game)

            # Joueur actuel PLACE la piece
            print(f"Joueur {player}, placez votre piece:")
            legal_moves = self.game.get_legal_moves()
            move = self.input_handler.get_square_input(legal_moves)
            self.game.play_move(move)
            print(f"Joueur {player} a place la piece sur la case {move}")

            if self.game.game_over:
                break

            # Joueur actuel CHOISIT une piece pour l'adversaire
            pieces = self.game.get_available_pieces()
            if not pieces:
                break
            next_player = 3 - player  # Alterne entre 1 et 2
            print(f"\nJoueur {player}, choisissez une piece pour Joueur {next_player}:")
            self.display.render_available_pieces(self.game)
            piece = self.input_handler.get_piece_input(pieces)
            self.game.choose_piece(piece)
            self.display.render_piece_choice(piece, f"Joueur {next_player} devra jouer avec la piece ")

            player = next_player
            turn += 1

        # Fin du jeu
        self.display.render_game_over(self.game, ("Joueur 1", "Joueur 2"))


def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(
        description='Jouer au Quarto',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python scripts/play.py --mode human_vs_ai --difficulty easy
  python scripts/play.py --mode human_vs_ai --difficulty hard
  python scripts/play.py --mode ai_vs_ai --difficulty medium
  python scripts/play.py --mode human_vs_human

Niveaux de difficulte:
  easy   - IA aleatoire (instantane)
  medium - MCTS avec 50 simulations (rapide)
  hard   - MCTS avec 200 simulations (reflechie)

Regles du Quarto:
  - Apres avoir PLACE une piece, vous CHOISISSEZ une piece pour l'adversaire
  - L'adversaire place cette piece, puis choisit pour vous
  - Gagnez en alignant 4 pieces avec une propriete commune
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='human_vs_ai',
        choices=['human_vs_ai', 'ai_vs_ai', 'human_vs_human'],
        help='Mode de jeu (defaut: human_vs_ai)'
    )
    parser.add_argument(
        '--difficulty',
        type=str,
        default='easy',
        choices=['easy', 'medium', 'hard'],
        help='Difficulte de l\'IA (defaut: easy)'
    )

    args = parser.parse_args()

    # Lancer le jeu
    game = SimpleQuartoGame(difficulty=args.difficulty)

    if args.mode == 'human_vs_ai':
        game.play_human_vs_ai()
    elif args.mode == 'ai_vs_ai':
        game.play_ai_vs_ai()
    elif args.mode == 'human_vs_human':
        game.play_human_vs_human()

    print("\nMerci d'avoir joue !\n")


if __name__ == '__main__':
    main()

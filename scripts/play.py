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
    'random': {'num_simulations': 0, 'use_mcts': False},    # Aleatoire
    'easy': {'num_simulations': 100, 'use_mcts': True},     # MCTS leger (Rapide)
    'medium': {'num_simulations': 200, 'use_mcts': True},   # MCTS moyen (Standard)
    'hard': {'num_simulations': 500, 'use_mcts': True},     # MCTS intensif (Intensif)
}


class SimpleQuartoGame:
    """Jeu Quarto avec IA configurable"""

    def __init__(self, difficulty: str = 'easy', difficulty2: str = None):
        self.game = Quarto()
        self.input_handler = InputHandler()
        self.display = QuartoDisplay()
        self.difficulty = difficulty
        self.difficulty2 = difficulty2 if difficulty2 else difficulty

        # Configurer l'IA 0 selon la difficulte
        config = DIFFICULTY_CONFIG[difficulty]
        self.use_mcts = config['use_mcts']
        if self.use_mcts:
            self.mcts = MCTS(
                num_simulations=config['num_simulations'],
                use_dirichlet=False  # Pas de bruit pour le jeu
            )
        else:
            self.mcts = None

        # Configurer l'IA 1 (pour mode ai_vs_ai)
        config2 = DIFFICULTY_CONFIG[self.difficulty2]
        self.use_mcts2 = config2['use_mcts']
        if self.use_mcts2:
            self.mcts2 = MCTS(
                num_simulations=config2['num_simulations'],
                use_dirichlet=False
            )
        else:
            self.mcts2 = None

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

    def _get_ai_move_for(self, ai_id: int) -> int:
        """Retourne le coup de l'IA specifique selon sa difficulte"""
        if ai_id == 0:
            use_mcts = self.use_mcts
            mcts = self.mcts
        else:
            use_mcts = self.use_mcts2
            mcts = self.mcts2

        if use_mcts and mcts:
            print(f"{Colors.BLUE}(L'IA reflechit...){Colors.RESET}")
            return mcts.get_best_move(self.game)
        else:
            legal_moves = self.game.get_legal_moves()
            return np.random.choice(legal_moves)

    def _get_ai_piece_for(self, ai_id: int) -> int:
        """Retourne la piece choisie par l'IA specifique selon sa difficulte"""
        if ai_id == 0:
            use_mcts = self.use_mcts
            mcts = self.mcts
        else:
            use_mcts = self.use_mcts2
            mcts = self.mcts2

        if use_mcts and mcts:
            return mcts.get_best_piece(self.game)
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
        difficulty_names = {'random': 'Random', 'easy': 'Facile', 'medium': 'Moyen', 'hard': 'Difficile'}
        title = f"Quarto - Humain vs IA ({difficulty_names[self.difficulty]})"
        self.display.render_game_header(title)

        # Premiere piece - l'humain choisit pour lui-meme (simplification)
        print("\n" + "=" * 50)
        print("Choisissez la première pièce (que vous allez placer):")
        self.display.render_available_pieces(self.game)
        first_piece = self.input_handler.get_piece_input(self.game.get_available_pieces())
        self.game.choose_piece(first_piece)
        self.display.render_piece_choice(first_piece, "Vous commencez avec la pièce ")
        print()

        turn = 1

        while not self.game.game_over:
            # =====================
            # TOUR DE L'HUMAIN
            # =====================
            self.display.render_turn_header(turn)
            self.display.render(self.game, show_available_pieces=False, show_piece_in_hand=False)
            self.display.render_available_pieces(self.game, show_index=False)
            piece_in_hand = encode_piece_code(self.game.current_piece)
            self.display.render_message(f"Votre tour - PLACEZ la pièce {Colors.LIGHT_BROWN}{piece_in_hand}{Colors.RESET}", Colors.GREEN)
            self.display.render_board_mapping(self.game)

            # Humain PLACE la pièce courante
            legal_moves = self.game.get_legal_moves()
            move = self.input_handler.get_square_input(legal_moves)
            piece_code = encode_piece_code(self.game.current_piece)
            self.game.play_move(move)
            self.display.set_last_placed(move // 4, move % 4)
            print(f"Vous avez placé la pièce {piece_code} sur la case {index_to_coord(move)}")
            self.display.render(self.game)

            if self.game.game_over:
                break

            # Humain CHOISIT une pièce pour l'IA
            pieces = self.game.get_available_pieces()
            if not pieces:
                break
            print(f"{Colors.GREEN}Choisissez une pièce pour l'IA:{Colors.RESET}")
            self.display.render_available_pieces(self.game)
            human_chosen_piece = self.input_handler.get_piece_input(pieces)
            self.game.choose_piece(human_chosen_piece)
            self.display.render_piece_choice(human_chosen_piece, "L'IA devra jouer avec la pièce ")
            # =====================
            # TOUR DE L'IA
            # =====================
            print(f"\n{Colors.BLUE}--- Tour de l'IA ---{Colors.RESET}")

            # IA PLACE la pièce que l'humain a choisie pour elle
            ai_move = self._get_ai_move()
            piece_code = encode_piece_code(self.game.current_piece)
            self.game.play_move(ai_move)
            self.display.set_last_placed(ai_move // 4, ai_move % 4)
            print(f"L'IA a placé la pièce {piece_code} sur la case {index_to_coord(ai_move)}")
            self.display.render(self.game)

            if self.game.game_over:
                break

            # IA CHOISIT une pièce pour l'Humain
            pieces = self.game.get_available_pieces()
            if not pieces:
                break
            ai_piece = self._get_ai_piece()
            self.game.choose_piece(ai_piece)
            self.display.render_piece_choice(ai_piece, f"{Colors.BLUE}L'IA choisit la pièce ", f" pour vous{Colors.RESET}")

            turn += 1

        # Fin du jeu
        self.display.render_game_over(self.game, ("Humain", "IA"))

    def _play_single_ai_game_silent(self, starting_ai: int = 0) -> int:
        """
        Joue une partie IA vs IA silencieusement (sans affichage).

        Args:
            starting_ai: L'IA qui commence (0 ou 1)

        Returns:
            0 si IA 0 gagne, 1 si IA 1 gagne, -1 si match nul
        """
        # Reset du jeu
        self.game = Quarto()

        # Premiere piece aleatoire
        first_piece = np.random.randint(1, 17)
        self.game.choose_piece(first_piece)

        current_ai = starting_ai

        while not self.game.game_over:
            # IA courante PLACE la piece
            if current_ai == 0:
                use_mcts = self.use_mcts
                mcts = self.mcts
            else:
                use_mcts = self.use_mcts2
                mcts = self.mcts2

            if use_mcts and mcts:
                move = mcts.get_best_move(self.game)
            else:
                legal_moves = self.game.get_legal_moves()
                move = np.random.choice(legal_moves)

            self.game.play_move(move)

            if self.game.game_over:
                break

            # IA courante CHOISIT une piece pour l'autre IA
            pieces = self.game.get_available_pieces()
            if not pieces:
                break

            if use_mcts and mcts:
                piece = mcts.get_best_piece(self.game)
            else:
                piece = np.random.choice(list(pieces))

            self.game.choose_piece(piece)
            current_ai = 1 - current_ai

        # Determiner le gagnant
        if self.game.winner is None:
            return -1  # Match nul
        # Le gagnant est l'IA qui a joue le dernier coup (current_ai)
        return current_ai

    def play_ai_vs_ai_tournament(self, num_games: int):
        """
        Lance un tournoi de plusieurs parties IA vs IA.

        Args:
            num_games: Nombre de parties a jouer
        """
        difficulty_names = {'random': 'Random', 'easy': 'Facile', 'medium': 'Moyen', 'hard': 'Difficile'}
        level_0 = difficulty_names[self.difficulty]
        level_1 = difficulty_names[self.difficulty2]

        print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
        print(f"{Colors.BROWN}{Colors.BOLD}  TOURNOI IA VS IA{Colors.RESET}")
        print(f"{Colors.BOLD}{'='*60}{Colors.RESET}")
        print(f"\n  IA 0: {Colors.YELLOW}{level_0}{Colors.RESET}")
        print(f"  IA 1: {Colors.YELLOW}{level_1}{Colors.RESET}")
        print(f"  Parties: {Colors.CYAN}{num_games}{Colors.RESET}")
        print(f"\n{Colors.BLUE}Tournoi en cours...{Colors.RESET}\n")

        # Statistiques
        wins = {0: 0, 1: 0}
        draws = 0

        # Barre de progression
        progress_interval = max(1, num_games // 50)

        for i in range(num_games):
            # Alterner qui commence pour eliminer le biais du premier joueur
            starting_ai = i % 2
            result = self._play_single_ai_game_silent(starting_ai)

            if result == -1:
                draws += 1
            else:
                wins[result] += 1

            # Afficher la progression (caracteres ASCII pour compatibilite Windows)
            if (i + 1) % progress_interval == 0 or i == num_games - 1:
                progress = (i + 1) / num_games * 100
                bar_width = 40
                filled = int(bar_width * (i + 1) / num_games)
                bar = '#' * filled + '-' * (bar_width - filled)
                print(f"\r  [{bar}] {progress:5.1f}% ({i + 1}/{num_games})", end="", flush=True)

        print("\n")

        # Afficher le tableau recapitulatif
        self._render_tournament_results(wins, draws, num_games, level_0, level_1)

    def _render_tournament_results(self, wins: dict, draws: int, total: int, level_0: str, level_1: str):
        """Affiche le tableau recapitulatif du tournoi"""
        losses = {0: wins[1], 1: wins[0]}

        # Calculer les pourcentages
        def pct(n):
            return (n / total * 100) if total > 0 else 0

        # Largeurs des colonnes
        col_player = 20
        col_stat = 16  # Largeur pour chaque colonne de stats (nombre + pourcentage)

        # Fonction pour formater une cellule de stats
        def format_stat(value, pct_val):
            pct_str = f"{pct_val:.1f}%"
            return f"{value:>6}  {pct_str:>7}"

        # Ligne de separation
        total_width = col_player + 2 + (col_stat + 3) * 3
        separator = f"+{'-' * (col_player + 2)}+{'-' * (col_stat + 2)}+{'-' * (col_stat + 2)}+{'-' * (col_stat + 2)}+"

        # En-tete
        print(f"{Colors.BOLD}{'=' * total_width}{Colors.RESET}")
        print(f"{Colors.BROWN}{Colors.BOLD}  RESULTATS DU TOURNOI{Colors.RESET}")
        print(f"{Colors.BOLD}{'=' * total_width}{Colors.RESET}")
        print()

        # Ligne d'en-tete du tableau
        header = f"| {'Joueur':<{col_player}} | {'Gains':^{col_stat}} | {'Pertes':^{col_stat}} | {'Nuls':^{col_stat}} |"
        print(separator)
        print(header)
        print(separator)

        # Ligne IA 0
        player0 = f"IA 0 ({level_0})"
        row0 = f"| {player0:<{col_player}} | {format_stat(wins[0], pct(wins[0]))} | {format_stat(losses[0], pct(losses[0]))} | {format_stat(draws, pct(draws))} |"
        print(row0)

        # Ligne IA 1
        player1 = f"IA 1 ({level_1})"
        row1 = f"| {player1:<{col_player}} | {format_stat(wins[1], pct(wins[1]))} | {format_stat(losses[1], pct(losses[1]))} | {format_stat(draws, pct(draws))} |"
        print(row1)

        print(separator)

        # Ligne des totaux
        total_wins = wins[0] + wins[1]
        total_losses = losses[0] + losses[1]

        row_total = f"| {Colors.BOLD}{'TOTAL':<{col_player}}{Colors.RESET} | {format_stat(total_wins, pct(total_wins))} | {format_stat(total_losses, pct(total_losses))} | {format_stat(draws, pct(draws))} |"
        print(row_total)
        print(separator)

        # Resume
        print()
        print(f"  Total parties jouees: {Colors.CYAN}{total}{Colors.RESET}")
        if wins[0] > wins[1]:
            print(f"  {Colors.GREEN}Vainqueur: IA 0 ({level_0}){Colors.RESET}")
        elif wins[1] > wins[0]:
            print(f"  {Colors.GREEN}Vainqueur: IA 1 ({level_1}){Colors.RESET}")
        else:
            print(f"  {Colors.YELLOW}Egalite parfaite !{Colors.RESET}")
        print()

    def play_ai_vs_ai(self):
        """
        Lance une partie IA vs IA.

        Flux:
        1. Premiere piece choisie aleatoirement
        2. IA 0 PLACE, puis CHOISIT pour IA 1
        3. IA 1 PLACE, puis CHOISIT pour IA 0
        4. Alternance jusqu'a fin de partie
        """
        difficulty_names = {'random': 'Random', 'easy': 'Facile', 'medium': 'Moyen', 'hard': 'Difficile'}
        level_0 = difficulty_names[self.difficulty]
        level_1 = difficulty_names[self.difficulty2]
        title = f"Quarto - IA 0 ({level_0}) vs IA 1 ({level_1})"
        self.display.render_game_header(title, show_rules=False)

        # Noms des IAs avec leur niveau
        ai_names = {
            0: f"IA 0 ({level_0})",
            1: f"IA 1 ({level_1})"
        }

        # Premiere piece aleatoire
        first_piece = np.random.randint(1, 17)
        self.game.choose_piece(first_piece)
        self.display.render_piece_choice(first_piece, "Première pièce: ")
        print()

        turn = 1
        current_ai = 0  # Alterne entre 0 et 1

        while not self.game.game_over:
            self.display.render_turn_header(turn)
            print(f"{Colors.BLUE}{ai_names[current_ai]} joue{Colors.RESET}")
            self.display.render(self.game)

            # IA courante PLACE la piece
            move = self._get_ai_move_for(current_ai)
            piece_code = encode_piece_code(self.game.current_piece)
            self.game.play_move(move)
            self.display.set_last_placed(move // 4, move % 4)
            print(f"{ai_names[current_ai]} a placé la pièce {piece_code} sur la case {index_to_coord(move)}")

            if self.game.game_over:
                break

            # IA courante CHOISIT une pièce pour l'autre IA
            pieces = self.game.get_available_pieces()
            if not pieces:
                break
            piece = self._get_ai_piece_for(current_ai)
            self.game.choose_piece(piece)
            next_ai = 1 - current_ai
            self.display.render_piece_choice(piece, f"{ai_names[current_ai]} choisit la pièce ", f" pour {ai_names[next_ai]}")

            current_ai = next_ai
            turn += 1

        # Fin du jeu
        self.display.render_game_over(self.game, (ai_names[0], ai_names[1]))

    def play_human_vs_human(self):
        """
        Lance une partie Humain vs Humain.

        Flux:
        1. Joueur 1 choisit la première pièce (pour lui-même)
        2. Joueur 1 PLACE, puis CHOISIT pour Joueur 2
        3. Joueur 2 PLACE, puis CHOISIT pour Joueur 1
        4. Alternance jusqu'a fin de partie
        """
        self.display.render_game_header("Quarto - Humain vs Humain")

        # Première pièce
        print("Joueur 1, choisissez la première pièce (que vous allez placer):")
        self.display.render_available_pieces(self.game)
        first_piece = self.input_handler.get_piece_input(self.game.get_available_pieces())
        self.game.choose_piece(first_piece)
        self.display.render_piece_choice(first_piece, "Joueur 1 commence avec la pièce ")
        player = 1
        turn = 1

        while not self.game.game_over:
            self.display.render_turn_header(turn, player)
            self.display.render(self.game)
            self.display.render_board_mapping(self.game)

            # Joueur actuel PLACE la pièce
            print(f"Joueur {player}, placez votre pièce:")
            legal_moves = self.game.get_legal_moves()
            move = self.input_handler.get_square_input(legal_moves)
            piece_code = encode_piece_code(self.game.current_piece)
            self.game.play_move(move)
            self.display.set_last_placed(move // 4, move % 4)
            print(f"Joueur {player} a placé la pièce {piece_code} sur la case {index_to_coord(move)}")
            if self.game.game_over:
                break

            # Joueur actuel CHOISIT une pièce pour l'adversaire
            pieces = self.game.get_available_pieces()
            if not pieces:
                break
            next_player = 3 - player  # Alterne entre 1 et 2
            print(f"\nJoueur {player}, choisissez une pièce pour Joueur {next_player}:")
            self.display.render_available_pieces(self.game)
            piece = self.input_handler.get_piece_input(pieces)
            self.game.choose_piece(piece)
            self.display.render_piece_choice(piece, f"Joueur {next_player} devra jouer avec la pièce ")
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
  python scripts/play.py --mode ai_vs_ai --difficulty easy --difficulty2 hard
  python scripts/play.py --mode ai_vs_ai --difficulty easy --difficulty2 hard --games 100
  python scripts/play.py --mode human_vs_human

Niveaux de difficulte:
  random - IA aleatoire (instantane)
  easy   - MCTS avec 100 simulations (Rapide)
  medium - MCTS avec 200 simulations (Standard)
  hard   - MCTS avec 500 simulations (Intensif)

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
        default='random',
        choices=['random', 'easy', 'medium', 'hard'],
        help='Difficulte de l\'IA 0 (defaut: random)'
    )
    parser.add_argument(
        '--difficulty2',
        type=str,
        default=None,
        choices=['random', 'easy', 'medium', 'hard'],
        help='Difficulte de l\'IA 1 pour le mode ai_vs_ai (defaut: meme que difficulty)'
    )
    parser.add_argument(
        '--games',
        type=int,
        default=1,
        help='Nombre de parties pour le mode tournoi ai_vs_ai (defaut: 1)'
    )

    args = parser.parse_args()

    # Lancer le jeu
    game = SimpleQuartoGame(difficulty=args.difficulty, difficulty2=args.difficulty2)

    if args.mode == 'human_vs_ai':
        game.play_human_vs_ai()
    elif args.mode == 'ai_vs_ai':
        if args.games > 1:
            game.play_ai_vs_ai_tournament(args.games)
        else:
            game.play_ai_vs_ai()
    elif args.mode == 'human_vs_human':
        game.play_human_vs_human()

    print("\nMerci d'avoir joue !\n")


if __name__ == '__main__':
    main()

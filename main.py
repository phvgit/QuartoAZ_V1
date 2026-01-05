#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lanceur principal de Quarto AlphaZero avec menu interactif"""

import subprocess
import sys
from pathlib import Path


# Couleurs pour le terminal
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    BROWN = "\033[38;5;130m"


# Chemin vers les scripts
SCRIPTS_DIR = Path(__file__).parent / "scripts"


def clear_screen():
    """Efface l'ecran du terminal"""
    print("\033[2J\033[H", end="")


def print_header():
    """Affiche l'en-tete du menu"""
    print(f"{Colors.BROWN}{Colors.BOLD}")
    print(r"      ___   __    ____  _   _   ___   ____   _   _   ___   ____  ____  ___  ")
    print(r"     / _ | / /   / __ \/ / / / / _ | / __ \ / / / / / _ | / __ \/_  _// _ \ ")
    print(r"    / __ |/ /__ / /_/ /  _  / / __ |/ /_/ // /_/ / / __ |/ /_/ / / / / // / ")
    print(r"   /_/ |_|____// ____//_//_/ /_/ |_|\___\_\\__,_/ /_/ |_/_/  \_\/_/ /\___/  ")
    print(r"              /_/                                                          ")
    print(f"{Colors.RESET}")


def print_main_menu():
    """Affiche le menu principal"""
    print_header()
    print(f"{Colors.BROWN}  JOUER{Colors.RESET}")
    print(f"    1. Humain vs IA {Colors.YELLOW}(Random){Colors.RESET}")
    print(f"    2. Humain vs IA {Colors.YELLOW}(Facile){Colors.RESET}")
    print(f"    3. Humain vs IA {Colors.YELLOW}(Moyen){Colors.RESET}")
    print(f"    4. Humain vs IA {Colors.YELLOW}(Difficile){Colors.RESET}")
    print(f"    5. IA vs IA")
    print(f"    6. Humain vs Humain")
    print()
    print(f"{Colors.BROWN}  ENTRAINEMENT{Colors.RESET}")
    print(f"    7. Entrainer l'IA")
    print()
    print(f"{Colors.RED}    0. Quitter{Colors.RESET}")
    print()


def print_training_menu():
    """Affiche le sous-menu d'entrainement"""
    clear_screen()
    print_header()
    print(f"{Colors.BROWN}{Colors.BOLD}  ENTRAINEMENT DE L'IA{Colors.RESET}")
    print()
    print(f"  {Colors.BROWN}Presets:{Colors.RESET}")
    print(f"    1. Rapide    (10 iterations, 10 parties, 100 simulations)")
    print(f"    2. Standard  (100 iterations, 100 parties, 200 simulations)")
    print(f"    3. Intensif  (1000 iterations, 1000 parties, 500 simulations)")
    print()
    print(f"  {Colors.BROWN}Personnalise:{Colors.RESET}")
    print(f"    4. Parametres personnalises")
    print()
    print(f"  {Colors.RED}  0. Retour{Colors.RESET}")
    print()


def get_custom_training_params():
    """Demande les parametres personnalises pour l'entrainement"""
    print(f"\n{Colors.CYAN}Parametres personnalises:{Colors.RESET}")

    try:
        iterations = input(f"  Iterations (defaut: 10): ").strip()
        iterations = int(iterations) if iterations else 10

        games = input(f"  Parties par iteration (defaut: 10): ").strip()
        games = int(games) if games else 10

        sims = input(f"  Simulations MCTS (defaut: 100): ").strip()
        sims = int(sims) if sims else 100

        return iterations, games, sims
    except ValueError:
        print(f"{Colors.RED}Valeur invalide, utilisation des valeurs par defaut{Colors.RESET}")
        return 10, 10, 100


def run_play(mode: str, difficulty: str = "easy"):
    """Lance une partie de Quarto"""
    cmd = [sys.executable, str(SCRIPTS_DIR / "play.py"), "--mode", mode, "--difficulty", difficulty]
    subprocess.run(cmd)
    input(f"\n{Colors.CYAN}Appuyez sur Entree pour revenir au menu...{Colors.RESET}")


def run_training(iterations: int, games_per_iter: int, mcts_sims: int):
    """Lance l'entrainement de l'IA"""
    clear_screen()
    print(f"{Colors.BLUE}Lancement de l'entrainement...{Colors.RESET}")
    print(f"  - Iterations: {iterations}")
    print(f"  - Parties/iteration: {games_per_iter}")
    print(f"  - Simulations MCTS: {mcts_sims}")
    print()

    cmd = [
        sys.executable, str(SCRIPTS_DIR / "train.py"),
        "--iterations", str(iterations),
        "--games-per-iter", str(games_per_iter),
        "--mcts-sims", str(mcts_sims)
    ]
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n{Colors.RED}Echec de l'entrainement (code de sortie: {result.returncode}){Colors.RESET}")

    input(f"\n{Colors.CYAN}Appuyez sur Entree pour revenir au menu...{Colors.RESET}")


def handle_training_menu():
    """Gere le sous-menu d'entrainement"""
    # Presets d'entrainement
    presets = {
        "1": (10, 10, 100),       # Rapide
        "2": (100, 100, 200),     # Standard
        "3": (1000, 1000, 500),   # Intensif
    }

    while True:
        print_training_menu()
        choice = input(f"{Colors.BOLD}Choix [0-4]: {Colors.RESET}").strip()

        if choice == "0":
            return
        elif choice in presets:
            iterations, games, sims = presets[choice]
            run_training(iterations, games, sims)
        elif choice == "4":
            iterations, games, sims = get_custom_training_params()
            run_training(iterations, games, sims)
        else:
            print(f"{Colors.RED}Choix invalide{Colors.RESET}")


def main():
    """Fonction principale"""
    while True:
        clear_screen()
        print_main_menu()
        choice = input(f"{Colors.BOLD}Choix [0-7]: {Colors.RESET}").strip()

        if choice == "0":
            print(f"\n{Colors.GREEN}Merci d'avoir joue ! A bientot !{Colors.RESET}\n")
            break
        elif choice == "1":
            run_play("human_vs_ai", "random")
        elif choice == "2":
            run_play("human_vs_ai", "easy")
        elif choice == "3":
            run_play("human_vs_ai", "medium")
        elif choice == "4":
            run_play("human_vs_ai", "hard")
        elif choice == "5":
            run_play("ai_vs_ai", "medium")
        elif choice == "6":
            run_play("human_vs_human")
        elif choice == "7":
            handle_training_menu()
        else:
            print(f"{Colors.RED}Choix invalide{Colors.RESET}")


if __name__ == "__main__":
    main()

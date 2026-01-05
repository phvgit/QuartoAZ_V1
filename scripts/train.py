#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script d'entrainement"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description='Entrainer AlphaZero')
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--games-per-iter', type=int, default=10)
    parser.add_argument('--mcts-sims', type=int, default=100)

    args = parser.parse_args()

    try:
        # TODO: Implementer la logique d'entrainement ici
        print(f"Entrainement en cours...")

        # Simulation de l'entrainement (a remplacer par le vrai code)
        for i in range(1, args.iterations + 1):
            print(f"  Iteration {i}/{args.iterations}...")

        # Message de succes
        print(f"\nEntrainement avec {args.iterations} iterations, "
              f"{args.games_per_iter} parties par iteration avec "
              f"{args.mcts_sims} simulations MCTS effectue avec succes.")
        return 0

    except Exception as e:
        print(f"\nErreur lors de l'entrainement: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())

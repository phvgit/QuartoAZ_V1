#!/usr/bin/env python3
"""Script d'entraînement"""

import argparse

def main():
    parser = argparse.ArgumentParser(description='Entraîner AlphaZero')
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--games-per-iter', type=int, default=10)
    parser.add_argument('--mcts-sims', type=int, default=100)
    
    args = parser.parse_args()
    print(f"Entraînement avec {args.iterations} itérations")

if __name__ == '__main__':
    main()

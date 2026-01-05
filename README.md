# AlphaQuarto V1

Une implémentation d'**AlphaZero** pour le jeu **Quarto**.

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

### Entraîner
```bash
python scripts/train.py --iterations 5
```

### Jouer
```bash
python scripts/play.py --difficulty easy
```

## Structure
- `alphaquarto_v1/game/` : Moteur du jeu
- `alphaquarto_v1/ai/` : Algorithmes IA
- `alphaquarto_v1/ui/` : Interface CLI
- `scripts/` : Scripts exécutables
- `tests/` : Tests unitaires

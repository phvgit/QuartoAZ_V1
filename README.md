# AlphaQuarto V1

Une impl�mentation d'**AlphaZero** pour le jeu **Quarto**.

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

### Entra�ner
```bash
python scripts/train.py --iterations 5
```

### Jouer
```bash
python scripts/play.py --difficulty easy
```

## Structure
- `alphaquarto/game/` : Moteur du jeu
- `alphaquarto/ai/` : Algorithmes IA
- `alphaquarto/ui/` : Interface CLI
- `scripts/` : Scripts ex�cutables
- `tests/` : Tests unitaires

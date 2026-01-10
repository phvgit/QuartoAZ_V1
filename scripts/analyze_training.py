#!/usr/bin/env python3
"""
Analyse des statistiques d'entraînement AlphaZero pour Quarto.
Génère des graphiques et identifie le meilleur checkpoint.
"""

import json
import os
import math
from pathlib import Path

# Essayer d'importer matplotlib, sinon afficher en mode texte
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib non installé. Affichage en mode texte uniquement.")
    print("Pour installer: pip install matplotlib\n")


def load_training_stats(stats_path: str) -> list:
    """Charge les statistiques d'entraînement."""
    with open(stats_path, 'r') as f:
        return json.load(f)


def find_best_checkpoint(stats: list) -> dict:
    """
    Trouve le meilleur checkpoint avant l'explosion NaN.
    Critère: loss la plus basse (non-NaN)
    """
    best = None
    best_loss = float('inf')

    for s in stats:
        loss = s['training']['loss']
        if loss is not None and not math.isnan(loss) and loss < best_loss:
            best_loss = loss
            best = s

    return best


def find_nan_start(stats: list) -> int:
    """Trouve l'itération où le NaN a commencé."""
    for s in stats:
        loss = s['training']['loss']
        if loss is None or math.isnan(loss):
            return s['iteration']
    return -1


def print_summary_table(stats: list):
    """Affiche un tableau récapitulatif des statistiques."""
    print("\n" + "=" * 80)
    print("RÉSUMÉ DE L'ENTRAÎNEMENT")
    print("=" * 80)

    # Statistiques générales
    total_games = stats[-1]['total_games']
    total_iterations = len(stats)
    nan_start = find_nan_start(stats)
    best = find_best_checkpoint(stats)

    print(f"\n{'Métrique':<30} {'Valeur':>20}")
    print("-" * 52)
    print(f"{'Itérations totales':<30} {total_iterations:>20}")
    print(f"{'Parties jouées':<30} {total_games:>20}")
    print(f"{'Itération explosion NaN':<30} {nan_start:>20}")
    print(f"{'Itérations valides':<30} {nan_start - 1:>20}")
    print(f"{'Meilleur checkpoint':<30} {'Itération ' + str(best['iteration']):>20}")
    print(f"{'Meilleure loss':<30} {best['training']['loss']:>20.4f}")

    # Tableau détaillé des 15 premières itérations
    print("\n" + "=" * 80)
    print("DÉTAIL PAR ITÉRATION (avant explosion)")
    print("=" * 80)

    print(f"\n{'Iter':>5} {'Loss':>10} {'Policy':>10} {'Piece':>10} {'Value':>10} {'Pol.Acc':>10} {'Games/s':>10}")
    print("-" * 75)

    for s in stats[:min(15, nan_start)]:
        t = s['training']
        sp = s['selfplay']

        loss = t['loss']
        policy_loss = t['policy_loss']
        piece_loss = t['piece_loss']
        value_loss = t['value_loss']
        policy_acc = t['policy_accuracy']
        games_per_sec = sp['games_per_second']

        # Formatage
        loss_str = f"{loss:.4f}" if not math.isnan(loss) else "NaN"
        piece_str = f"{piece_loss:.4f}" if not math.isnan(piece_loss) else "NaN"

        print(f"{s['iteration']:>5} {loss_str:>10} {policy_loss:>10.4f} {piece_str:>10} {value_loss:>10.4f} {policy_acc:>10.2%} {games_per_sec:>10.3f}")

    if nan_start > 15:
        print(f"  ... (itérations 16 à {nan_start - 1} omises)")

    # Statistiques de self-play
    print("\n" + "=" * 80)
    print("STATISTIQUES SELF-PLAY")
    print("=" * 80)

    total_wins_p0 = sum(s['selfplay']['wins_player_0'] for s in stats)
    total_wins_p1 = sum(s['selfplay']['wins_player_1'] for s in stats)
    total_draws = sum(s['selfplay']['draws'] for s in stats)
    avg_moves = sum(s['selfplay']['avg_moves'] for s in stats) / len(stats)
    avg_speed = sum(s['selfplay']['games_per_second'] for s in stats) / len(stats)

    print(f"\n{'Métrique':<30} {'Valeur':>20}")
    print("-" * 52)
    print(f"{'Victoires Joueur 0':<30} {total_wins_p0:>20} ({total_wins_p0/total_games:.1%})")
    print(f"{'Victoires Joueur 1':<30} {total_wins_p1:>20} ({total_wins_p1/total_games:.1%})")
    print(f"{'Matchs nuls':<30} {total_draws:>20} ({total_draws/total_games:.1%})")
    print(f"{'Coups moyens/partie':<30} {avg_moves:>20.2f}")
    print(f"{'Vitesse moyenne':<30} {avg_speed:>17.3f} p/s")


def print_recommendations(stats: list):
    """Affiche les recommandations basées sur l'analyse."""
    best = find_best_checkpoint(stats)
    nan_start = find_nan_start(stats)

    print("\n" + "=" * 80)
    print("RECOMMANDATIONS")
    print("=" * 80)

    print(f"""
1. UTILISER LE CHECKPOINT DE L'ITÉRATION {best['iteration']}
   C'est le dernier checkpoint valide avec la meilleure loss ({best['training']['loss']:.4f})

   Commande pour charger ce modèle:
   python main.py --checkpoint {best['iteration']}

2. PROBLÈME IDENTIFIÉ: Explosion des gradients à l'itération {nan_start}
   - Le 'piece_loss' a explosé (NaN) tandis que policy_loss restait stable
   - Cause probable: mixed precision (float16) + learning rate trop élevé

3. POUR REPRENDRE L'ENTRAÎNEMENT (corrigé):
   python scripts/train.py --resume-from {best['iteration']} --iterations 100 \\
       --learning-rate 0.0001 --no-mixed-precision

4. AMÉLIORATIONS SUGGÉRÉES:
   - Réduire le learning rate: 0.001 → 0.0001
   - Désactiver mixed precision OU ajouter gradient clipping
   - Augmenter le nombre de parties par itération (100 → 200)
""")


def plot_training_curves(stats: list, output_dir: str):
    """Génère des graphiques des courbes d'entraînement."""
    if not HAS_MATPLOTLIB:
        print("Graphiques non générés (matplotlib non disponible)")
        return

    # Préparer les données
    iterations = [s['iteration'] for s in stats]

    # Filtrer les NaN pour les graphiques
    valid_stats = [s for s in stats if not math.isnan(s['training']['loss'])]
    valid_iters = [s['iteration'] for s in valid_stats]

    losses = [s['training']['loss'] for s in valid_stats]
    policy_losses = [s['training']['policy_loss'] for s in valid_stats]
    piece_losses = [s['training']['piece_loss'] for s in valid_stats if not math.isnan(s['training']['piece_loss'])]
    piece_iters = [s['iteration'] for s in valid_stats if not math.isnan(s['training']['piece_loss'])]
    value_losses = [s['training']['value_loss'] for s in valid_stats]

    policy_accs = [s['training']['policy_accuracy'] for s in valid_stats]
    value_maes = [s['training']['value_mae'] for s in valid_stats]

    games_per_sec = [s['selfplay']['games_per_second'] for s in stats]
    wins_p0 = [s['selfplay']['wins_player_0'] for s in stats]
    wins_p1 = [s['selfplay']['wins_player_1'] for s in stats]
    draws = [s['selfplay']['draws'] for s in stats]

    # Créer la figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Analyse de l\'Entraînement AlphaZero - Quarto', fontsize=14, fontweight='bold')

    # 1. Loss totale
    ax1 = axes[0, 0]
    ax1.plot(valid_iters, losses, 'b-', linewidth=2, label='Total Loss')
    ax1.axvline(x=find_nan_start(stats), color='r', linestyle='--', label='Explosion NaN')
    ax1.set_xlabel('Itération')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Totale')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Losses individuelles
    ax2 = axes[0, 1]
    ax2.plot(valid_iters, policy_losses, 'g-', linewidth=1.5, label='Policy Loss')
    ax2.plot(piece_iters, piece_losses, 'orange', linewidth=1.5, label='Piece Loss')
    ax2.plot(valid_iters, value_losses, 'purple', linewidth=1.5, label='Value Loss')
    ax2.axvline(x=find_nan_start(stats), color='r', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Itération')
    ax2.set_ylabel('Loss')
    ax2.set_title('Losses par Composant')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Précision
    ax3 = axes[0, 2]
    ax3.plot(valid_iters, [a * 100 for a in policy_accs], 'g-', linewidth=2, label='Policy Accuracy')
    ax3.axvline(x=find_nan_start(stats), color='r', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Itération')
    ax3.set_ylabel('Précision (%)')
    ax3.set_title('Précision de la Policy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Vitesse de génération
    ax4 = axes[1, 0]
    ax4.plot(iterations, games_per_sec, 'b-', linewidth=2)
    ax4.axhline(y=sum(games_per_sec)/len(games_per_sec), color='r', linestyle='--',
                label=f'Moyenne: {sum(games_per_sec)/len(games_per_sec):.3f} p/s')
    ax4.set_xlabel('Itération')
    ax4.set_ylabel('Parties/seconde')
    ax4.set_title('Vitesse de Self-Play')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Distribution des victoires
    ax5 = axes[1, 1]
    ax5.stackplot(iterations, wins_p0, wins_p1, draws,
                  labels=['Joueur 0', 'Joueur 1', 'Nuls'],
                  colors=['#2ecc71', '#e74c3c', '#95a5a6'], alpha=0.8)
    ax5.set_xlabel('Itération')
    ax5.set_ylabel('Nombre de parties')
    ax5.set_title('Distribution des Résultats')
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)

    # 6. Value MAE
    ax6 = axes[1, 2]
    ax6.plot(valid_iters, value_maes, 'purple', linewidth=2)
    ax6.axvline(x=find_nan_start(stats), color='r', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Itération')
    ax6.set_ylabel('MAE')
    ax6.set_title('Erreur de Prédiction de Valeur')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    # Sauvegarder
    output_path = os.path.join(output_dir, 'training_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nGraphique sauvegardé: {output_path}")

    # Afficher
    plt.show()


def main():
    # Chemins
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    stats_path = project_dir / 'data' / 'models' / 'training_stats.json'
    output_dir = project_dir / 'data' / 'models'

    print("=" * 80)
    print("ANALYSE DES STATISTIQUES D'ENTRAÎNEMENT")
    print("=" * 80)
    print(f"\nFichier: {stats_path}")

    # Charger les données
    if not stats_path.exists():
        print(f"Erreur: Fichier non trouvé: {stats_path}")
        return

    stats = load_training_stats(stats_path)
    print(f"Itérations chargées: {len(stats)}")

    # Afficher les analyses
    print_summary_table(stats)
    print_recommendations(stats)

    # Générer les graphiques
    plot_training_curves(stats, str(output_dir))


if __name__ == '__main__':
    main()

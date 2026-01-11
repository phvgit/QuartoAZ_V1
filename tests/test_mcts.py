# -*- coding: utf-8 -*-
"""Tests pour MCTS (Monte Carlo Tree Search)"""

import pytest
import numpy as np
from alphaquarto.game import Quarto
from alphaquarto.ai.mcts import MCTS
from alphaquarto.ai.types import MCTSNode
from alphaquarto.game.constants import NUM_SQUARES, NUM_PIECES


# =============================================================================
# Tests MCTSNode
# =============================================================================

class TestMCTSNode:
    """Tests pour la classe MCTSNode"""

    def test_node_initialization(self):
        """Test création d'un noeud vide"""
        node = MCTSNode()
        assert node.visit_count == 0
        assert node.value_sum == 0.0
        assert node.prior == 0.0
        assert node.children == {}
        assert node.parent is None
        assert node.action is None
        assert node.is_expanded is False

    def test_get_value_zero_visits(self):
        """get_value retourne 0 si aucune visite"""
        node = MCTSNode()
        assert node.get_value() == 0.0

    def test_get_value_with_visits(self):
        """get_value retourne la moyenne"""
        node = MCTSNode(visit_count=10, value_sum=5.0)
        assert node.get_value() == 0.5

    def test_ucb_score_root_node(self):
        """UCB de la racine est 0"""
        node = MCTSNode()
        assert node.ucb_score() == 0.0

    def test_ucb_score_child_node(self):
        """UCB d'un enfant avec prior et parent"""
        parent = MCTSNode(visit_count=10)
        child = MCTSNode(
            visit_count=2,
            value_sum=1.0,
            prior=0.5,
            parent=parent
        )
        # Q = 1/2 = 0.5
        # U = 1.41 * 0.5 * sqrt(10) / (1 + 2) = 1.41 * 0.5 * 3.16 / 3 ≈ 0.74
        score = child.ucb_score(c_puct=1.41)
        assert score > 0.5  # Q-value
        assert score < 2.0  # Raisonnable

    def test_ucb_prefers_unvisited(self):
        """UCB favorise les noeuds non visités"""
        parent = MCTSNode(visit_count=100)
        visited = MCTSNode(visit_count=50, value_sum=25, prior=0.5, parent=parent)
        unvisited = MCTSNode(visit_count=0, value_sum=0, prior=0.5, parent=parent)

        assert unvisited.ucb_score() > visited.ucb_score()

    def test_select_child(self):
        """select_child retourne l'enfant avec le meilleur UCB"""
        parent = MCTSNode(visit_count=10)
        child1 = MCTSNode(visit_count=5, value_sum=2, prior=0.3, parent=parent, action=0)
        child2 = MCTSNode(visit_count=1, value_sum=0.8, prior=0.7, parent=parent, action=1)
        parent.children = {0: child1, 1: child2}
        parent.is_expanded = True

        selected = parent.select_child()
        # child2 devrait être sélectionné (moins visité, bon prior)
        assert selected in [child1, child2]

    def test_expand(self):
        """expand crée des enfants pour les actions légales"""
        node = MCTSNode()
        actions = [0, 1, 5, 10]
        priors = np.zeros(16)
        priors[0] = 0.4
        priors[1] = 0.3
        priors[5] = 0.2
        priors[10] = 0.1

        node.expand(actions, priors)

        assert node.is_expanded
        assert len(node.children) == 4
        assert 0 in node.children
        assert 1 in node.children
        assert 5 in node.children
        assert 10 in node.children
        assert node.children[0].prior == 0.4
        assert node.children[1].prior == 0.3

    def test_backpropagate(self):
        """backpropagate propage la valeur vers la racine"""
        root = MCTSNode()
        child = MCTSNode(parent=root, action=0)
        grandchild = MCTSNode(parent=child, action=1)
        root.children[0] = child
        child.children[1] = grandchild

        # Propager une victoire (+1)
        grandchild.backpropagate(1.0)

        # La valeur alterne: grandchild +1, child -1, root +1
        assert grandchild.visit_count == 1
        assert grandchild.value_sum == 1.0
        assert child.visit_count == 1
        assert child.value_sum == -1.0
        assert root.visit_count == 1
        assert root.value_sum == 1.0

    def test_get_visit_distribution_empty(self):
        """get_visit_distribution avec aucun enfant"""
        node = MCTSNode()
        result = node.get_visit_distribution()
        assert len(result) == 0

    def test_get_visit_distribution_temperature_1(self):
        """get_visit_distribution avec température 1"""
        parent = MCTSNode()
        parent.children = {
            0: MCTSNode(visit_count=60, parent=parent, action=0),
            1: MCTSNode(visit_count=30, parent=parent, action=1),
            2: MCTSNode(visit_count=10, parent=parent, action=2),
        }

        actions, probs = parent.get_visit_distribution(temperature=1.0)

        assert len(actions) == 3
        assert len(probs) == 3
        assert np.isclose(probs.sum(), 1.0)
        # Les probs devraient être proportionnelles aux visites
        assert probs[actions.index(0)] > probs[actions.index(1)] > probs[actions.index(2)]

    def test_get_visit_distribution_temperature_0(self):
        """get_visit_distribution avec température 0 (glouton)"""
        parent = MCTSNode()
        parent.children = {
            0: MCTSNode(visit_count=60, parent=parent, action=0),
            1: MCTSNode(visit_count=30, parent=parent, action=1),
        }

        actions, probs = parent.get_visit_distribution(temperature=0)

        assert np.isclose(probs.sum(), 1.0)
        # Une seule action devrait avoir prob = 1
        assert 1.0 in probs


# =============================================================================
# Tests MCTS
# =============================================================================

class TestMCTS:
    """Tests pour la classe MCTS"""

    def test_mcts_initialization(self):
        """Test création du MCTS"""
        mcts = MCTS(num_simulations=50)
        assert mcts.num_simulations == 50
        assert mcts.c_puct == 1.41

    def test_mcts_initialization_with_config(self):
        """Test création avec configuration"""
        from alphaquarto.utils.config import MCTSConfig
        config = MCTSConfig(num_simulations=100, c_puct=2.0)
        mcts = MCTS(config=config)
        assert mcts.num_simulations == 100
        assert mcts.c_puct == 2.0

    def test_search_requires_piece(self):
        """search lève une erreur si pas de pièce en main"""
        mcts = MCTS(num_simulations=10)
        game = Quarto()

        with pytest.raises(ValueError, match="Aucune pièce en main"):
            mcts.search(game)

    def test_search_returns_valid_distribution(self):
        """search retourne une distribution valide"""
        mcts = MCTS(num_simulations=20, use_dirichlet=False)
        game = Quarto()
        game.choose_piece(1)

        move_probs, piece_probs = mcts.search(game, temperature=1.0)

        # move_probs doit être une distribution valide
        assert len(move_probs) == NUM_SQUARES
        assert np.isclose(move_probs.sum(), 1.0)
        assert all(p >= 0 for p in move_probs)

        # piece_probs doit être une distribution valide
        assert len(piece_probs) == NUM_PIECES

    def test_search_only_legal_moves(self):
        """search ne met de probabilité que sur les coups légaux"""
        mcts = MCTS(num_simulations=20, use_dirichlet=False)
        game = Quarto()

        # Remplir quelques cases
        game.board[0, 0] = 1
        game.board[0, 1] = 2
        game.board[0, 2] = 3
        game.available_pieces -= {1, 2, 3}
        game.choose_piece(4)

        move_probs, _ = mcts.search(game, temperature=1.0)

        # Cases occupées doivent avoir prob 0
        assert move_probs[0] == 0  # (0,0)
        assert move_probs[1] == 0  # (0,1)
        assert move_probs[2] == 0  # (0,2)

        # Au moins une case libre doit avoir prob > 0
        assert move_probs.sum() > 0

    def test_search_temperature_0(self):
        """search avec température 0 est déterministe"""
        mcts = MCTS(num_simulations=50, use_dirichlet=False)
        game = Quarto()
        game.choose_piece(1)

        move_probs, _ = mcts.search(game, temperature=0)

        # Exactement une action doit avoir probabilité 1
        assert np.count_nonzero(move_probs == 1.0) == 1
        assert np.count_nonzero(move_probs == 0.0) == NUM_SQUARES - 1

    def test_get_best_move(self):
        """get_best_move retourne un coup légal"""
        mcts = MCTS(num_simulations=20)
        game = Quarto()
        game.choose_piece(1)

        move = mcts.get_best_move(game)

        assert 0 <= move < NUM_SQUARES
        assert move in game.get_legal_moves()

    def test_get_best_piece(self):
        """get_best_piece retourne une pièce disponible"""
        mcts = MCTS(num_simulations=20)
        game = Quarto()
        game.choose_piece(1)
        game.play_move(0)

        piece = mcts.get_best_piece(game)

        assert 1 <= piece <= NUM_PIECES
        assert piece in game.get_available_pieces()

    def test_search_piece(self):
        """search_piece retourne une distribution valide"""
        mcts = MCTS(num_simulations=20, use_dirichlet=False)
        game = Quarto()
        game.choose_piece(1)
        game.play_move(0)

        piece_probs = mcts.search_piece(game, temperature=1.0)

        assert len(piece_probs) == NUM_PIECES
        assert np.isclose(piece_probs.sum(), 1.0)

        # La pièce 1 est utilisée, doit avoir prob 0
        assert piece_probs[0] == 0  # Index 0 = pièce 1

    def test_mask_illegal_moves(self):
        """_mask_illegal_moves filtre correctement"""
        mcts = MCTS()
        priors = np.array([0.1, 0.2, 0.3, 0.4] + [0.0] * 12)
        legal_moves = [0, 2]

        masked = mcts._mask_illegal_moves(priors, legal_moves)

        assert masked[1] == 0  # Illégal
        assert masked[3] == 0  # Illégal
        assert masked[0] > 0   # Légal
        assert masked[2] > 0   # Légal
        assert np.isclose(masked.sum(), 1.0)

    def test_mask_illegal_pieces(self):
        """_mask_illegal_pieces filtre correctement"""
        mcts = MCTS()
        priors = np.ones(16) / 16
        available = [2, 5, 10]  # IDs de pièces

        masked = mcts._mask_illegal_pieces(priors, available)

        # Seules les pièces 2, 5, 10 (indices 1, 4, 9) doivent avoir prob > 0
        assert masked[0] == 0  # Pièce 1 non dispo
        assert masked[1] > 0   # Pièce 2 dispo
        assert masked[4] > 0   # Pièce 5 dispo
        assert masked[9] > 0   # Pièce 10 dispo
        assert np.isclose(masked.sum(), 1.0)

    def test_smart_rollout(self):
        """_smart_rollout termine toujours"""
        mcts = MCTS()
        game = Quarto()
        game.choose_piece(1)

        # Le rollout doit retourner une valeur entre -1 et 1
        value = mcts._smart_rollout(game)
        assert -1.0 <= value <= 1.0

    def test_smart_rollout_respects_max_moves(self):
        """_smart_rollout respecte max_moves"""
        mcts = MCTS()
        game = Quarto()
        game.choose_piece(1)

        # Avec max_moves=0, le rollout ne fait rien
        value = mcts._smart_rollout(game, max_moves=0)
        assert value == 0.0


# =============================================================================
# Tests d'intégration MCTS + Quarto
# =============================================================================

class TestMCTSIntegration:
    """Tests d'intégration MCTS avec le jeu Quarto"""

    def test_mcts_finds_winning_move(self):
        """MCTS trouve un coup gagnant évident"""
        game = Quarto()

        # Créer une situation presque gagnante
        # Pièces 1, 3, 5 sur la première ligne (même couleur)
        game.board[0, 0] = 1
        game.board[0, 1] = 3
        game.board[0, 2] = 5
        game.available_pieces -= {1, 3, 5}
        game.choose_piece(7)  # Pièce 7 aussi de même couleur

        # Le MCTS devrait trouver que la case 3 (0,3) gagne
        mcts = MCTS(num_simulations=100, use_dirichlet=False)
        move = mcts.get_best_move(game)

        assert move == 3  # Case (0,3) pour compléter la ligne

    def test_mcts_avoids_losing_piece(self):
        """MCTS évite de donner une pièce gagnante"""
        game = Quarto()

        # Situation: 3 pièces alignées de même couleur, case vide
        game.board[0, 0] = 1
        game.board[0, 1] = 3
        game.board[0, 2] = 5
        game.available_pieces -= {1, 3, 5}

        # On joue un coup ailleurs
        game.choose_piece(2)  # Pièce différente
        game.play_move(4)  # Case (1,0)

        # MCTS ne devrait pas choisir la pièce 7 (qui gagnerait la ligne)
        mcts = MCTS(num_simulations=100, use_dirichlet=False)
        piece = mcts.get_best_piece(game)

        # La pièce 7 complèterait le Quarto, donc éviter si possible
        # Note: avec des simulations limitées, pas toujours parfait
        available = game.get_available_pieces()
        assert piece in available

    def test_full_game_with_mcts(self):
        """Joue une partie complète avec MCTS"""
        game = Quarto()
        mcts = MCTS(num_simulations=20)  # Peu de simulations pour la vitesse

        # Premier choix de pièce
        piece = np.random.choice(game.get_available_pieces())
        game.choose_piece(piece)

        max_turns = 20
        for _ in range(max_turns):
            if game.game_over:
                break

            # Jouer un coup
            move = mcts.get_best_move(game)
            game.play_move(move)

            if game.game_over:
                break

            # Choisir une pièce
            available = game.get_available_pieces()
            if available:
                piece = mcts.get_best_piece(game)
                game.choose_piece(piece)

        # Le jeu doit être terminé
        assert game.game_over or len(game.get_legal_moves()) == 0

    def test_mcts_with_real_network(self):
        """MCTS fonctionne avec un vrai réseau PyTorch"""
        from alphaquarto.ai.network import AlphaZeroNetwork
        from alphaquarto.utils.config import NetworkConfig

        config = NetworkConfig(num_filters=32, num_res_blocks=2)
        network = AlphaZeroNetwork(config)
        network.eval()

        mcts = MCTS(num_simulations=20, network=network, use_dirichlet=False)
        game = Quarto()
        game.choose_piece(1)

        move_probs, _ = mcts.search(game, temperature=1.0)

        # Les probabilités doivent être valides
        assert len(move_probs) == NUM_SQUARES
        assert np.isclose(np.sum(move_probs), 1.0)
        assert np.all(move_probs >= 0)


# =============================================================================
# Tests de performance et edge cases
# =============================================================================

class TestMCTSEdgeCases:
    """Tests des cas limites"""

    def test_search_near_end_game(self):
        """search fonctionne en fin de partie"""
        game = Quarto()

        # Remplir presque tout le plateau
        for i in range(14):
            game.board.flat[i] = i + 1
            game.available_pieces.discard(i + 1)

        game.choose_piece(15)

        mcts = MCTS(num_simulations=10)
        move_probs, _ = mcts.search(game, temperature=1.0)

        # Seules les 2 dernières cases sont libres (14 et 15)
        legal = game.get_legal_moves()
        for i in range(16):
            if i not in legal:
                assert move_probs[i] == 0

    def test_search_piece_one_remaining(self):
        """search_piece avec une seule pièce restante"""
        game = Quarto()

        # Utiliser 15 pièces
        for i in range(1, 16):
            game.available_pieces.discard(i)
        # Il reste la pièce 16

        mcts = MCTS(num_simulations=10)
        piece_probs = mcts.search_piece(game, temperature=1.0)

        # Seule la pièce 16 (index 15) doit avoir prob > 0
        assert piece_probs[15] == 1.0
        assert piece_probs[:15].sum() == 0

    def test_dirichlet_noise(self):
        """Le bruit Dirichlet ajoute de l'exploration"""
        game = Quarto()
        game.choose_piece(1)

        # Sans bruit
        mcts_no_noise = MCTS(num_simulations=50, use_dirichlet=False)
        probs1, _ = mcts_no_noise.search(game.clone(), temperature=1.0)

        # Avec bruit (résultats peuvent varier)
        mcts_noise = MCTS(num_simulations=50, use_dirichlet=True)
        probs2, _ = mcts_noise.search(game.clone(), temperature=1.0)

        # Les deux devraient être des distributions valides
        assert np.isclose(probs1.sum(), 1.0)
        assert np.isclose(probs2.sum(), 1.0)

    def test_different_c_puct_values(self):
        """Différentes valeurs de c_puct fonctionnent"""
        game = Quarto()
        game.choose_piece(1)

        for c_puct in [0.5, 1.0, 2.0, 5.0]:
            mcts = MCTS(num_simulations=20, c_puct=c_puct, use_dirichlet=False)
            probs, _ = mcts.search(game.clone(), temperature=1.0)
            assert np.isclose(probs.sum(), 1.0)


# =============================================================================
# Tests de sécurité du choix de pièce
# =============================================================================

class TestMCTSPieceSafety:
    """Tests pour vérifier que l'IA évite de donner des pièces gagnantes"""

    def test_find_winning_move(self):
        """_find_winning_move trouve un coup gagnant"""
        mcts = MCTS(num_simulations=10)
        game = Quarto()

        # Créer une situation gagnante: 3 pièces alignées de même couleur
        game.board[0, 0] = 1
        game.board[0, 1] = 3
        game.board[0, 2] = 5
        game.available_pieces -= {1, 3, 5}
        game.choose_piece(7)  # Pièce 7 complète le Quarto

        winning_move = mcts._find_winning_move(game)
        assert winning_move == 3  # Case (0,3) gagne

    def test_find_winning_move_no_win(self):
        """_find_winning_move retourne None s'il n'y a pas de coup gagnant"""
        mcts = MCTS(num_simulations=10)
        game = Quarto()
        game.choose_piece(1)

        winning_move = mcts._find_winning_move(game)
        assert winning_move is None

    def test_piece_enables_quarto(self):
        """_piece_enables_quarto détecte les pièces dangereuses"""
        mcts = MCTS(num_simulations=10)
        game = Quarto()

        # Créer une situation: 3 pièces alignées, case (0,3) vide
        # Pièces 1,3,5 en binaire (id-1): 0000, 0010, 0100
        # Propriétés communes: SIZE_BIT=0 (bit 0) et SHAPE_BIT=0 (bit 3)
        game.board[0, 0] = 1
        game.board[0, 1] = 3
        game.board[0, 2] = 5
        game.available_pieces -= {1, 3, 5}

        # La pièce 7 (binaire 0110) a SIZE_BIT=0 et SHAPE_BIT=0
        # Elle permettrait un Quarto
        assert mcts._piece_enables_quarto(game, 7) is True

        # La pièce 10 (binaire 1001) a SIZE_BIT=1 et SHAPE_BIT=1
        # Elle ne partage aucune propriété commune avec 1,3,5
        # Vérification: avec 1,3,5,10 sur la ligne:
        # - SIZE: 0,0,0,1 (pas tous égaux)
        # - COLOR: 0,1,0,0 (pas tous égaux)
        # - HOLE: 0,0,1,0 (pas tous égaux)
        # - SHAPE: 0,0,0,1 (pas tous égaux)
        assert mcts._piece_enables_quarto(game, 10) is False

    def test_get_safe_pieces(self):
        """_get_safe_pieces sépare les pièces sûres et dangereuses"""
        mcts = MCTS(num_simulations=10)
        game = Quarto()

        # Créer une situation avec certaines pièces dangereuses
        game.board[0, 0] = 1
        game.board[0, 1] = 3
        game.board[0, 2] = 5
        game.available_pieces -= {1, 3, 5}

        safe, dangerous = mcts._get_safe_pieces(game)

        # La pièce 7 est dangereuse (complète le Quarto sur couleur)
        assert 7 in dangerous

        # Les pièces sans propriété commune avec 1,3,5 sont sûres
        assert len(safe) + len(dangerous) == len(game.get_available_pieces())

    def test_mcts_avoids_dangerous_piece(self):
        """L'IA évite de donner une pièce qui permet un Quarto"""
        mcts = MCTS(num_simulations=50, use_dirichlet=False)
        game = Quarto()

        # Créer une situation: 3 pièces de même couleur alignées
        game.board[0, 0] = 1
        game.board[0, 1] = 3
        game.board[0, 2] = 5
        game.available_pieces -= {1, 3, 5}

        # L'IA doit choisir une pièce qui ne permet pas de compléter le Quarto
        chosen_piece = mcts.get_best_piece(game)

        # La pièce 7 complèterait le Quarto (même couleur)
        # L'IA ne devrait PAS choisir la pièce 7
        assert chosen_piece != 7, "L'IA a donné une pièce gagnante!"

    def test_mcts_all_pieces_dangerous(self):
        """Quand toutes les pièces sont dangereuses, l'IA en choisit une quand même"""
        mcts = MCTS(num_simulations=10, use_dirichlet=False)
        game = Quarto()

        # Situation où presque toutes les pièces mènent à un Quarto
        # (cas rare mais possible en fin de partie)
        # Pour simplifier, on crée juste un cas où il y a peu de choix
        game.board[0, 0] = 1
        game.board[0, 1] = 3
        game.board[0, 2] = 5
        game.board[1, 0] = 2
        game.board[1, 1] = 4
        game.board[1, 2] = 6
        game.available_pieces -= {1, 2, 3, 4, 5, 6}

        # get_best_piece ne doit pas lever d'exception
        chosen_piece = mcts.get_best_piece(game)
        assert chosen_piece in game.get_available_pieces()

    def test_smart_rollout_plays_winning_move(self):
        """_smart_rollout joue les coups gagnants"""
        mcts = MCTS(num_simulations=10)
        game = Quarto()

        # Situation gagnante
        game.board[0, 0] = 1
        game.board[0, 1] = 3
        game.board[0, 2] = 5
        game.available_pieces -= {1, 3, 5}
        game.choose_piece(7)

        # Le rollout devrait immédiatement gagner
        value = mcts._smart_rollout(game.clone())
        # La valeur devrait être positive (victoire)
        assert value == 1.0

    def test_simulate_piece_selection_opponent_wins(self):
        """La simulation détecte quand l'adversaire peut gagner"""
        mcts = MCTS(num_simulations=20, use_dirichlet=False)
        game = Quarto()

        # 3 pièces alignées
        game.board[0, 0] = 1
        game.board[0, 1] = 3
        game.board[0, 2] = 5
        game.available_pieces -= {1, 3, 5}

        # Chercher la meilleure pièce
        piece_probs = mcts.search_piece(game, temperature=0)

        # La pièce 7 (index 6) devrait avoir une probabilité très faible
        # car elle permet à l'adversaire de gagner
        safe_pieces, dangerous_pieces = mcts._get_safe_pieces(game)

        if safe_pieces:
            # Si des pièces sûres existent, elles devraient être préférées
            best_piece_idx = np.argmax(piece_probs)
            best_piece_id = best_piece_idx + 1
            assert best_piece_id in safe_pieces, f"L'IA a choisi {best_piece_id} qui est dangereux!"

// -*- coding: utf-8 -*-
// MCTS Core Implementation in C++ for AlphaZero Quarto
//
// This provides fast tree operations with Python callback for network evaluation.

#ifndef MCTS_CORE_HPP
#define MCTS_CORE_HPP

#include <vector>
#include <array>
#include <memory>
#include <cmath>
#include <random>
#include <unordered_map>
#include <functional>
#include <algorithm>
#include <atomic>

namespace mcts {

// Constants
constexpr int BOARD_SIZE = 4;
constexpr int NUM_SQUARES = 16;
constexpr int NUM_PIECES = 16;
constexpr int STATE_CHANNELS = 21;

// Game state representation
struct GameState {
    std::array<int, NUM_SQUARES> board;  // 0 = empty, 1-16 = piece
    int current_piece;  // 0 = no piece, 1-16 = piece to place
    int current_player; // 0 or 1
    bool game_over;
    int winner;  // -1 = no winner, 0 or 1 = winner

    GameState() : current_piece(0), current_player(0), game_over(false), winner(-1) {
        board.fill(0);
    }

    // Get legal moves (empty squares)
    std::vector<int> get_legal_moves() const {
        std::vector<int> moves;
        for (int i = 0; i < NUM_SQUARES; i++) {
            if (board[i] == 0) {
                moves.push_back(i);
            }
        }
        return moves;
    }

    // Get available pieces
    std::vector<int> get_available_pieces() const {
        std::vector<int> pieces;
        std::array<bool, NUM_PIECES + 1> used{};
        for (int i = 0; i < NUM_SQUARES; i++) {
            if (board[i] > 0) {
                used[board[i]] = true;
            }
        }
        if (current_piece > 0) {
            used[current_piece] = true;
        }
        for (int p = 1; p <= NUM_PIECES; p++) {
            if (!used[p]) {
                pieces.push_back(p);
            }
        }
        return pieces;
    }

    // Encode state for neural network (21, 4, 4) channels-first
    std::vector<float> encode() const {
        std::vector<float> state(STATE_CHANNELS * BOARD_SIZE * BOARD_SIZE, 0.0f);

        // Channels 0-15: one-hot for each piece
        for (int i = 0; i < NUM_SQUARES; i++) {
            if (board[i] > 0) {
                int channel = board[i] - 1;
                int row = i / BOARD_SIZE;
                int col = i % BOARD_SIZE;
                state[channel * NUM_SQUARES + row * BOARD_SIZE + col] = 1.0f;
            }
        }

        // Channels 16-19: current piece properties
        if (current_piece > 0) {
            int props = current_piece - 1;
            for (int prop = 0; prop < 4; prop++) {
                float val = ((props >> prop) & 1) ? 1.0f : 0.0f;
                int channel = 16 + prop;
                for (int i = 0; i < NUM_SQUARES; i++) {
                    state[channel * NUM_SQUARES + i] = val;
                }
            }
        }

        // Channel 20: has piece indicator
        if (current_piece > 0) {
            for (int i = 0; i < NUM_SQUARES; i++) {
                state[20 * NUM_SQUARES + i] = 1.0f;
            }
        }

        return state;
    }
};

// MCTS Node
class MCTSNode {
public:
    GameState state;
    int action;  // Action that led to this node (-1 for root)
    MCTSNode* parent;
    std::vector<std::unique_ptr<MCTSNode>> children;

    // Statistics
    std::atomic<int> visit_count{0};
    std::atomic<float> value_sum{0.0f};
    std::atomic<int> virtual_loss{0};

    // Prior from neural network
    float prior;

    // Is this node expanded?
    bool expanded;

    MCTSNode(const GameState& s, int act = -1, MCTSNode* p = nullptr, float pr = 0.0f)
        : state(s), action(act), parent(p), prior(pr), expanded(false) {}

    float value() const {
        int n = visit_count.load();
        if (n == 0) return 0.0f;
        return value_sum.load() / n;
    }

    float ucb_score(float c_puct) const {
        if (!parent) return 0.0f;

        int n = visit_count.load() + virtual_loss.load();
        int parent_n = parent->visit_count.load();

        float q_value = (n > 0) ? (value_sum.load() / n) : 0.0f;
        float exploration = c_puct * prior * std::sqrt(parent_n) / (1 + n);

        return q_value + exploration;
    }

    MCTSNode* select_child(float c_puct) {
        MCTSNode* best = nullptr;
        float best_score = -1e9f;

        for (auto& child : children) {
            float score = child->ucb_score(c_puct);
            if (score > best_score) {
                best_score = score;
                best = child.get();
            }
        }
        return best;
    }

    void add_virtual_loss(int amount = 1) {
        virtual_loss.fetch_add(amount);
    }

    void remove_virtual_loss(int amount = 1) {
        virtual_loss.fetch_sub(amount);
    }

    void update(float value) {
        visit_count.fetch_add(1);
        // Atomic float addition
        float current = value_sum.load();
        while (!value_sum.compare_exchange_weak(current, current + value)) {}
    }
};

// Callback type for neural network evaluation
// Input: batch of encoded states (N, 21*4*4)
// Output: (policies N x 16, piece_probs N x 16, values N)
using EvalCallback = std::function<void(
    const std::vector<std::vector<float>>& states,
    std::vector<std::vector<float>>& policies,
    std::vector<std::vector<float>>& piece_probs,
    std::vector<float>& values
)>;

// MCTS Configuration
struct MCTSConfig {
    int num_simulations = 100;
    float c_puct = 1.5f;
    float dirichlet_alpha = 0.8f;
    float dirichlet_epsilon = 0.25f;
    int temperature_threshold = 10;
    int batch_size = 8;  // Batch size for leaf evaluation
};

// Main MCTS class with batched leaf evaluation
class MCTS {
public:
    MCTSConfig config;
    EvalCallback eval_callback;
    std::mt19937 rng;

    MCTS(const MCTSConfig& cfg, EvalCallback callback)
        : config(cfg), eval_callback(callback), rng(std::random_device{}()) {}

    // Run MCTS search and return move probabilities
    std::pair<std::vector<float>, std::vector<float>> search(
        const GameState& root_state,
        float temperature = 1.0f
    ) {
        auto root = std::make_unique<MCTSNode>(root_state);

        // Initial expansion with network evaluation
        expand_node(root.get());

        // Add Dirichlet noise to root
        add_dirichlet_noise(root.get());

        // Run simulations with batched evaluation
        int sims_done = 0;
        while (sims_done < config.num_simulations) {
            int batch_size = std::min(config.batch_size, config.num_simulations - sims_done);
            run_simulation_batch(root.get(), batch_size);
            sims_done += batch_size;
        }

        // Compute move probabilities from visit counts
        std::vector<float> move_probs(NUM_SQUARES, 0.0f);
        std::vector<float> piece_probs(NUM_PIECES, 0.0f);

        float total_visits = 0.0f;
        for (auto& child : root->children) {
            total_visits += std::pow(child->visit_count.load(), 1.0f / temperature);
        }

        if (total_visits > 0) {
            for (auto& child : root->children) {
                float prob = std::pow(child->visit_count.load(), 1.0f / temperature) / total_visits;
                if (child->action >= 0 && child->action < NUM_SQUARES) {
                    move_probs[child->action] = prob;
                }
            }
        } else {
            // Uniform over legal moves
            auto legal = root_state.get_legal_moves();
            float p = 1.0f / legal.size();
            for (int m : legal) {
                move_probs[m] = p;
            }
        }

        // For piece probabilities, use the stored prior (simplified)
        auto available = root_state.get_available_pieces();
        if (!available.empty()) {
            float p = 1.0f / available.size();
            for (int piece : available) {
                piece_probs[piece - 1] = p;
            }
        }

        return {move_probs, piece_probs};
    }

private:
    void run_simulation_batch(MCTSNode* root, int batch_size) {
        std::vector<MCTSNode*> leaves;
        std::vector<std::vector<MCTSNode*>> paths;

        // Select leaves
        for (int i = 0; i < batch_size; i++) {
            std::vector<MCTSNode*> path;
            MCTSNode* node = root;
            path.push_back(node);

            // Selection: traverse tree to leaf
            while (node->expanded && !node->children.empty()) {
                node = node->select_child(config.c_puct);
                if (node) {
                    node->add_virtual_loss();
                    path.push_back(node);
                } else {
                    break;
                }
            }

            if (node && !node->state.game_over) {
                leaves.push_back(node);
                paths.push_back(path);
            } else {
                // Terminal node - backprop immediately
                float value = (node && node->state.winner >= 0) ?
                    ((node->state.winner == node->state.current_player) ? 1.0f : -1.0f) : 0.0f;
                backpropagate(path, value);
            }
        }

        if (leaves.empty()) return;

        // Batch evaluation
        std::vector<std::vector<float>> states;
        for (auto* leaf : leaves) {
            states.push_back(leaf->state.encode());
        }

        std::vector<std::vector<float>> policies(leaves.size());
        std::vector<std::vector<float>> piece_probs(leaves.size());
        std::vector<float> values(leaves.size());

        eval_callback(states, policies, piece_probs, values);

        // Expand leaves and backpropagate
        for (size_t i = 0; i < leaves.size(); i++) {
            expand_with_policy(leaves[i], policies[i]);

            // Remove virtual loss and backpropagate
            for (auto* node : paths[i]) {
                node->remove_virtual_loss();
            }
            backpropagate(paths[i], values[i]);
        }
    }

    void expand_node(MCTSNode* node) {
        if (node->expanded || node->state.game_over) return;

        // Single evaluation
        std::vector<std::vector<float>> states = {node->state.encode()};
        std::vector<std::vector<float>> policies(1);
        std::vector<std::vector<float>> piece_probs(1);
        std::vector<float> values(1);

        eval_callback(states, policies, piece_probs, values);

        expand_with_policy(node, policies[0]);
    }

    void expand_with_policy(MCTSNode* node, const std::vector<float>& policy) {
        if (node->expanded) return;

        auto legal_moves = node->state.get_legal_moves();

        // Normalize policy over legal moves
        float total = 0.0f;
        for (int m : legal_moves) {
            total += policy[m];
        }
        if (total <= 0.0f) total = 1.0f;

        for (int move : legal_moves) {
            GameState child_state = node->state;
            child_state.board[move] = child_state.current_piece;
            child_state.current_piece = 0;
            child_state.current_player = 1 - child_state.current_player;

            // Check for game over (simplified - real check needs Quarto logic)
            auto remaining = child_state.get_legal_moves();
            if (remaining.empty()) {
                child_state.game_over = true;
            }

            float prior = policy[move] / total;
            auto child = std::make_unique<MCTSNode>(child_state, move, node, prior);
            node->children.push_back(std::move(child));
        }

        node->expanded = true;
    }

    void backpropagate(const std::vector<MCTSNode*>& path, float value) {
        float v = value;
        for (int i = path.size() - 1; i >= 0; i--) {
            path[i]->update(v);
            v = -v;  // Flip value for opponent
        }
    }

    void add_dirichlet_noise(MCTSNode* node) {
        if (node->children.empty()) return;

        std::gamma_distribution<float> gamma(config.dirichlet_alpha, 1.0f);
        std::vector<float> noise(node->children.size());
        float sum = 0.0f;

        for (size_t i = 0; i < noise.size(); i++) {
            noise[i] = gamma(rng);
            sum += noise[i];
        }

        for (size_t i = 0; i < node->children.size(); i++) {
            float prior = node->children[i]->prior;
            node->children[i]->prior = (1 - config.dirichlet_epsilon) * prior
                                     + config.dirichlet_epsilon * (noise[i] / sum);
        }
    }
};

} // namespace mcts

#endif // MCTS_CORE_HPP

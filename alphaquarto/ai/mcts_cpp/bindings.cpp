// -*- coding: utf-8 -*-
// Python bindings for MCTS C++ implementation using pybind11

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include "mcts_core.hpp"

namespace py = pybind11;

// Wrapper class to handle Python callback properly
class MCTSWrapper {
public:
    mcts::MCTSConfig config;
    py::object py_callback;

    MCTSWrapper(int num_simulations, float c_puct, float dirichlet_alpha,
                float dirichlet_epsilon, int batch_size, py::object callback)
        : py_callback(callback) {
        config.num_simulations = num_simulations;
        config.c_puct = c_puct;
        config.dirichlet_alpha = dirichlet_alpha;
        config.dirichlet_epsilon = dirichlet_epsilon;
        config.batch_size = batch_size;
    }

    std::pair<std::vector<float>, std::vector<float>> search(
        py::array_t<int> board,
        int current_piece,
        int current_player,
        bool game_over,
        int winner,
        float temperature
    ) {
        // Create game state from Python inputs
        mcts::GameState state;
        auto buf = board.request();
        int* ptr = static_cast<int*>(buf.ptr);
        for (int i = 0; i < mcts::NUM_SQUARES; i++) {
            state.board[i] = ptr[i];
        }
        state.current_piece = current_piece;
        state.current_player = current_player;
        state.game_over = game_over;
        state.winner = winner;

        // Create eval callback that calls Python
        auto eval_callback = [this](
            const std::vector<std::vector<float>>& states,
            std::vector<std::vector<float>>& policies,
            std::vector<std::vector<float>>& piece_probs,
            std::vector<float>& values
        ) {
            py::gil_scoped_acquire acquire;

            // Convert states to numpy array
            size_t batch_size = states.size();
            size_t state_size = mcts::STATE_CHANNELS * mcts::NUM_SQUARES;

            py::array_t<float> states_np({batch_size, state_size});
            auto states_buf = states_np.request();
            float* states_ptr = static_cast<float*>(states_buf.ptr);

            for (size_t i = 0; i < batch_size; i++) {
                for (size_t j = 0; j < state_size; j++) {
                    states_ptr[i * state_size + j] = states[i][j];
                }
            }

            // Reshape to (batch, 21, 4, 4)
            states_np = states_np.reshape({static_cast<py::ssize_t>(batch_size),
                                           mcts::STATE_CHANNELS,
                                           mcts::BOARD_SIZE,
                                           mcts::BOARD_SIZE});

            // Call Python callback
            py::object result = py_callback(states_np);

            // Parse results
            py::tuple result_tuple = result.cast<py::tuple>();
            py::array_t<float> policies_np = result_tuple[0].cast<py::array_t<float>>();
            py::array_t<float> pieces_np = result_tuple[1].cast<py::array_t<float>>();
            py::array_t<float> values_np = result_tuple[2].cast<py::array_t<float>>();

            auto pol_buf = policies_np.request();
            auto piece_buf = pieces_np.request();
            auto val_buf = values_np.request();

            float* pol_ptr = static_cast<float*>(pol_buf.ptr);
            float* piece_ptr = static_cast<float*>(piece_buf.ptr);
            float* val_ptr = static_cast<float*>(val_buf.ptr);

            policies.resize(batch_size);
            piece_probs.resize(batch_size);
            values.resize(batch_size);

            for (size_t i = 0; i < batch_size; i++) {
                policies[i].resize(mcts::NUM_SQUARES);
                piece_probs[i].resize(mcts::NUM_PIECES);

                for (int j = 0; j < mcts::NUM_SQUARES; j++) {
                    policies[i][j] = pol_ptr[i * mcts::NUM_SQUARES + j];
                }
                for (int j = 0; j < mcts::NUM_PIECES; j++) {
                    piece_probs[i][j] = piece_ptr[i * mcts::NUM_PIECES + j];
                }
                values[i] = val_ptr[i];
            }
        };

        // Run MCTS
        mcts::MCTS mcts_engine(config, eval_callback);

        // Release GIL during C++ computation
        py::gil_scoped_release release;
        auto result = mcts_engine.search(state, temperature);

        return result;
    }
};

PYBIND11_MODULE(mcts_cpp, m) {
    m.doc() = "Fast MCTS implementation in C++ for AlphaZero Quarto";

    py::class_<MCTSWrapper>(m, "MCTS")
        .def(py::init<int, float, float, float, int, py::object>(),
             py::arg("num_simulations") = 100,
             py::arg("c_puct") = 1.5f,
             py::arg("dirichlet_alpha") = 0.8f,
             py::arg("dirichlet_epsilon") = 0.25f,
             py::arg("batch_size") = 8,
             py::arg("eval_callback") = py::none())
        .def("search", &MCTSWrapper::search,
             py::arg("board"),
             py::arg("current_piece"),
             py::arg("current_player"),
             py::arg("game_over"),
             py::arg("winner"),
             py::arg("temperature") = 1.0f,
             "Run MCTS search and return move probabilities");

    m.attr("NUM_SQUARES") = mcts::NUM_SQUARES;
    m.attr("NUM_PIECES") = mcts::NUM_PIECES;
    m.attr("STATE_CHANNELS") = mcts::STATE_CHANNELS;
}

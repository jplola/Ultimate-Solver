import torch
import torch.nn as nn
import torch.nn.functional as F
from Simple_Simulator import SimpleTicTacToe
from monte_carlo_tree_search import SearchTreeNode,make_a_choice
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np





def generate_training_data(num_games=100,parallelise = False):
    game_states = []
    probabilities_list = []
    print('Generating training data...')
    for i in range(num_games):
        game = SimpleTicTacToe()
        root = SearchTreeNode(game, sim_class=SimpleTicTacToe)

        best_move, root = make_a_choice(root, my_move=None, deepness=243, simulations=1, sim_class=SimpleTicTacToe,
                                        batch_size=100,parallelise=parallelise)

        state, probabilities = root.get_state_probabilities_simple(root)

        state *= root.state.current_player

        game_states.append(state)
        probabilities_list.append(probabilities)

        continue_loop = True
        count = 0
        while continue_loop:
            if count % 2 == 0:
                best_move, root = make_a_choice(root, my_move=best_move, deepness=243, simulations=1,
                                                sim_class=SimpleTicTacToe, batch_size=100, c=np.sqrt(2),parallelise=parallelise)
            else:
                best_move, root = make_a_choice(root, my_move=best_move, deepness=243, simulations=1,
                                                sim_class=SimpleTicTacToe, batch_size=100, c=10,parallelise=parallelise)

            if best_move != 'Terminal':
                state, probabilities = root.get_state_probabilities_simple(root)
                game_states.append(state)
                probabilities_list.append(probabilities)
            else:
                continue_loop = False
            if root.is_terminal:
                continue_loop = False
            count += 1
        print(f'game {i+1}/{num_games} simulated')


    return np.array(game_states),np.array(probabilities_list)


def create_tictactoe_cnn():
    # Input shape: 3x3x1 (single channel for the board state)
    inputs = layers.Input(shape=(3, 3, 1))

    # First convolutional layer
    x = layers.Conv2D(32, (2, 2), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)

    # Second convolutional layer
    x = layers.Conv2D(64, (2, 2), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)

    # Final convolutional layer (no activation)
    x = layers.Conv2D(1, (1, 1), padding='same')(x)

    # Reshape to flatten for softmax
    x = layers.Reshape((9,))(x)

    # Apply softmax
    x = layers.Softmax()(x)

    # Reshape back to 3x3
    outputs = layers.Reshape((3, 3))(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',  # Changed from binary_crossentropy
                  metrics=['accuracy'])

    return model


def prepare_data(boards, probabilities):
    """
    Prepare the data for training

    Parameters:
    boards: numpy array of shape (N, 3, 3) containing board positions with -1, 0, 1
    probabilities: numpy array of shape (N, 3, 3) containing move probabilities

    Returns:
    X: numpy array of shape (N, 3, 3, 1) - input for the model
    y: numpy array of shape (N, 3, 3) - target probabilities
    """
    # Add channel dimension to boards
    X = boards.reshape(-1, 3, 3, 1)
    # Probabilities are already in the correct shape
    y = probabilities
    return X, y


def train_model(model, boards, probabilities, epochs=10, batch_size=32, validation_split=0.2):
    """
    Train the model using provided board positions and probability distributions

    Parameters:
    model: the CNN model
    boards: numpy array of shape (N, 3, 3) containing board positions
    probabilities: numpy array of shape (N, 3, 3) containing move probabilities
    epochs: number of training epochs
    batch_size: batch size for training
    validation_split: fraction of data to use for validation
    """
    # Prepare data
    X, y = prepare_data(boards, probabilities)

    # Train the model
    history = model.fit(
        X, y,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        verbose=1
    )

    return history


def get_move_probabilities(model, board):
    """Get move probabilities for a single board position"""
    # Add batch and channel dimensions
    input_board = board.reshape(1, 3, 3, 1)
    predictions = model.predict(input_board, verbose=0)
    return predictions[0]  # Remove batch dimension





class CONVOLmodel:
    def __init__(self,num_games_training=1000):
        # Create example data (replace this with your actual data)
        states, probabilities = generate_training_data(num_games=num_games_training, parallelise=False)

        train_boards, train_probabilities = prepare_data(states, probabilities)

        # Create and train the model
        model = create_tictactoe_cnn()

        # Train the model with your data
        history = train_model(
            model,
            train_boards,
            train_probabilities,
            epochs=20,
            batch_size=32
        )

        self.model = model

    def next_move(self,state):

        probabilities = get_move_probabilities(self.model, state.board)
        possible_prob = np.argmax(probabilities)
        if possible_prob in state.legal_moves:
            state.step_forward(possible_prob)
            state.current_player *= -1


"""game = SimpleTicTacToe()
model = CONVOLmodel()

model.next_move(game)"""
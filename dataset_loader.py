from torch.utils.data import Dataset
from torchvision import transforms
import torch

import chess
import pandas as pd
import numpy as np

import os

from tqdm import tqdm

class ChessDataset(Dataset):

    def __init__(self, dataset_path="./dataset/", save_compressed=True, encoding_type="one_hot"):

        self.dataset_path = dataset_path
        self.save_compressed = save_compressed

        self.already_loaded = False

        self.csv_dataset = os.path.join(self.dataset_path, "games.csv")
        self.compressed_dataset_X = os.path.join(self.dataset_path,
                                                 "games_compressed_{}.npz".format(encoding_type))
        self.compressed_dataset_Y = os.path.join(self.dataset_path,
                                                 "values_compressed_{}.npz".format(encoding_type))

        if os.path.isfile(self.compressed_dataset_X) and os.path.isfile(self.compressed_dataset_Y):
            self.dataset_X = np.load(self.compressed_dataset_X)['arr_0']
            self.dataset_Y = np.load(self.compressed_dataset_Y)['arr_0']
            self.already_loaded = True
        else:
            self.original_dataset = pd.read_csv(self.csv_dataset)
            self.dataset_X = []
            self.dataset_Y = []

        if encoding_type == "one_hot":
            self.chess_dict = {
                'p': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'P': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                'n': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'N': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                'b': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'B': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                'r': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                'R': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                'q': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                'Q': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                'k': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                'K': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                '.': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            }
        else:
            self.chess_dict = {
                'p': [1,0,0,0],
                'P': [0,1,0,0],
                'n': [1,1,0,0],
                'N': [0,0,1,0],
                'b': [1,0,1,0],
                'B': [0,1,1,0],
                'r': [1,1,1,0],
                'R': [0,0,0,1],
                'q': [1,0,0,1],
                'Q': [0,1,0,1],
                'k': [1,1,0,1],
                'K': [0,0,1,1],
                '.': [0,0,0,0]
            }

    def __getitem__(self, index):
        return torch.FloatTensor(self.dataset_X[index]), \
               torch.FloatTensor(self.dataset_Y[index])

    def __len__(self):
        return len(self.dataset_Y)

    def _convert_string_to_matrix(self, board):
        converted_board = []
        for rows in board.split("\n"):
            converted_board.append(rows.split(" "))
        return converted_board

    def _convert_boards_to_one_hot(self, board):
        one_hot_board = []
        for row in board:
            terms = []
            for element in row:
                terms.append(self.chess_dict[element])
            one_hot_board.append(terms)
        return one_hot_board

    def convert_games(self):
        """
        Convert the games into a more appropriate one-hot enconding.
        :return: None
        """

        if self.already_loaded:
            return

        self.original_dataset = self.original_dataset[self.original_dataset["victory_status"] == "mate"]

        for index, game in tqdm(self.original_dataset.iterrows(), total=self.original_dataset.shape[0]):

            # Get only the final 8 moves of each game.
            moves = game["moves"].split(" ")[:-20]
            total_moves = len(moves)
            game_winner = 1 if game["winner"] == "white" else -1

            board = chess.Board()
            move_counter = 0
            for move in moves:
                board.push_san(move)

                # Skip the first 10 moves (5 for each). This should ensure better
                # convergence since the first moves are usually not informative.
                if move_counter > 9:
                    value = game_winner * (move_counter / total_moves)
                    board_cnv = self._convert_string_to_matrix(str(board))
                    board_cnv = self._convert_boards_to_one_hot(board_cnv)
                    self.dataset_X.append(board_cnv)
                    self.dataset_Y.append(np.array([value]))

                move_counter += 1

        self.dataset_X = np.array(self.dataset_X)
        self.dataset_Y = np.array(self.dataset_Y)

        # Save the results
        if self.save_compressed:
            np.savez_compressed(self.compressed_dataset_X, self.dataset_X)
            np.savez_compressed(self.compressed_dataset_Y, self.dataset_Y)


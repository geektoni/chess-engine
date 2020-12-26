from torch.utils.data import Dataset
from torchvision import transforms
import torch

import chess
import pandas as pd
import numpy as np

import os

from tqdm import tqdm

class ChessDataset(Dataset):

    def __init__(self, dataset_path="./dataset/games.csv"):

        self.original_dataset = pd.read_csv(dataset_path)
        self.already_loaded = False

        if os.path.isfile("./dataset/games_compressed.npz") and os.path.isfile("./dataset/values_compressed.npz"):
            self.dataset_X = np.load("./dataset/games_compressed.npz")['arr_0']
            self.dataset_Y = np.load("./dataset/values_compressed.npz")['arr_0']
            self.already_loaded = True
        else:
            self.dataset_X = []
            self.dataset_Y = []

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

        self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        return self.dataset_X[index], \
               self.dataset_Y[index]

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

            moves = game["moves"].split(" ")
            total_moves = len(moves)
            game_winner = 1 if game["winner"] == "white" else -1

            board = chess.Board()
            move_counter = 0
            for move in moves:
                board.push_san(move)
                value = game_winner * (move_counter/total_moves)
                board_cnv = self._convert_string_to_matrix(str(board))
                board_cnv = self._convert_boards_to_one_hot(board_cnv)

                board_cnv = np.array(board_cnv).reshape(8,8,12)
                self.dataset_X.append(board_cnv)
                self.dataset_Y.append(np.array([value]))
                move_counter += 1

        self.dataset_X = np.array(self.dataset_X)
        self.dataset_Y = np.array(self.dataset_Y)

        np.savez_compressed("games_compressed", self.dataset_X)
        np.savez_compressed("values_compressed", self.dataset_Y)


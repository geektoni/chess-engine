#!/usr/bin/env python
# coding: utf-8

import chess
import torch
from model import ChessEngine, ChessEngine2
from minimax import ActorChess

# Load the model
chess_engine = ChessEngine2()
chess_engine.load_state_dict(torch.load("./chess_engine3.pth"))

# Renact the Scholar's mate and try to evaluate the
# position with the network. 

board = chess.Board()

board.push_san("e4")
board.push_san("e5")
board.push_san("Qh5")
board.push_san("Nc6")
#board.push_san("Bc4")
#board.push_san("Nf6")
#board.push_san("Qxf7")

actor = ActorChess(chess_engine, board, depth=2)
value, next_move = actor.compute_next_move()

print(value, next_move)





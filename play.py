import chess.pgn
import datetime
import chess.engine

from chess_board import display
from time import sleep

from model import ChessEngine
from minimax import ActorChess

import torch

from argparse import ArgumentParser


if __name__ == "__main__":

    # Set up the argument parser
    parser = ArgumentParser()
    parser.add_argument("--load-model", type=str, help="Location where to load the trained model", default="./models/trained.pth")
    parser.add_argument("--encoding", type=str, help="Select the encoding used for the chess positions (one_hot "
                                                     "or binary)", default="one_hot")
    parser.add_argument("--depth", type=int, help="Depth of B.C.E search tree", default=2)

    # Parse the arguments
    args = parser.parse_args()

    model_file = args.load_model
    encoding = args.encoding
    tree_depth = args.depth

    engine = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")

    board = chess.Board()

    # Load the model
    chess_engine = ChessEngine(encoding_type=encoding)
    chess_engine.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
    actor = ActorChess(chess_engine, board, depth=tree_depth)

    movehistory = []
    game = chess.pgn.Game()
    game.headers["Event"] = "Test Match Stockfish vs B.C.E."
    game.headers["Site"] = "Internet"
    game.headers["Date"] = str(datetime.datetime.now().date())
    game.headers["Round"] = 1
    game.headers["White"] = "B.C.E"
    game.headers["Black"] = "Stockfish9"

    # Start the game
    display.start(board.fen())

    while not board.is_game_over(claim_draw=True):
        if board.turn:
            print("Play B. C. E")
            value, next_move = actor.compute_next_move()
            board.push_san(str(next_move))
        else:
            print("Play Stockfish")
            result = engine.play(board, chess.engine.Limit(time=0.1))
            movehistory.append(result.move)
            board.push(result.move)

        display.update(board.fen())
        sleep(0.5)

    game.add_line(movehistory)
    game.headers["Result"] = str(board.result(claim_draw=True))
    print(game)
    print(game, file=open("test.pgn", "w"), end="\n\n")
    display.terminate()

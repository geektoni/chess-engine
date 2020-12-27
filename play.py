import chess.pgn
import datetime
import chess.engine

global gameboard

#from chessboard import display
#from time import sleep

from model import ChessEngine2
from minimax import ActorChess

import torch

if __name__ == "__main__":

    engine = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")

    board = chess.Board()

    # Load the model
    chess_engine = ChessEngine2()
    chess_engine.load_state_dict(torch.load("./chess_engine3.pth"))
    actor = ActorChess(chess_engine, board, depth=2)

    movehistory = []
    game = chess.pgn.Game()
    game.headers["Event"] = "Example"
    game.headers["Site"] = "Linz"
    game.headers["Date"] = str(datetime.datetime.now().date())
    game.headers["Round"] = 1
    game.headers["White"] = "B.C.E"
    game.headers["Black"] = "Stockfish9"

    # Start the game
    #display.start(board.fen())

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

        #display.update(board.fen())
        #sleep(1)

    game.add_line(movehistory)
    game.headers["Result"] = str(board.result(claim_draw=True))
    print(game)
    print(game, file=open("test.pgn", "w"), end="\n\n")
    #display.terminate()
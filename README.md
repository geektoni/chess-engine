# DLC (Deep Learning for Chess)

This repository contains a simple deep network engine which can be used to play chess in a similar fashion to other much popular engines (e.g., Stockfish). 
The engine uses a search procedure (minimax search + alpha/beta pruning + position tables) to decide the next good move by looking at the current board setup. 
A deep network is used to evaluate the chess board and to generate an estimate about how good is the current position for the White or Black player. The deep
networks returns an estimate between -1 and 1. The positive value means that White is winning, while the negative value means that Black is winning.

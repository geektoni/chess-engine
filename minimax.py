import chess

import torch

class ActorChess:

    def __init__(self, model, board):

        self.model = model
        self.board = board
        self.depth = 2

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

    def _convert_board(self, board):
        matrix = self._convert_string_to_matrix(str(board))
        return torch.FloatTensor(self._convert_boards_to_one_hot(matrix))

    def _build_tree(self, node, board, maximizer):

        childs = []
        for next_move in node["next_moves"]:

            # Play one move
            board.push_san(str(next_move))

            # Get an input for the neural network
            board_input = self._convert_board(board)
            board_input = torch.reshape(board_input, (1, 8, 8, 12))

            # Create a new node and append it to
            # the childs list
            tmp_node = {
                "parent": node,
                "move": str(next_move),
                "next_moves": list(board.legal_moves),
                "maximizer": not maximizer,
                "childs": [],
                "terminal": len(list(self.board.legal_moves)) == 0 or self.board.is_game_over(),
                "value": self.model(board_input).item(),
                "best_value": 0.0
            }
            childs.append(tmp_node)

            # Unmake the last move
            board.pop()

        node["childs"] += childs

    def compute_next_move(self):

        # Get an input for the neural network
        board_input = self._convert_board(self.board)
        board_input = torch.reshape(board_input, (1, 8, 8, 12))

        # Create a new node and append it to
        # the childs list
        root = {
            "parent": None,
            "move": None,
            "next_moves": list(self.board.legal_moves),
            "maximizer": True,
            "childs": [],
            "terminal": len(list(self.board.legal_moves)) == 0 or self.board.is_game_over(),
            "value": self.model(board_input).item(),
            "best_value": 0.0
        }

        # Compute the Alpha-beta algorithm
        self._alpha_beta(root, self.depth, -1, 1, True)

        # Then, for each child node we get the one with the highest value
        max_node = -1
        move = ""
        for child in root["childs"]:
            if child["best_value"] > max_node:
                max_node = child["best_value"]
                move = child["move"]

        return max_node, move

    def _alpha_beta(self, node, depth, alpha, beta, maximizing):

        if node["terminal"] or depth == 0:
            node["best_value"] = node["value"]
            return node["value"]

        if maximizing:

            # Populate the node with its childs
            if len(node["childs"]) == 0:
                self._build_tree(node, self.board, maximizing)

            # Iterate over all the childs
            value = -1
            for child in node["childs"]:
                self.board.push_san(str(child["move"]))
                node["best_value"] = max(node["best_value"], self._alpha_beta(child, depth-1, alpha, beta, not maximizing))
                alpha = max(alpha, value)
                self.board.pop()
                if alpha >= beta:
                    break

            return node["best_value"]
        else:

            # Populate the node with its childs
            if len(node["childs"]) == 0:
                self._build_tree(node, self.board, maximizing)

            # Iterate over all the childs
            value = 1
            for child in node["childs"]:
                self.board.push_san(str(child["move"]))
                node["best_value"] = min(node["best_value"], self._alpha_beta(child, depth - 1, alpha, beta, not maximizing))
                beta = min(beta, value)
                self.board.pop()
                if beta >= alpha:
                    break

            return node["best_value"]


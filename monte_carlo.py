import threading

from mctspy.games.common import TwoPlayersAbstractGameState
from mctspy.tree.search import MonteCarloTreeSearch
from mctspy.tree.nodes import TwoPlayersGameMonteCarloTreeSearchNode
from game import Game

class HnefataflState(TwoPlayersAbstractGameState):
    def __init__(self, game, turn):
        self.game = game  # instance of the Game class
        self.turn = turn

    @property
    def next_to_move(self):
        return self.turn

    @property
    def game_result(self):
        if self.game.is_over():
            return self.game.get_reward(self.game.turn)  # Return 1 for player 1 win, -1 for player 2 win, 0 for draw/tie.
        return None  # game is not over

    def is_game_over(self):
        return self.game.is_over()

    def move(self, move):
        # Make a copy of the game to avoid altering the original state
        new_game = Game()
        new_game.board = [row[:] for row in self.game.board]
        new_game.turn = self.turn
        new_game.place_piece(move[2], move[3], move[0], move[1], new_game.turn)
        return HnefataflState(new_game, 3 - self.turn)

    def get_legal_actions(self):
        return self.game.get_possible_moves()

    def __repr__(self):
        return f"{self.game.board}"


def msim():
    while not app.is_over():
        board_state = HnefataflState(app, app.turn)
        last_board = board_state.game.board
        root = TwoPlayersGameMonteCarloTreeSearchNode(state=board_state)
        mcts = MonteCarloTreeSearch(root)
        best_action = mcts.best_action(10)
        move = []
        for i in range(len(last_board)):
            for j in range(len(last_board[0])):
                old_piece = last_board[i][j]
                new_piece = best_action.state.game.board[i][j]
                if old_piece != new_piece:
                    move.append((i, j, new_piece))

        if move and len(move) == 2 and app.turn == 1:
            if move[0][2] == 0:
                old_place = [move[0][0], move[0][1]]
                new_place = [move[1][0], move[1][1]]
            else:
                old_place = [move[1][0], move[1][1]]
                new_place = [move[0][0], move[0][1]]
            # Apply the best action to the game
            app.place_piece(new_place[0], new_place[1], old_place[0], old_place[1], app.turn)
            app.bot_action = move


# Instantiate the game
app = Game()
app.setup_board()

threading.Thread(target=msim).start()

app.play_game()

# Run MCTS


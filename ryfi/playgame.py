"""
Runs trained models on a game

:author: ryfi
"""

import os
import time

from game import Game
from ryfi.deep_learning import DQNAgent

game = Game()

state_size = 11
# comes from...
output_size = 14641

# imports bots
agent1 = DQNAgent(state_size, output_size, epsilon = 0, temperature=.3)
if os.path.exists("saved_models/agent1.pth"):
    agent1.load_model("saved_models/agent1.pth")
agent2 = DQNAgent(state_size, output_size, epsilon = 0, temperature=.3)
if os.path.exists("saved_models/agent2.pth"):
    agent2.load_model("saved_models/agent2.pth")

# sets up input functions
def agent_1_move(game):
    action = agent1.act(game.get_state_representation(), game)


    new_col = action[3]
    new_row = action[2]
    curr_row = action[0]
    curr_col = action[1]

    return curr_row, curr_col, new_row, new_col

def agent_2_move(game):
    action = agent2.act(game.get_state_representation(), game)

    new_col = action[3]
    new_row = action[2]
    curr_row = action[0]
    curr_col = action[1]

    return curr_row, curr_col, new_row, new_col

# true if the bot is being used
bots = [True, False]

# adds bots to the game
if bots[0]:
    game.bots[1] = agent_1_move
else:
    game.bots[1] = None

if bots[1]:
    game.bots[2] = agent_2_move
else:
    game.bots[2] = None


game.play_game()
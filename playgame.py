"""
Runs trained models on a game

:author: ryfi
"""

import os
from game import Game
from deep_learning import DQNAgent

game = Game()

state_size = 11 * 11
# comes from...
output_size = 14641
game.botone = DQNAgent(state_size, output_size, epsilon = 0)
game.bottwo = DQNAgent(state_size, output_size, epsilon = 0)
if os.path.exists("saved_models/agent1.pth"):
    game.botone.load_model("saved_models/agent1.pth")
if os.path.exists("saved_models/agent2.pth"):
    game.bottwo.load_model("saved_models/agent2.pth")

game.play_game()
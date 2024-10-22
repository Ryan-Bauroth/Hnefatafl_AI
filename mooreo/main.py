'''
playing the game
'''

import train_ori
from game_ori import *

bot = train_ori.ori_model()
app = Game(bot1=bot)#, bot2=bot) # bot2=bot
app.play_game()
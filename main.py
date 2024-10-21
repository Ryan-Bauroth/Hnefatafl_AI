import train_ori
from game import *

if 1:
    trainer = train_ori.Trainer()
    trainer.train()
else:
    bot = train_ori.ori_model()
    app = Game(bot1=None, bot2=bot)
    app.play_game()
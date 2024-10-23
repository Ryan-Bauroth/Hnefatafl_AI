**Hnefatafl Bot**
- My code is in the mooreo directory
- Run the game in main.py with option to run one or both sides with my bot
- temp_train.py is where to run training, which calls training in train_ori.py
- checkpoint.pth (not pushed to GitHub) stores data from training
- My training algorithm uses Deep Q learning with convolutional neural networks to identify moves that optimize my reward system that incentivizes bringing the bot closer to victory
- I feed data from a tree of possible next moves into my DQN to pick the best move from a combination of rewarding simulated future moves and rewarding DQN future states
- This game is a collaboration with Ryan, using a similar game.py file, but due to a small difference in merging, we now have separate game files (mine is game_ori.py in the mooreo directory)
# Hnefatafl AI

## The Game

Hnefatafl is an asymmetric game where the Attackers (Black) try to completely surround the King (Yellow) before the Defenders (White) manage to move the King to one of the corners. Either team can capture an opponent's piece by surrounding it on both sides. All pieces, including the King, move like rooks in chess. Only the King can occupy the corner squares or the square in the middle.

<img width="300" alt="Screenshot 2024-10-22 at 11 25 11 AM" src="https://github.com/user-attachments/assets/f97977bb-b64a-4932-be78-2fd4d616055c">

[More Rules](https://www.mastersofgames.com/rules/hnefatafl-viking-chess-rules.htm?srsltid=AfmBOopAaKbxPVKCWk0I2xJ8YuU_bzgFB6CYwSY_y9bsbk1gLby-hOZR)

## Description

The goal of this project is to build the strongest Hnefatafl AI. Ultimately, I hope to beat Ori in a competition between our AIs.

For this project, I use two models (one for the Attackers, one for the Defenders) and play them against each other to train. This approach allows one model to develop new strategies while the other learns to counter them.

### Q-Values

The first method I implemented for my Hnefatafl AI was a Q-Learning Algorithm. This algorithm creates a function Q(s, a) where *s* represents the state of the board and *a* represents an action, predicting the reward for that action over time. To build this function Q, I start with a high epsilon, which causes the bot to take highly randomized actions. Over time, this builds a dataset for learning, which I use to update the Q function. As my AI trains over more episodes, it acts less randomly and starts using the learned Q-values. This allows the model to test its Q function and adjust it based on the success of the moves it makes. To further encourage exploration at the beginning, I have both bots make a random number of random moves (between 0 and 5).

### Rewards

To implement my Q-Learning Algorithm, I designed a reward system. After some trial and error, I settled on the following rewards:
- King moves closer to the corner (Defender: +0.005)
- King moves further from the corner (Defender: -0.005)
- Piece captured (Capturing side: +0.02, Captured side: -0.02)
- Piece moves next to the King (Attacker: +0.015, Defender: -0.015)
- Win: +1
- Loss: -1

### Monte Carlo Simulation

While training my model with the Q-Learning Algorithm, I found that the model wasn't learning effectively. Two key issues emerged: the game was heavily imbalanced, with the Defenders winning nearly every time, and the bot failed to recognize imminent one-move wins. To improve my model, I added a Monte Carlo Simulation. This simulation takes the current board state (the root) and simulates random games from this state until a win or loss occurs (trees). It then uses the results to assess the strength of different moves. For example, if moving the King towards the edge of the board resulted in many wins during simulations, that move would receive a high rating. Conversely, if moving the King to a spot surrounded by three opposing pieces resulted in frequent losses, that move would be rated poorly. However, this approach proved inefficient due to lengthy training times and still struggled with one-move wins.

### Hard-Coding Victories

Ultimately, I chose to hard-code the robot to recognize and take advantage of one-move wins. Without this implementation, the randomness of choosing a winning move meant that good moves were often not rewarded. Although I believe the model could learn this given enough training, time and processing constraints led me to adopt this solution. I continue to use the Monte Carlo Simulation alongside the Q-Learning Algorithm for all other moves.

### HPGN

During training, I found it helpful to review how certain games played out to make decisions about future model development. When training without the Monte Carlo Simulation, games played very quickly and were hard to follow. To address this, I created a file format called HPGN (Hnefatafl Portable Game Notation) that stores data for each game in a specific format, allowing me to review matches whenever needed.

## Use Instructions

In order to use this repo, simpily run the deep_learning.py file to train the model. Then run the playgame.py file to play a computer. You also can run the game.py file in order to play a two player version of Hnefatafl.

## Future Changes

One major improvement I want to implement is an easier way for the model to identify the King piece from other pieces. Currently, I don’t believe my model fully understands the importance of the King because I haven’t included piece types in the 'action' definition. By adding this information and accounting for it in the model, I believe the AI would perform better.

Additionally, I would like to add a 'back' button to my HPGN playthrough file viewer. In its current form, you can only move forward through the matches, but not backward.


## Tools

I utilized ChatGPT alongside JetBrains AI to help with the coding and documenting process. All code generated by AI is labeled in the comments. All method comments are generated by JetBrains AI.

I also used the Threading, mctspy, and torch libraries.

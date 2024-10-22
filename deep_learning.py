"""
Deep Q-Learning Agent
:author: ryfi
"""

import gc
import os
from collections import deque
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random
import threading
from hpgn import HPGN


# Neural network model for the Q-learning agent
# Generated by AI
class DQNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.ln1 = nn.LayerNorm(128)  # Use Layer Normalization
        self.fc2 = nn.Linear(128, 64)
        self.ln2 = nn.LayerNorm(64)   # Use Layer Normalization
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)


# Q-learning agent class
# Generated by AI, then edited by me
class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, lr=0.001, batch_size=64, memory_size=7500, epsilon=1.0,
                 epsilon_decay=0.9975, epsilon_min=0.01, temperature=1.0):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # Discount factor => short term or long term rewards
        self.lr = lr  # Learning rate => small = slower but more stable
        self.batch_size = batch_size # how many runs
        self.memory = deque(maxlen=memory_size) # stores the memory of previous actions
        self.epsilon = epsilon  # Exploration rate => randomness rate
        self.epsilon_decay = epsilon_decay  # Decay rate of epsilon
        self.epsilon_min = epsilon_min
        self.temperature = temperature

        # Initialize the Q-network
        self.network = DQNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

    def act(self, state, env, override_random=False):
        """
        :param state: The current state of the environment which the agent will use to decide the next action.
        :param env: The environment object which provides necessary methods like getting possible moves.
        :param override_random: A boolean flag to override the epsilon-greedy policy and choose a random action regardless of Q-values.
        :return: The selected action based on an epsilon-greedy policy or the best Q-value if not acting randomly.
        """
        valid_actions = env.get_possible_moves()

        if random.random() < self.epsilon or override_random:
            # Choose a random valid action
            action = random.choice(valid_actions)
        else:
            # Get Q-values for all possible actions
            q_values = self.network(torch.tensor(state, dtype=torch.float32))

            # Convert valid actions to indices
            action_indices = [self.coords_to_index(*action) for action in valid_actions]

            # Retrieve Q-values for the valid actions only
            valid_q_values = q_values[action_indices]

            # Apply softmax with temperature
            softmax_probs = torch.softmax(valid_q_values / self.temperature, dim=-1)

            # Sample an action based on the probability distribution
            action_index = torch.multinomial(softmax_probs, 1).item()
            action = valid_actions[action_index]

        return action

    def coords_to_index(self, old_x, old_y, new_x, new_y):
        """
        :param old_x: The x-coordinate of the initial position
        :param old_y: The y-coordinate of the initial position
        :param new_x: The x-coordinate of the target position
        :param new_y: The y-coordinate of the target position
        :return: A unique index representing the move from (old_x, old_y) to (new_x, new_y), based on the board size
        """
        board_size = 11  # Adjust this to your board size if needed
        return old_x * board_size ** 3 + old_y * board_size ** 2 + new_x * board_size + new_y

    def index_to_coords(self, index):
        """
        :param index: An integer representing a flattened index in a 4D (2x2x2) board of size 11x11
        :return: A tuple containing four integers (old_x, old_y, new_x, new_y) representing the coordinates on the board
        """
        board_size = 11
        old_x = index // (board_size ** 3)
        index %= board_size ** 3
        old_y = index // (board_size ** 2)
        index %= board_size ** 2
        new_x = index // board_size
        new_y = index % board_size
        return old_x, old_y, new_x, new_y

    def replay(self):
        """
        Executes a training step by replaying a batch of experiences stored in memory. This involves sampling a batch of
        experiences, computing Q-values for both current and next states, calculating target Q-values, and performing
        backpropagation to update the network weights.

        :return: None
        """
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch of experiences from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to torch tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor([self.coords_to_index(*action) for action in actions], dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)  # 1 for done, 0 for not done

        # Compute Q-values for current states
        q_values = self.network(states)

        # Select Q-values for the taken actions
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute Q-values for next states
        next_q_values = self.network(next_states).max(1)[0]

        # Calculate the target Q-values
        targets = rewards + (1 - dones) * self.gamma * next_q_values

        # Calculate loss using mean squared error
        loss = nn.functional.mse_loss(q_values, targets.detach())

        # Perform backpropagation and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, agent_name, episode, save_dir="saved_models"):
        """
        :param agent_name: The name of the agent whose model is being saved.
        :param episode: The current episode number.
        :param save_dir: The directory where the model will be saved. Defaults to 'saved_models'.
        :return: None
        """
        # Create the directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        filename = f"{save_dir}/{agent_name}.pth"

        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode': episode
        }, filename)

    def load_model(self, filename):
        """
        :param filename: path to the file containing the model checkpoint.
        :return: the episode number from which the model was loaded.
        """
        print("loaded model: " + filename)
        checkpoint = torch.load(filename, weights_only=False)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        return checkpoint['episode']

    def log_progress(self, episode, total_reward):
        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {self.epsilon:.3f}")

STATE_SIZE = 11 * 11
# comes from...
OUTPUT_SIZE = 14641

num_episodes = 10000  # Adjust as needed

COL_NAMES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]


class TrainModel:
    def __init__(self):
        from game import Game
        self.agent1 = DQNAgent(STATE_SIZE, OUTPUT_SIZE)
        self.agent2 = DQNAgent(STATE_SIZE, OUTPUT_SIZE)
        self.game = Game()
        self.game.mode = "training"

        thread = threading.Thread(target=lambda: self.train_model(num_episodes=num_episodes))
        thread.start()

        self.game.play_game()

    def train_model(self, num_episodes=num_episodes):
        """
        :param num_episodes: The number of episodes to train the model for.
        :return: None
        """
        winning_team = [0, 0, 0]
        starting_episode = 1
        if os.path.exists("saved_models/agent1.pth"):
            starting_episode = self.agent1.load_model("saved_models/agent1.pth") + 1
        if os.path.exists("saved_models/agent2.pth"):
            starting_episode = self.agent2.load_model("saved_models/agent2.pth") + 1
        for episode in range(starting_episode, num_episodes):
            state = self.game.reset()
            self.game.episode = episode
            done = False
            total_reward_agent1 = 0
            total_reward_agent2 = 0

            run = 0
            moves = []

            while not done:
                override_random = run <= 5
                # Agent 1's turn
                action1 = self.agent1.act(state, self.game, override_random=override_random)

                grid_x = action1[3]
                grid_y = action1[2]
                row = action1[0]
                col = action1[1]


                turn1 = self.game.turn
                if self.game.board[row][col] == 3:
                    self.game.place_piece(grid_x, grid_y, row, col, 3)
                    moves.append("k" + COL_NAMES[col] + str(row + 1) + "-" + "k" + COL_NAMES[grid_x] + str(grid_y + 1))
                else:
                    turn_letter = "D" if self.game.turn % 2 else "A"
                    self.game.place_piece(grid_x, grid_y, row, col, self.game.turn)
                    moves.append(turn_letter + COL_NAMES[col] + str(row + 1) + "-" + turn_letter + COL_NAMES[grid_x] + str(
                        grid_y + 1))
                states1 = (state, self.game.get_state_representation())
                done = self.game.is_over()

                if done:
                    #time.sleep(10)
                    winning_team[self.game.winning_team] += 1
                    reward1 = self.game.get_reward(turn1)
                    total_reward_agent1 += reward1
                    total_reward_agent2 -= 1
                    self.agent1.memory.append((states1[0], action1, reward1, states1[1], True))
                    break

                if run > 0:
                    reward2 = self.game.get_reward(turn2)
                    total_reward_agent2 += reward2
                    self.agent2.memory.append((states2[0], action2, reward2, states2[1], False))

                # Agent 2's turn
                action2 = self.agent2.act(states1[1], self.game, override_random=override_random)


                grid_x = action2[3]
                grid_y = action2[2]
                row = action2[0]
                col = action2[1]

                turn2 = self.game.turn
                if self.game.board[row][col] == 3:
                    self.game.place_piece(grid_x, grid_y, row, col, 3)
                    moves.append("k" + COL_NAMES[col] + str(row + 1) + "-" + "k" + COL_NAMES[grid_x] + str(grid_y + 1))
                else:
                    turn_letter = "D" if self.game.turn % 2 else "A"
                    self.game.place_piece(grid_x, grid_y, row, col, self.game.turn)
                    moves.append(turn_letter + COL_NAMES[col] + str(row + 1) + "-" + turn_letter + COL_NAMES[grid_x] + str(grid_y + 1))
                states2 = (states1[1], self.game.get_state_representation())
                done = self.game.is_over()

                if done:
                    # if episode % 100 == 0 and episode > 0:
                    #     time.sleep(2)
                    winning_team[self.game.winning_team] += 1
                    reward2 = self.game.get_reward(turn2)
                    total_reward_agent2 += reward2
                    total_reward_agent1 -= 1
                    self.agent2.memory.append((states2[0], action2, reward2, states2[1], True))
                    break

                state = self.game.get_state_representation()

                reward1 = self.game.get_reward(turn1)
                total_reward_agent1 += reward1
                self.agent1.memory.append((states1[0], action1, reward1, states1[1], False))

                run += 1

            # Experience replay
            self.agent1.replay()
            self.agent2.replay()

            # Update exploration rates
            self.agent1.epsilon = max(self.agent1.epsilon * self.agent1.epsilon_decay, self.agent1.epsilon_min)
            self.agent2.epsilon = max(self.agent2.epsilon * self.agent2.epsilon_decay, self.agent2.epsilon_min)

            self.agent1.temperature = max(self.agent1.temperature * 0.99, 0.1)
            self.agent2.temperature = max(self.agent2.temperature * 0.99, 0.1)

            if episode % 100 == 0 and episode > starting_episode:
                hgpn = HPGN(self.game.winning_team, datetime.now().strftime("%m.%d.%y"), moves)
                hgpn.create_file("saved_games/episode_" + str(episode))
                self.agent1.save_model("agent1", episode)
                self.agent2.save_model("agent2", episode)
                print(winning_team)
                print("Episode " + str(episode))
                print("Reward 1 " + str(total_reward_agent1))
                print("Reward 2 " + str(total_reward_agent2))
                gc.collect()

        print("done training")

if __name__ == "__main__":
    TrainModel()
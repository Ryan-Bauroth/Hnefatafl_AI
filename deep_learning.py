import os
from collections import deque
from datetime import time, datetime

import torch
import torch.nn as nn
import torch.optim as optim

import random
import threading
import time

from game import Game



# Neural network model for the Q-learning agent
class DQNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Q-learning agent class
class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, lr=0.001, batch_size=64, memory_size=10000, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # Discount factor => short term or long term rewards
        self.lr = lr  # Learning rate => small = slower but more stable
        self.batch_size = batch_size # how many runs
        self.memory = deque(maxlen=memory_size) # stores the memory of previous actions
        self.epsilon = epsilon  # Exploration rate => randomness rate
        self.epsilon_decay = epsilon_decay  # Decay rate of epsilon
        self.epsilon_min = epsilon_min

        # Initialize the Q-network
        self.network = DQNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

    def act(self, state, env):
        valid_actions = env.get_possible_moves()

        if random.random() < self.epsilon:
            # Choose a random valid action
            action = random.choice(valid_actions)
        else:
            # Get Q-values for all possible actions
            q_values = self.network(torch.tensor(state, dtype=torch.float32))

            # Convert valid actions to indices
            action_indices = [self.coords_to_index(*action) for action in valid_actions]

            # Retrieve Q-values for the valid actions only
            valid_q_values = q_values[action_indices]

            # Find the index of the maximum Q-value among valid actions
            max_index = torch.argmax(valid_q_values).item()

            # Retrieve the best action based on this max index
            action = valid_actions[max_index]

        return action

    def coords_to_index(self, old_x, old_y, new_x, new_y):
        board_size = 11  # Adjust this to your board size if needed
        return old_x * board_size ** 3 + old_y * board_size ** 2 + new_x * board_size + new_y

    def index_to_coords(self, index):
        board_size = 11
        old_x = index // (board_size ** 3)
        index %= board_size ** 3
        old_y = index // (board_size ** 2)
        index %= board_size ** 2
        new_x = index // board_size
        new_y = index % board_size
        return old_x, old_y, new_x, new_y

    def replay(self):
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
        # Create the directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        filename = f"{save_dir}/{agent_name}_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}_episodes_{episode}.pth"

        torch.save(self.network.state_dict(), filename)

    def load_model(self, filename):
        self.network.load_state_dict(torch.load(filename))

    def log_progress(self, episode, total_reward):
        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {self.epsilon:.3f}")

STATE_SIZE = 11 * 11
# comes from..
OUTPUT_SIZE = 14641

num_episodes = 1000  # Adjust as needed

agent1 = DQNAgent(STATE_SIZE, OUTPUT_SIZE)
agent2 = DQNAgent(STATE_SIZE, OUTPUT_SIZE)

game = Game()


def train_model(game, agent1, agent2, num_episodes=1000):
    winning_team = [0, 0, 0]
    for episode in range(num_episodes):
        state = game.reset()
        done = False
        total_reward_agent1 = 0
        total_reward_agent2 = 0

        run = 0

        while not done:

            # Agent 1's turn
            action1 = agent1.act(state, game)

            grid_x = action1[3]
            grid_y = action1[2]
            row = action1[0]
            col = action1[1]


            turn1 = game.turn
            if game.board[row][col] == 3:
                game.place_piece(grid_x, grid_y, row, col, 3)
            else:
                game.place_piece(grid_x, grid_y, row, col, game.turn)
            states1 = (state, game.get_state_representation())
            done = game.is_over()

            if done:
                time.sleep(10)
                winning_team[game.winning_team] += 1
                reward1 = game.get_reward(turn1)
                total_reward_agent1 += reward1
                total_reward_agent2 -= 1
                agent1.memory.append((states1[0], action1, reward1, states1[1], True))
                break

            if run > 0:
                reward2 = game.get_reward(turn2)
                total_reward_agent2 += reward2
                agent2.memory.append((states2[0], action2, reward2, states2[1], False))

            # Agent 2's turn
            action2 = agent2.act(states1[1], game)


            grid_x = action2[3]
            grid_y = action2[2]
            row = action2[0]
            col = action2[1]

            turn2 = game.turn
            if game.board[row][col] == 3:
                game.place_piece(grid_x, grid_y, row, col, 3)
            else:
                game.place_piece(grid_x, grid_y, row, col, game.turn)
            states2 = (states1[1], game.get_state_representation())
            reward2 = game.get_reward(turn2)
            done = game.is_over()

            if done:
                if episode % 100 == 0 and episode > 0:
                    time.sleep(2)
                winning_team[game.winning_team] += 1
                reward2 = game.get_reward(turn2)
                total_reward_agent2 += reward2
                total_reward_agent1 -= 1
                agent2.memory.append((states2[0], action2, reward2, states2[1], True))
                break

            state = game.get_state_representation()

            reward1 = game.get_reward(turn1)
            total_reward_agent1 += reward1
            agent1.memory.append((states1[0], action1, reward1, states1[1], False))

            run += 1

        # Experience replay
        agent1.replay()
        agent2.replay()

        # Update exploration rates
        agent1.epsilon = max(agent1.epsilon * agent1.epsilon_decay, agent1.epsilon_min)
        agent2.epsilon = max(agent2.epsilon * agent2.epsilon_decay, agent2.epsilon_min)

        if episode % 500 == 0 and episode > 0:
            agent1.save_model("agent1", episode)
            agent2.save_model("agent2", episode)


        print(
            f"Episode {episode + 1}, Agent 1 Total Reward: {total_reward_agent1}, Agent 2 Total Reward: {total_reward_agent2}")
        print(winning_team)

thread = threading.Thread(target=lambda: train_model(game, agent1, agent2, num_episodes))
thread.start()

game.play_game()
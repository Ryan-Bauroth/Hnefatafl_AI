'''
IGNORE FILE
'''

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from game import *
from datetime import datetime as DateTime

import os  # Add this to handle file checks and paths

class Simulator:
    def __init__(self, game, depth):
        self.depth = depth
        self.turn = game.turn
        self.reward = {root: [] for root in game.get_possible_actions()}
        self._run_sim(game)
    def _run_sim(self, game, depth=0, reward_so_far=0, root=None):
        new_game = Game()
        new_game.board = [row[:] for row in game.board]
        new_game.turn = game.turn
        for possible_move in new_game.get_possible_moves():
            cur_row, cur_col, new_row, new_col = possible_move
            new_game.place_piece(new_col, new_row, cur_row, cur_col, new_game.board[cur_row][cur_col])
            reward_so_far += (30 if game.turn == 1 else 20) * len(new_game.kill_coords) * (1 if self.turn==game.turn else -1)
            if game.is_over():
                reward_so_far += 200 * (1 if new_game.winning_team == self.turn else -1)
                self.reward[root].append(reward_so_far)
            elif depth + 1 == self.depth:
                self.reward[root].append(reward_so_far)
            else:
                self._run_sim(new_game, depth + 1, reward_so_far, possible_move if root is None else root)
    def best_moves(self):
        mins = {key: min(val) for key, val in self.reward.items()}
        max_min = max(mins.values())
        return [key for key, val in mins.items() if val == max_min]

def multi_channel_board_representation(board):
    # Create an empty array for the multi-channel representation
    multi_channel_board = np.zeros((4, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    # Fill in the channels based on the pieces on the board
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            multi_channel_board[board[r][c]-1, r, c] = 1
    return multi_channel_board

'''def get_possible_captures(game):
    return [
        move for move in game.get_possible_moves() if
        game.board[move[0]][move[1]] != 3 and (
            (move[2] - 2 >= 0 and game.board[move[2] - 1][move[3]] == 3-game.turn and (game.board[move[2] - 2][move[3]] == game.turn or (move[2] - 2, move[3]) in CORNERS)) or
            (move[2] + 2 <= BOARD_SIZE - 1 and game.board[move[2] + 1][move[3]] == 3-game.turn and (game.board[move[2] + 2][move[3]] == game.turn or (move[2] + 2, move[3]) in CORNERS)) or
            (move[3] - 2 >= 0 and game.board[move[2]][move[3] - 1] == 3-game.turn and (game.board[move[2]][move[3] - 2] == game.turn or (move[2], move[3] - 2) in CORNERS)) or
            (move[3] + 2 <= BOARD_SIZE - 1 and game.board[move[2]][move[3] + 1] == 3-game.turn and (game.board[move[2]][move[3] + 2] == game.turn or (move[2], move[3] + 2) in CORNERS))
        )
    ]'''

def rotate_and_flatten_board(board, k):
    return [float(v) for v in np.rot90(np.array(board).reshape(BOARD_SIZE, BOARD_SIZE), k).flatten()]

def rotate_board(board, k):
    new_board = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            rotated = rotate_coords(r, c, k)
            new_board[rotated[0]][rotated[1]] = board[r][c]
    return new_board

def rotate_action(action, k):
    """
    Rotates an action based on how the board is rotated.
    :param action: A tuple (cur_row, cur_col, new_row, new_col).
    :param k: Number of 90-degree rotations to apply.
    :return: Rotated action.
    """
    cur_row, cur_col, new_row, new_col = action
    
    # Rotate the coordinates using the same rotation logic
    cur_row_rotated, cur_col_rotated = rotate_coords(cur_row, cur_col, k)
    new_row_rotated, new_col_rotated = rotate_coords(new_row, new_col, k)
    
    return cur_row_rotated, cur_col_rotated, new_row_rotated, new_col_rotated


def rotate_coords(row, col, k):
    """
    Rotates a coordinate pair (row, col) by 90 degrees counter-clockwise 'k' times.
    :param row: Row index.
    :param col: Column index.
    :param k: Number of 90-degree rotations to apply.
    :return: Rotated coordinates (row, col).
    """
    for _ in range(k):
        row, col = col, BOARD_SIZE - 1 - row
    return row, col

def flatten_board(board):
    new = []
    for row in board:
        new.extend(row)
    return new

def flatten_action(action):
    return action[0] * BOARD_SIZE**3 + action[1] * BOARD_SIZE**2 + action[2] * BOARD_SIZE + action[3]

def unflatten_action(action):
    cur_row = action // (BOARD_SIZE**3)
    action %= BOARD_SIZE**3
    cur_col = action // (BOARD_SIZE**2)
    action %= BOARD_SIZE**2
    new_row = action // BOARD_SIZE
    new_col = action % BOARD_SIZE
    return (
        cur_row, cur_col, new_row, new_col
    )

def select_action(game, policy_net, epsilon):
    """Selects an action using epsilon-greedy policy."""
    possible_moves = game.get_possible_moves()  # Get valid moves from the game
    if game.turn == 1:
        for choice in possible_moves:
            cur_row, cur_col, new_row, new_col = choice
            # north king capture
            if new_col - 1 > 0 and new_col + 1 < BOARD_SIZE - 1 and new_row != BOARD_SIZE - 1 and new_row != 0:
                if game.board[new_row - 1][new_col] == 3 and (
                  new_row - 2 < 0 or game.board[new_row - 2][new_col] == 1) and \
                  game.board[new_row - 1][new_col - 1] == 1 and game.board[new_row - 1][new_col + 1] == 1:
                    return flatten_action(choice)
            # south king capture
            if new_col - 1 > 0 and new_col + 1 < BOARD_SIZE - 1 and new_row != BOARD_SIZE - 1 and new_row != 0:
                if game.board[new_row + 1][new_col] == 3 and (
                  new_row + 2 > BOARD_SIZE - 1 or game.board[new_row + 2][new_col] == 1) and \
                  game.board[new_row + 1][new_col - 1] == 1 and game.board[new_row + 1][new_col + 1] == 1:
                    return flatten_action(choice)
            # west king check
            if new_row - 1 > 0 and new_row + 1 < BOARD_SIZE - 1 and new_col != BOARD_SIZE - 1 and new_col != 0:
                if game.board[new_row][new_col - 1] == 3 and (
                  new_col - 2 < 0 or game.board[new_row][new_col - 2] == 1) and \
                  game.board[new_row - 1][new_col - 1] == 1 and game.board[new_row + 1][new_col - 1] == 1:
                    return flatten_action(choice)
            # east king check
            if new_row - 1 > 0 and new_row + 1 < BOARD_SIZE - 1 and new_col != BOARD_SIZE - 1 and new_col != 0:
                if game.board[new_row][new_col + 1] == 3 and (
                  new_col + 2 > BOARD_SIZE - 1 or game.board[new_row][new_col + 2] == 1) and \
                  game.board[new_row - 1][new_col + 1] == 1 and game.board[new_row + 1][new_col + 1] == 1:
                    return flatten_action(choice)
    else:
        for choice in possible_moves:
            cur_row, cur_col, new_row, new_col = choice
            if game.board[cur_row][cur_col] == 3:
                if (new_row, new_col) in CORNERS:
                    return flatten_action(choice)
    possible_captures = get_possible_captures(game)
    if random.random() >= epsilon:
        with torch.no_grad():
            # Select action with highest Q-value
            k = random.randrange(4)
            q_values = policy_net[game.turn](torch.tensor([multi_channel_board_representation(rotate_board(game.board, k))], dtype=torch.float32))
            mask = torch.zeros(BOARD_SIZE ** 4)  # Initialize the mask for all possible moves
            for move in possible_moves:
                move_index = flatten_action(rotate_action(move, k))
                mask[move_index] = 1  # Set valid moves to 1 in the mask
            masked_q_values = q_values * mask + (1 - mask) * -float('inf')
            return flatten_action(rotate_action(unflatten_action(masked_q_values.max(1)[1].item()), 4-k))  # Return action with max Q-value
    else:
        if possible_captures and random.random() < 0.9:
            return flatten_action(random.choice(possible_captures))
        else:
            return flatten_action(random.choice(possible_moves))

def ori_model():
    trainer = Trainer()
    trainer.load_checkpoint()
    return lambda game: unflatten_action(select_action(game, trainer.policy_net, 0.01))

class DQN_CNN(nn.Module):
    def __init__(self, input_channels):
        super(DQN_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)  # First convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)              # Second convolutional layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)             # Third convolutional layer
        self.fc1 = nn.Linear(128 * BOARD_SIZE**2, 512)              # Fully connected layer
        self.fc2 = nn.Linear(512, BOARD_SIZE**4)                                # Output layer

    def forward(self, x):
        #print(x)
        #batch_size = x.size(0)  # Batch size
        #x = x.view(batch_size, 1, BOARD_SIZE, BOARD_SIZE)
        x = torch.relu(self.conv1(x))      # First conv layer with ReLU
        x = torch.relu(self.conv2(x))      # Second conv layer with ReLU
        x = torch.relu(self.conv3(x))      # Third conv layer with ReLU
        x = x.view(x.size(0), -1)          # Flatten the output from conv layers
        x = torch.relu(self.fc1(x))        # Fully connected layer with ReLU
        return self.fc2(x)                 # Output layer (no activation, since it's Q-values)

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Saves a transition."""
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Randomly samples a batch from memory."""
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)

# Training loop
class Trainer:
    def __init__(self):
        # Hyperparameters
        self.batch_size = 32
        self.gamma = 0.99  # Discount factor
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.999
        self.target_update_freq = 10  # How often to update target network
        self.memory_capacity = 10000
        self.learning_rate = 1e-3
        self.num_episodes = 100000  # Number of episodes to train
        
        # Initialize the DQN and the target network
        input_channels = 4  # For example, single channel board
        self.policy_net = {key: DQN_CNN(input_channels) for key in (1, 2)}
        self.target_net = {key: DQN_CNN(input_channels) for key in (1, 2)}
        for key in (1, 2):
            self.target_net[key].load_state_dict(self.policy_net[key].state_dict())  # Copy parameters from policy to target
            self.target_net[key].eval()  # Set the target network to evaluation mode (no backprop)
        
        self.optimizer = {key: optim.Adam(self.policy_net[key].parameters(), lr=self.learning_rate) for key in (1, 2)}
        self.memory = {key: ReplayMemory(self.memory_capacity) for key in (1, 2)}
        
        # Loss function (Mean Squared Error)
        self.loss_fn = nn.MSELoss()
        self.epsilon = self.epsilon_start
    
    def save_checkpoint(self, episode):
        """Saves the model, optimizer, memory, and episode number."""
        checkpoint = {
            'episode': episode,
            'policy_net_state_dict': {key: net.state_dict() for key, net in self.policy_net.items()},
            'target_net_state_dict': {key: net.state_dict() for key, net in self.target_net.items()},
            'optimizer_state_dict': {key: optim.state_dict() for key, optim in self.optimizer.items()},
            'epsilon': self.epsilon,
            'memory': {key: self.memory[key].memory for key in (1, 2)},  # Save the replay memory
        }
        torch.save(checkpoint, 'checkpoint.pth')
    def load_checkpoint(self):
        """Loads the model, optimizer, memory, and episode number from a checkpoint."""
        if os.path.exists('checkpoint.pth'):
            checkpoint = torch.load('checkpoint.pth')
            self.epsilon = checkpoint['epsilon']  # Restore epsilon
            for key in (1, 2):
                self.policy_net[key].load_state_dict(checkpoint['policy_net_state_dict'][key])
                self.target_net[key].load_state_dict(checkpoint['target_net_state_dict'][key])
                self.optimizer[key].load_state_dict(checkpoint['optimizer_state_dict'][key])
                # Restore the replay memory
                self.memory[key].memory = checkpoint['memory'][key]
            print(f"Checkpoint loaded: Resuming from episode {checkpoint['episode']}")
            return checkpoint['episode']  # Return the last episode number
        return -1
    def train(self):
        episode = self.load_checkpoint()
        while episode < self.num_episodes:
            episode += 1
            game = Game()
            game.setup_board()
            board = [row[:] for row in game.board]
            done = False
            previous_board = None
            previous_action = None
            previous_reward = None
            test_data = []
            move_i = -1
            while not done:
                move_i += 1
                turn = game.turn
                
                # Select an action using epsilon-greedy policy
                action = select_action(game, self.policy_net, self.epsilon)
        
                # Take the action in the environment
                cur_row, cur_col, new_row, new_col = unflatten_action(action)
                game.place_piece(new_col, new_row, cur_row, cur_col, game.board[cur_row][cur_col])
                
                next_board = [row[:] for row in game.board]
                done = game.is_over()
                could_capture = len(get_possible_captures(game)) > 0
                reward = 0
                if done:
                    if game.winning_team==turn:
                        reward += 250
                        if previous_reward is not None:
                            previous_reward -= 250
                    else:
                        reward -= 250
                        if previous_reward is not None:
                            previous_reward += 250
                elif turn == 1:
                    reward += (30 * len(game.kill_coords)) if game.kill_coords else (-15 if could_capture else -6)
                    if previous_reward is not None:
                        previous_reward -= 30 * could_capture
                else:
                    reward += (20 * len(game.kill_coords)) if game.kill_coords else (-10 if could_capture else -4)
                    if previous_reward is not None:
                        previous_reward -= 20 * could_capture
                test_data.append(f"{''.join(str(v) for v in flatten_board(board))}{previous_reward}")
                #print(test_data[-1])
        
                # Store the transition in replay memory
                if previous_reward is not None:
                    for k in range(4):
                        self.memory[3-turn].push(multi_channel_board_representation(rotate_board(previous_board, k)), flatten_action(rotate_action(unflatten_action(previous_action), k)), previous_reward, multi_channel_board_representation(rotate_board(next_board, k)), done)
                if done:
                    for k in range(4):
                        self.memory[turn].push(multi_channel_board_representation(rotate_board(board, k)), flatten_action(rotate_action(unflatten_action(action), k)), reward, multi_channel_board_representation(rotate_board(next_board, k)), done)
        
                # Move to the next state
                previous_board = board
                board = next_board
                previous_action = action
                previous_reward = reward
                
            print(f'{game.winning_team} wins episode {episode} in {move_i//2+1} moves')
            
            # Perform optimization on the policy network
            self.optimize_model()
            
            #print('Optimized')
            
            # Decrease epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon_decay * self.epsilon)
        
            # Update the target network every few episodes
            if episode % self.target_update_freq == self.target_update_freq-1:
                for key in (1, 2):
                    self.target_net[key].load_state_dict(self.policy_net[key].state_dict())
            if episode % 100 == 99:
                open('test_game.txt', 'w').write('\n'.join(test_data))
                self.save_checkpoint(episode)
        
    # Function to optimize the model
    def optimize_model(self):
        for key in (1, 2):
            if len(self.memory[key]) < self.batch_size:
                return
        
            # Sample a batch from memory
            states, actions, rewards, next_states, dones = self.memory[key].sample(self.batch_size)
        
            # Convert to tensors
            states = torch.tensor(np.array(states), dtype=torch.float32)
            actions = torch.tensor(actions)
            rewards = torch.tensor(rewards)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
            dones = torch.tensor(dones)
        
            # Compute Q-values for current states
            state_action_values = self.policy_net[key](states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
            # Compute expected Q-values for next states
            next_state_values = self.target_net[key](next_states).max(1)[0]
            next_state_values = next_state_values * (~dones)
        
            expected_state_action_values = rewards + (self.gamma * next_state_values)
        
            # Compute loss
            loss = self.loss_fn(state_action_values, expected_state_action_values)
        
            # Backpropagation
            self.optimizer[key].zero_grad()
            loss.backward()
            self.optimizer[key].step()
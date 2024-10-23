'''
primary file for model training
'''

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from game_ori import *
#from datetime import datetime as DateTime

import os  # Add this to handle file checks and paths

class Simulator:
    '''
    simulates tree to pick out ideal moves in the short term, hopefully identifying future benefits
    '''
    def __init__(self, game, depth):
        '''
        Simulator initializer
        :param game: game object at current state
        :param depth: depth of simulation
        '''
        self.depth = depth
        self.turn = game.turn
        self.reward = {root: [] for root in game.get_possible_moves()}
        self._run_sim(game)
    def _run_sim(self, game, depth=0, reward_so_far=0, root=None):
        '''
        recursively simulates future moves at given depth
        :param game: game object at state to pick move
        :param depth: current depth of simulation
        :param reward_so_far: reward at current depth
        :param root: initial move of branch
        '''
        depth += 1
        if game.turn == 1:
            king_r, king_c = game.king_loc
            surrounded_before = sum((
                king_r == 0 or game.board[king_r-1][king_c] > 0,
                king_r == BOARD_SIZE - 1 or game.board[king_r+1][king_c] > 0,
                king_c == 0 or game.board[king_r][king_c-1] > 0,
                king_c == BOARD_SIZE - 1 or game.board[king_r][king_c+1] > 0
            ))
        keys_by_piece = {}
        for key in game.get_possible_moves():
            piece = key[0], key[1]
            if piece not in keys_by_piece: keys_by_piece[piece] = []
            keys_by_piece[piece].append(key)
        rewards = {}
        new_games = {}
        for keys in keys_by_piece.values():
            for key in keys:
                rewards[key] = reward_so_far
                new_games[key] = Game()
                new_games[key].board = [row[:] for row in game.board]
                new_games[key].turn = game.turn
                new_games[key].king_loc = game.king_loc
                new_game = new_games[key]
                cur_row, cur_col, new_row, new_col = key
                new_game.place_piece(new_col, new_row, cur_row, cur_col, new_game.board[cur_row][cur_col])
                if new_game.is_over():
                    rewards[key] += 250 * (1 if new_game.winning_team == self.turn else -1) * (self.depth - depth + 1)
                else:
                    rewards[key] += (30 if game.turn == 1 else 20) * len(new_game.kill_coords) * (1 if self.turn==game.turn else -1)
                    if game.turn == 1:
                        surrounded_after = sum((
                            king_r == 0 or new_game.board[king_r-1][king_c] > 0,
                            king_r == BOARD_SIZE - 1 or new_game.board[king_r+1][king_c] > 0,
                            king_c == 0 or new_game.board[king_r][king_c-1] > 0,
                            king_c == BOARD_SIZE - 1 or new_game.board[king_r][king_c+1] > 0
                        ))
                        if surrounded_after == 4:
                            rewards[key] += 250 * (1 if self.turn==game.turn else -1)
                        else:
                            rewards[key] += 5 * (surrounded_after - surrounded_before) * (1 if self.turn==game.turn else -1)
                    if game.board[cur_row][cur_col] == 3:
                        empty_file_to_corner = (
                          (new_row == 0 and (all(new_game.board[0][c]==0 for c in range(1, new_col)) or all(new_game.board[0][c]==0 for c in range(new_col+1, BOARD_SIZE-1)))) or
                          (new_row == BOARD_SIZE - 1 and (all(new_game.board[BOARD_SIZE - 1][c]==0 for c in range(1, new_col)) or all(new_game.board[BOARD_SIZE - 1][c]==0 for c in range(new_col+1, BOARD_SIZE-1)))) or
                          (new_col == 0 and (all(new_game.board[r][0]==0 for r in range(1, new_row)) or all(new_game.board[r][0]==0 for r in range(new_row+1, BOARD_SIZE-1)))) or
                          (new_col == BOARD_SIZE - 1 and (all(new_game.board[r][BOARD_SIZE - 1]==0 for r in range(1, new_row)) or all(new_game.board[r][BOARD_SIZE - 1]==0 for r in range(new_row+1, BOARD_SIZE-1))))
                        )
                        if empty_file_to_corner:
                            rewards[key] += 50 * (1 if self.turn==game.turn else -1) * (1 if game.turn==2 else -1)
                            if new_row == 1 or new_row == BOARD_SIZE - 2 or new_col == 1 or new_col == BOARD_SIZE - 2:
                                rewards[key] += 200 * (1 if self.turn==game.turn else -1) * (1 if game.turn==2 else -1)
        all_keys = list(rewards)
        moves = []
        if depth == 1:
            moves.extend(all_keys)
            '''for piece, keys in keys_by_piece.items():
                random.shuffle(keys)
                moves.append(max(keys, key=lambda x: rewards[x] * (1 if self.turn == game.turn else -1)))'''
        else:
            random.shuffle(all_keys)
            all_keys.sort(key=lambda x: rewards[x], reverse=self.turn==game.turn)
            moves.append(all_keys[0])
            moves.append(all_keys[-1])
        for key in moves:
            new_game = new_games[key]
            if new_game.is_over():
                self.reward[key if root is None else root].append(rewards[key])
            elif depth == self.depth:
                self.reward[key if root is None else root].append(rewards[key])
            else:
                self._run_sim(new_game, depth, rewards[key], key if root is None else root)
    def best_moves(self):
        '''
        :return: best moves after running the simulation
        '''
        #print(self.reward)
        mins = {key: min(val) for key, val in self.reward.items() if val is not None}
        best_min = max(mins.values())
        maxes = {key: max(self.reward[key]) for key, val in mins.items() if val == best_min}
        best_max = max(maxes.values())
        return [key for key, val in maxes.items() if val == best_max]

def multi_channel_board_representation(board):
    '''
    creates board representation with separate channels for each piece type
    method created with help of ChatGPT
    :param board: current board
    '''
    # Create an empty array for the multi-channel representation
    multi_channel_board = np.zeros((4, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    # Fill in the channels based on the pieces on the board
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            multi_channel_board[board[r][c]-1, r, c] = 1
    return multi_channel_board

def rotate_board(board, k):
    '''
    rotates board by k rotations
    :param board: current board
    :param k: number of rotations
    :return: rotated board
    '''
    new_board = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            rotated = rotate_coords(r, c, k)
            new_board[rotated[0]][rotated[1]] = board[r][c]
    return new_board

def rotate_action(action, k):
    """
    rotates action by k rotations
    method created with help of ChatGPT
    :param action: tuple action
    :param k: number of rotations
    :return: rotated action
    """
    cur_row, cur_col, new_row, new_col = action
    
    # Rotate the coordinates using the same rotation logic
    cur_row_rotated, cur_col_rotated = rotate_coords(cur_row, cur_col, k)
    new_row_rotated, new_col_rotated = rotate_coords(new_row, new_col, k)
    
    return cur_row_rotated, cur_col_rotated, new_row_rotated, new_col_rotated


def rotate_coords(row, col, k):
    """
    Rotates a coordinate pair (row, col) by 90 degrees counter-clockwise 'k' times.
    method created with help of ChatGPT
    :param row: Row index.
    :param col: Column index.
    :param k: Number of 90-degree rotations to apply.
    :return: Rotated coordinates (row, col).
    """
    for _ in range(k):
        row, col = col, BOARD_SIZE - 1 - row
    return row, col

def flatten_board(board):
    '''
    flattens board
    :param board: current board
    :return: flattened board
    '''
    new = []
    for row in board:
        new.extend(row)
    return new

def flatten_action(action):
    '''
    flattens action
    :param action: tuple action
    :return: flattened action
    '''
    return action[0] * BOARD_SIZE**3 + action[1] * BOARD_SIZE**2 + action[2] * BOARD_SIZE + action[3]

def unflatten_action(action):
    '''
    unflattens action
    :param action: flattened action
    :return: unflattened action
    '''
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
    """
    selects action with combination of reinforcement learning and Simulator with epsilon policy
    method created with help of ChatGPT
    :param game: current game
    :param policy_net: policy network
    :param epsilon: epsilon
    """
    best_moves = Simulator(game, 1 if random.random() < epsilon else 2).best_moves()
    if random.random() >= epsilon:
        with torch.no_grad():
            # Select action with highest Q-value
            k = random.randrange(4)
            q_values = policy_net[game.turn](torch.tensor([multi_channel_board_representation(rotate_board(game.board, k))], dtype=torch.float32))
            mask = torch.zeros(BOARD_SIZE ** 4)  # Initialize the mask for all possible moves
            for move in best_moves:
                move_index = flatten_action(rotate_action(move, k))
                mask[move_index] = 1  # Set valid moves to 1 in the mask
            masked_q_values = q_values * mask + (1 - mask) * -float('inf')
            return flatten_action(rotate_action(unflatten_action(masked_q_values.max(1)[1].item()), 4-k))  # Return action with max Q-value
    else:
        return flatten_action(random.choice(best_moves))

def ori_model():
    """
    my model
    :return: function of current game with best move
    """
    trainer = Trainer()
    trainer.load_checkpoint()
    return lambda game: unflatten_action(select_action(game, trainer.policy_net, 0))

class DQN_CNN(nn.Module):
    '''
    reinforcement Deep Q model with convolutional layers
    class created with help of ChatGPT
    '''
    def __init__(self, input_channels):
        super(DQN_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)  # First convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)              # Second convolutional layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)             # Third convolutional layer
        self.fc1 = nn.Linear(128 * BOARD_SIZE**2, 512)              # Fully connected layer
        self.fc2 = nn.Linear(512, BOARD_SIZE**4)                                # Output layer

    def forward(self, x):
        '''
        flattens convolutional network and connects layers with ReLU
        :param x: input tensor
        '''
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
    '''
    buffer deque for replay memory
    class created with help of ChatGPT
    '''
    def __init__(self, capacity):
        '''
        initializes memory
        :param capacity: capacity of memory
        '''
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        adds action to replay memory
        :param state: current state
        :param action: action taken
        :param reward: reward received
        :param next_state: next state
        :param done: game done
        """
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        randomly samples a batch from memory
        :param batch_size: batch size
        :return: sampled batch
        """
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        '''
        length of memory
        :return: length of memory
        '''
        return len(self.memory)

class Trainer:
    '''
    trains model
    class created with help of ChatGPT
    '''
    def __init__(self):
        '''
        Trainer initializer
        '''
        # Hyperparameters
        self.batch_size = 32
        self.gamma = 0.99  # Discount factor
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995
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
        """
        saves model, optimizer, memory, episode number
        """
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
        """
        Loads the model, optimizer, memory, episode number
        """
        if os.path.exists('checkpoint.pth'):
            checkpoint = torch.load('checkpoint.pth')
            self.epsilon = checkpoint['epsilon']  # Restore epsilon
            for key in (1, 2):
                self.policy_net[key].load_state_dict(checkpoint['policy_net_state_dict'][key])
                self.target_net[key].load_state_dict(checkpoint['target_net_state_dict'][key])
                self.optimizer[key].load_state_dict(checkpoint['optimizer_state_dict'][key])
                # Restore the replay memory
                self.memory[key].memory = checkpoint['memory'][key]
            print(f"Checkpoint loaded: Resuming from episode {checkpoint['episode']+1}")
            return checkpoint['episode']  # Return the last episode number
        return -1
    def train(self):
        '''
        training loop
        '''
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
                #print('here')
                move_i += 1
                turn = game.turn
                
                # Select an action using epsilon-greedy policy
                action = select_action(game, self.policy_net, self.epsilon)
        
                # Take the action in the environment
                cur_row, cur_col, new_row, new_col = unflatten_action(action)
                game.place_piece(new_col, new_row, cur_row, cur_col, game.board[cur_row][cur_col])
                
                next_board = [row[:] for row in game.board]
                done = game.is_over()
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
                    reward += (30 * len(game.kill_coords)) if game.kill_coords else -6
                else:
                    reward += (20 * len(game.kill_coords)) if game.kill_coords else -4
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
                open('test_game.txt', 'w').write('\n'.join(test_data))
                self.save_checkpoint(episode)
        
    def optimize_model(self):
        '''
        optimizes double DQN (policy network and target network)
        '''
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
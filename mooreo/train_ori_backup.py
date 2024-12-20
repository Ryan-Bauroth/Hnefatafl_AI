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

def rotate_and_flatten_board(board, k):
    return [float(v) for v in np.rot90(np.array(board).reshape(BOARD_SIZE, BOARD_SIZE), k).flatten()]

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


def save_checkpoint(dqn, optimizer, episode, epsilon, replay_buffer, filepath):
    checkpoint = {
        'dqn_1_state_dict': dqn[1].state_dict(),
        'dqn_2_state_dict': dqn[2].state_dict(),
        'optimizer_1_state_dict': optimizer[1].state_dict(),
        'optimizer_2_state_dict': optimizer[2].state_dict(),
        'episode': episode,
        'epsilon': epsilon,
        'replay_buffer_1': replay_buffer[1],
        'replay_buffer_2': replay_buffer[2],
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(dqn, optimizer, filepath):
    try:
        checkpoint = torch.load(filepath)
        dqn[1].load_state_dict(checkpoint['dqn_1_state_dict'])
        dqn[2].load_state_dict(checkpoint['dqn_2_state_dict'])
        optimizer[1].load_state_dict(checkpoint['optimizer_1_state_dict'])
        optimizer[2].load_state_dict(checkpoint['optimizer_2_state_dict'])
        episode = checkpoint['episode']
        epsilon = checkpoint['epsilon']
        replay_buffer = {1: checkpoint['replay_buffer_1'], 2: checkpoint['replay_buffer_2']}
        print(f'Loaded checkpoint from episode {episode}')
        return episode, epsilon, replay_buffer
    except FileNotFoundError:
        print('No checkpoint found, starting from scratch.')
        return 0, 1.0, {i: ReplayBuffer(max_size=10000) for i in (1, 2)}  # Initialize new replay buffers if none exist

class DQN_CNN(nn.Module):
    def __init__(self, input_channels, board_size, output_size):
        super(DQN_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)  # First convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)              # Second convolutional layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)             # Third convolutional layer
        self.fc1 = nn.Linear(128 * board_size * board_size, 512)              # Fully connected layer
        self.fc2 = nn.Linear(512, output_size)                                # Output layer

    def forward(self, x):
        x = torch.relu(self.conv1(x))      # First conv layer with ReLU
        x = torch.relu(self.conv2(x))      # Second conv layer with ReLU
        x = torch.relu(self.conv3(x))      # Third conv layer with ReLU
        x = x.view(x.size(0), -1)          # Flatten the output from conv layers
        x = torch.relu(self.fc1(x))        # Fully connected layer with ReLU
        return self.fc2(x)                 # Output layer (no activation, since it's Q-values)

    
class ReplayBuffer:
    def __init__(self, max_size):
        '''
        max_size: This parameter limits how many experiences can be stored in the buffer. Once it reaches this size, the oldest experiences are discarded (FIFO - First In, First Out).
        self.buffer = deque(maxlen=max_size): Initializes a deque (double-ended queue) to store the experiences
        :param max_size:
        '''
        self.buffer = deque(maxlen=max_size)
    def add(self, experience):
        '''
        experience: A tuple that contains (state, action, reward, next state, done).
        self.buffer.append(experience): Adds the new experience to the end of the buffer.
        :param experience:
        :return:
        '''
        self.buffer.append(experience)
    def sample(self, batch_size):
        '''
        batch_size: The number of experiences to randomly sample from the buffer.
        random.sample(self.buffer, batch_size): Returns a list of randomly selected experiences from the buffer. This randomness helps the model generalize better during training.
        :param batch_size:
        :return:
        '''
        return random.sample(self.buffer, batch_size)
    def size(self):
        return len(self.buffer)

def train_on_batch(batch, dqn, optimizer, gamma):
    states, actions, rewards, next_states, dones = zip(*batch)

    '''
    torch.FloatTensor(...): Converts the lists into PyTorch tensors, which are the main data structures for training in PyTorch.
    actions = torch.LongTensor(actions): Converts actions to LongTensor because they are indices (whole numbers).
    '''
    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)

    '''
    q_values = dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1):
    dqn(states): Passes the batch of states through the DQN to get the Q-values for all actions.
    gather(1, actions.unsqueeze(1)): Selects the Q-values corresponding to the actions taken using their indices.
    squeeze(1): Removes unnecessary dimensions from the tensor for easier manipulation.
    '''
    # Get current Q-values for the actions taken
    q_values = dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    '''
    next_q_values = dqn(next_states).max(1)[0]:
    Passes the next states through the DQN and retrieves the maximum Q-value for each next state. This represents the best expected future reward from the next state.
    '''
    # Get max Q-values for the next states
    next_q_values = dqn(next_states).max(1)[0]
    '''
    targets = rewards + gamma * next_q_values * (1 - dones):
    Calculates the target Q-values. The target is composed of the immediate reward plus the discounted maximum future reward:
    rewards: Immediate reward from taking the action.
    gamma * next_q_values: Discounted future reward.
    (1 - dones): If the episode has ended (done is True), the future reward contribution should be zero. This is why we multiply by (1 - dones).
    '''
    # Compute the target Q-values
    targets = rewards + gamma * next_q_values * (1 - dones)

    '''
    loss = nn.MSELoss()(q_values, targets.detach()):
    Calculates the mean squared error (MSE) loss between the predicted Q-values and the target Q-values. This measures how far off our predictions are from the true expected values.
    '''
    # Compute the loss and optimize the DQN
    loss = nn.MSELoss()(q_values, targets.detach())
    '''
    optimizer.zero_grad(): Clears the previous gradients before backpropagation.
    loss.backward(): Computes the gradients of the loss with respect to the model parameters.
    optimizer.step(): Updates the model parameters based on the gradients computed.
    '''
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

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

def train_dqn():
    dqn = {i: DQN(BOARD_SIZE ** 4) for i in (1, 2)}
    optimizer = {i: optim.Adam(dqn[i].parameters(), lr=0.001) for i in (1, 2)}
    batch_size = 64
    gamma = 0.99  # Discount factor
    epsilon_decay = 0.995  # Decay rate for epsilon
    min_epsilon = 0.01  # Minimum epsilon
    start_episode, epsilon, replay_buffer = load_checkpoint(dqn, optimizer, 'model_ori.pth')
    num_episodes = 1000000  # Number of training episodes
    for episode in range(start_episode, start_episode + num_episodes):
        game = Game()
        game.setup_board()
        episode_reward = {i: 0 for i in (1, 2)}
        move_i = -1
        previous_board = None
        previous_action = None
        previous_reward = None
        test_data = []
        while not game.is_over():
            move_i += 1
            turn = game.turn
            state = game.get_state_representation()
            board = [row[:] for row in game.board]
            '''print('—'*10)
            print(' \t' + ' '.join(str(v) for v in range(BOARD_SIZE)))
            for board_i in range(BOARD_SIZE):
                print(f'{board_i}\t' + ' '.join(str(v) if v else ' ' for v in state[BOARD_SIZE*board_i:BOARD_SIZE*(board_i+1)]))
            input()'''
            
            # Create a mask to filter out invalid moves
            possible_moves = game.get_possible_moves()  # Get valid moves from the game
            next_to_opp = [
                move for move in possible_moves if
                (move[2] > 0 and game.board[move[2] - 1][move[3]] not in (0, turn)) or
                (move[2] < BOARD_SIZE - 1 and game.board[move[2] + 1][move[3]] not in (0, turn)) or
                (move[3] > 0 and game.board[move[2]][move[3] - 1] not in (0, turn)) or
                (move[3] < BOARD_SIZE - 1 and game.board[move[2]][move[3] + 1] not in (0, turn))
            ]
            could_capture = [
                move for move in next_to_opp if
                (move[2] - 2 >= 0 and game.board[move[2] - 1][move[3]] == 3-turn and (game.board[move[2] - 2][move[3]] == turn or (move[2] - 2, move[3]) in CORNERS)) or
                (move[2] + 2 <= BOARD_SIZE - 1 and game.board[move[2] + 1][move[3]] == 3-turn and (game.board[move[2] + 2][move[3]] == turn or (move[2] + 2, move[3]) in CORNERS)) or
                (move[3] - 2 >= 0 and game.board[move[2]][move[3] - 1] == 3-turn and (game.board[move[2]][move[3] - 2] == turn or (move[2], move[3] - 2) in CORNERS)) or
                (move[3] + 2 <= BOARD_SIZE - 1 and game.board[move[2]][move[3] + 1] == 3-turn and (game.board[move[2]][move[3] + 2] == turn or (move[2], move[3] + 2) in CORNERS))
            ]
            action_choice = None
            if turn == 1:
                for choice in possible_moves:
                    cur_row, cur_col, new_row, new_col = choice
                    # north king capture
                    if new_col - 1 > 0 and new_col + 1 < BOARD_SIZE - 1 and new_row != BOARD_SIZE - 1 and new_row != 0:
                        if game.board[new_row - 1][new_col] == 3 and (new_row - 2 < 0 or game.board[new_row - 2][new_col] == 1) and \
                                game.board[new_row - 1][new_col - 1] == 1 and game.board[new_row - 1][new_col + 1] == 1:
                            action_choice = flatten_action(choice)
                            break
                    # south king capture
                    if new_col - 1 > 0 and new_col + 1 < BOARD_SIZE - 1 and new_row != BOARD_SIZE - 1 and new_row != 0:
                        if game.board[new_row + 1][new_col] == 3 and (new_row + 2 > BOARD_SIZE - 1 or game.board[new_row + 2][new_col] == 1) and \
                                game.board[new_row + 1][new_col - 1] == 1 and game.board[new_row + 1][new_col + 1] == 1:
                            action_choice = flatten_action(choice)
                            break
                    # west king check
                    if new_row - 1 > 0 and new_row + 1 < BOARD_SIZE - 1 and new_col != BOARD_SIZE - 1 and new_col != 0:
                        if game.board[new_row][new_col - 1] == 3 and (new_col - 2 < 0 or game.board[new_row][new_col - 2] == 1) and \
                                game.board[new_row - 1][new_col - 1] == 1 and game.board[new_row + 1][new_col - 1] == 1:
                            action_choice = flatten_action(choice)
                            break
                    # east king check
                    if new_row - 1 > 0 and new_row + 1 < BOARD_SIZE - 1 and new_col != BOARD_SIZE - 1 and new_col != 0:
                        if game.board[new_row][new_col + 1] == 3 and (new_col + 2 > BOARD_SIZE - 1 or game.board[new_row][new_col + 2] == 1) and \
                                game.board[new_row - 1][new_col + 1] == 1 and game.board[new_row + 1][new_col + 1] == 1:
                            action_choice = flatten_action(choice)
                            break
            else:
                for choice in possible_moves:
                    cur_row, cur_col, new_row, new_col = choice
                    if game.board[cur_row][cur_col] == 3:
                        if (new_row, new_col) in CORNERS:
                            action_choice = flatten_action(choice)
                            break
            if action_choice is None:
                # Choose action based on exploration strategy
                if random.random() < epsilon:  # Exploration
                    rand_val = random.random()
                    if could_capture and rand_val < 0.9:
                        choice = random.choice(could_capture)
                        # print(choice)
                        cur_row, cur_col, new_row, new_col = choice
                        action_choice = flatten_action(choice)
                    elif (next_to_opp and rand_val < 0.8):
                        choice = random.choice(next_to_opp)
                        # print(choice)
                        cur_row, cur_col, new_row, new_col = choice
                        action_choice = flatten_action(choice)
                    else:
                        choice = random.choice(possible_moves)
                        #print(choice)
                        cur_row, cur_col, new_row, new_col = choice
                        action_choice = flatten_action(choice)
                else:  # Exploitation
                    k = random.randrange(4)
                    state_tensor = torch.FloatTensor(rotate_and_flatten_board(game.board, k)).unsqueeze(0)  # Prepare the state for the DQN
                    with torch.no_grad():
                        q_values = dqn[turn](state_tensor)  # Get Q-values from the DQN
                    mask = torch.zeros(BOARD_SIZE ** 4)  # Initialize the mask for all possible moves
                    for move in possible_moves:
                        move_index = flatten_action(rotate_action(move, k))
                        mask[move_index] = 1  # Set valid moves to 1 in the mask
                    # Apply the mask to the Q-values (invalid moves get a Q-value of -inf)
                    masked_q_values = q_values * mask + (1 - mask) * -float('inf')
                    action_choice = flatten_action(rotate_action(unflatten_action(torch.argmax(masked_q_values).item()), 4-k))
                    # Convert the action index back to (current_row, current_col, new_row, new_col)
                    cur_row, cur_col, new_row, new_col = unflatten_action(action_choice)           # Apply the chosen action to the game
            game.place_piece(new_col, new_row, cur_row, cur_col, game.board[cur_row][cur_col])
            #next_state = game.get_state_representation()
            done = game.is_over()
            #print(game.turn)
            reward = 0
            if done:
                if game.winning_team==turn:
                    reward += 500
                    if previous_reward is not None:
                        previous_reward -= 500
                else:
                    reward -= 500
                    if previous_reward is not None:
                        previous_reward += 500
            elif turn == 1:
                reward += (30 * len(game.kill_coords)) if game.kill_coords else (-30 if could_capture else -6)
                if previous_reward is not None:
                    previous_reward -= 30 * len(game.kill_coords)
            else:
                reward += (20 * len(game.kill_coords)) if game.kill_coords else (-20 if could_capture else -5)
                if previous_reward is not None:
                    previous_reward -= 20 * len(game.kill_coords)
            test_data.append(f"{''.join(str(v) for v in state)}{previous_reward}")
            #reward = (150 if game.winning_team==turn else -150) if done else len(game.kill_coords) * 10#(10 if turn == 1 else 5)
            '''if turn == 1:
                if (new_row > 0 and game.board[new_row-1][new_col]==3) or (new_row < BOARD_SIZE-1 and game.board[new_row+1][new_col]==3) or (new_col > 0 and game.board[new_row][new_col-1]==3) or (new_col < BOARD_SIZE-1 and game.board[new_row][new_col+1]==3):
                    reward += 1'''
            #reward += 1 if turn == 1 else 0
            #print(reward)
            # Store the experience in the replay buffer
            if done:
                #next_state = game.get_state_representation()
                #next_board = [row[:] for row in game.board]
                for k in range(4):
                    replay_buffer[turn].add((
                        rotate_and_flatten_board(board, k),
                        flatten_action(rotate_action(unflatten_action(action_choice), k)),
                        reward,
                        rotate_and_flatten_board(game.board, k),
                        done
                    ))
                episode_reward[turn] += reward
            if previous_reward is not None:
                for k in range(4):
                    replay_buffer[3-turn].add((
                        rotate_and_flatten_board(previous_board, k),
                        flatten_action(rotate_action(unflatten_action(previous_action), k)),
                        previous_reward,
                        rotate_and_flatten_board(board, k),
                        done
                    ))
                episode_reward[3-turn] += previous_reward
            # Train the DQN if enough experiences are in the buffer
            if replay_buffer[turn].size() > batch_size:
                batch = replay_buffer[turn].sample(batch_size)
                train_on_batch(batch, dqn[turn], optimizer[turn], gamma)
            
            previous_board = board
            previous_action = action_choice
            previous_reward = reward
            
            #episode_reward[turn] += reward  # Accumulate reward for the current player
        
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        print(f'{game.winning_team} wins episode {episode} in {move_i//2+1} moves with reward {episode_reward[game.winning_team]}')
        if episode % 100 == 99:
            save_checkpoint(dqn, optimizer, episode+1, epsilon, replay_buffer, 'model_ori.pth')
        open('test_game.txt', 'w').write('\n'.join(test_data))


train_dqn()
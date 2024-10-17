import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from game import *

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128) # First hidden layer
        self.fc2 = nn.Linear(128, 64)         # Second hidden layer
        self.fc3 = nn.Linear(64, output_size) # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x)) # Apply ReLU activation
        x = torch.relu(self.fc2(x)) # Apply ReLU activation
        return self.fc3(x)          # Output layer
    
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
    dqn = {i: DQN(input_size=BOARD_SIZE ** 2, output_size=BOARD_SIZE ** 4) for i in (1, 2)}
    optimizer = {i: optim.Adam(dqn[i].parameters(), lr=0.001) for i in (1, 2)}
    replay_buffer = {i: ReplayBuffer(max_size=10000) for i in (1, 2)}
    batch_size = 64
    gamma = 0.99  # Discount factor
    epsilon = 1.0  # Exploration rate
    epsilon_decay = 0.995  # Decay rate for epsilon
    min_epsilon = 0.01  # Minimum epsilon

    num_episodes = 1000  # Number of training episodes
    for episode in range(num_episodes):
        game = Game()
        episode_reward = {i: 0 for i in (1, 2)}

        while not game.is_over():
            turn = game.turn
            state = game.get_state_representation()
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Prepare the state for the DQN
            with torch.no_grad():
                q_values = dqn[turn](state_tensor)  # Get Q-values from the DQN
            
            # Create a mask to filter out invalid moves
            possible_moves = game.get_possible_moves()  # Get valid moves from the game
            print(possible_moves)
            mask = torch.zeros(BOARD_SIZE ** 4)  # Initialize the mask for all possible moves
            for move in possible_moves:
                move_index = flatten_action(move)
                mask[move_index] = 1  # Set valid moves to 1 in the mask
            
            # Apply the mask to the Q-values (invalid moves get a Q-value of -inf)
            masked_q_values = q_values * mask + (1 - mask) * -float('inf')
            # Choose action based on exploration strategy
            if random.random() < epsilon:  # Exploration
                best_action = flatten_action(random.choice(possible_moves))
            else:  # Exploitation
                best_action = torch.argmax(masked_q_values).item()
            
            # Convert the action index back to (current_row, current_col, new_row, new_col)
            cur_row, cur_col, new_row, new_col = unflatten_action(best_action)
            
            # Apply the chosen action to the game
            game.place_piece(cur_row, cur_col, new_row, new_col, turn)
            next_state = game.get_state_representation()
            done = game.is_over()
            reward = ((100 if game.winning_team==turn else -100) if done else 10*len(game.kill_coords)) + turn==1
            
            # Store the experience in the replay buffer
            replay_buffer[turn].add((state, best_action, reward, next_state, done))
            
            # Train the DQN if enough experiences are in the buffer
            if replay_buffer[turn].size() > batch_size:
                batch = replay_buffer[turn].sample(batch_size)
                train_on_batch(batch, dqn[turn], optimizer[turn], gamma)
            
            episode_reward[turn] += reward  # Accumulate reward for the current player
        
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        print(f'{game.winning_team} wins with reward {episode_reward[game.winning_team]} over {episode_reward[3-game.winning_team]}')
    
    # Save the trained models for both players
    torch.save(dqn[1].state_dict(), 'models/trained_dqn_1.pth')
    torch.save(dqn[2].state_dict(), 'models/trained_dqn_2.pth')
train_dqn()
import math
import random
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from kaggle_environments import make

Transitions = namedtuple(
    'Transitions',
    ('state', 'action', 'next_state', 'reward')
)


class ReplayMemory:
    def __init__(self, memory_size):
        self.capacity = memory_size
        self.position = 0
        self.memory = []

    def __len__(self):
        return len(self.memory)

    def push(self, *args):
        if len(self) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transitions(*args)
        self.position += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


class DeepQNetwork(nn.Module):
    def __init__(self):
        super(DeepQNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=(3, 4), padding=1)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=4)
        self.dense1 = nn.Linear(9, 18)
        self.head = nn.Linear(18, 7)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dense1(x.view(x.size(0), -1))
        return self.head(x)


BATCH_SIZE = 4
EPSILON = 0.9
EPSILON_END = 0.05
DECAY = 200
GAMMA = 0.9
policy_net = DeepQNetwork()
target_net = DeepQNetwork()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())

memory = ReplayMemory(3000)
env = make("connectx", debug=True)
env_config = env.configuration
COLS = env_config["columns"]
ROWS = env_config["rows"]


def optimize_network():
    if len(memory) > BATCH_SIZE:
        transition = memory.sample(BATCH_SIZE)

        batch = Transitions(*zip(*transition))

        state = torch.cat(batch.state)
        next_state = torch.cat(batch.next_state)
        reward = torch.tensor(batch.reward)
        action = torch.tensor(batch.action).reshape(1, -1)

        state_actions_values = policy_net(state).gather(1, action).view(-1)
        next_state_values = target_net(next_state).max(1)[0].detach()
        expected_state_action_values = next_state_values * GAMMA + reward

        loss = func.smooth_l1_loss(expected_state_action_values, state_actions_values)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def get_state_and_reward(observation):
    board = torch.tensor(observation["board"]).resize(ROWS, COLS)
    reward = observation["reward"]
    return board, reward


def agent(obs, config):
    epsilon = EPSILON_END + (EPSILON - EPSILON_END) * math.exp(-1 * config.step_no / DECAY)
    if random.random() > epsilon:
        state = torch.tensor(obs["board"], dtype=torch.float).resize(1, 1, ROWS, COLS)
        with torch.no_grad():
            move = policy_net(state)
            return move.argmax().item()
    else:
        return random.choice(range(config["columns"]))


def train_loop():
    env_config.step_no = 0
    for i in range(10000):
        trainer = env.train([None, 'random'])
        observation = trainer.reset()
        while not env.done:
            action = agent(observation, env_config)
            state = torch.tensor(observation["board"], dtype=torch.float).reshape(1, 1, ROWS, COLS)
            observation, reward, _, info = trainer.step(action)
            next_state = torch.tensor(observation["board"], dtype=torch.float).reshape(1, 1, ROWS, COLS)
            if reward is None:
                memory.push(state, action, next_state, -10)

            optimize_network()
            env_config.step_no += 1

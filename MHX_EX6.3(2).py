import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


# 环境定义
class MultiAgentEnv:
    def __init__(self, grid_size, num_hunters):
        self.grid_size = grid_size
        self.num_hunters = num_hunters
        self.hunter_positions = self._init_positions(num_hunters)
        self.prey_position = self._init_positions(1)[0]
        self.steps = []

    def _init_positions(self, num_agents):
        return [tuple(np.random.randint(0, self.grid_size, size=2)) for _ in range(num_agents)]

    def move_prey(self, action):
        # 逃跑者根据动作移动
        move_options = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        move = move_options[action]
        new_position = (self.prey_position[0] + move[0], self.prey_position[1] + move[1])
        if self._is_within_grid(new_position):
            self.prey_position = new_position

    def move_hunters(self):
        # 追捕者策略：接近逃跑者
        move_options = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for i, hunter in enumerate(self.hunter_positions):
            best_move = hunter
            min_distance = np.linalg.norm(np.array(hunter) - np.array(self.prey_position))
            for move in move_options:
                new_position = (hunter[0] + move[0], hunter[1] + move[1])
                new_distance = np.linalg.norm(np.array(new_position) - np.array(self.prey_position))
                if new_distance < min_distance and self._is_within_grid(new_position):
                    best_move = new_position
                    min_distance = new_distance

            self.hunter_positions[i] = best_move

    def _is_within_grid(self, position):
        return 0 <= position[0] < self.grid_size and 0 <= position[1] < self.grid_size

    def check_capture(self):
        return any(np.array_equal(hunter, self.prey_position) for hunter in self.hunter_positions)

    def step(self, action):
        self.move_prey(action)
        self.move_hunters()
        state = self.get_state()
        reward = -1  # 每步奖励为-1
        done = self.check_capture()
        if done:
            reward = -100  # 被捕获惩罚
        self.steps.append((self.hunter_positions.copy(), self.prey_position))
        return state, reward, done

    def reset(self):
        self.hunter_positions = self._init_positions(self.num_hunters)
        self.prey_position = self._init_positions(1)[0]
        self.steps = []
        return self.get_state()

    def get_state(self):
        state = np.zeros((self.grid_size, self.grid_size))
        for hunter in self.hunter_positions:
            state[hunter] = -1  # 追捕者的位置
        state[self.prey_position] = 1  # 逃跑者的位置
        return state.flatten()


# DQN算法实现
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)


def train(env, model, optimizer, replay_buffer, batch_size, gamma):
    if len(replay_buffer) < batch_size:
        return
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = torch.FloatTensor(state)
    next_state = torch.FloatTensor(next_state)
    action = torch.LongTensor(action)
    reward = torch.FloatTensor(reward)
    done = torch.FloatTensor(done)

    q_values = model(state)
    next_q_values = model(next_state)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss = nn.MSELoss()(q_value, expected_q_value)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def main():
    grid_size = 20
    num_hunters = 5
    env = MultiAgentEnv(grid_size, num_hunters)
    input_dim = grid_size * grid_size
    output_dim = 4
    model = DQN(input_dim, output_dim)
    optimizer = optim.Adam(model.parameters())
    replay_buffer = ReplayBuffer(10000)
    batch_size = 64
    gamma = 0.99
    epsilon = 0.1
    num_episodes = 500

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        for t in range(100):
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                with torch.no_grad():
                    q_values = model(torch.FloatTensor(state).unsqueeze(0))
                    action = q_values.max(1)[1].item()
            next_state, reward, done = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            train(env, model, optimizer, replay_buffer, batch_size, gamma)
            if done:
                break
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    # 动画部分
    fig, ax = plt.subplots()
    ax.set_xlim(0, env.grid_size)
    ax.set_ylim(0, env.grid_size)

    hunter_scatter = ax.scatter([], [], c='red', label='Hunter')
    prey_scatter = ax.scatter([], [], c='blue', label='Prey')
    ax.legend()

    def init():
        hunter_scatter.set_offsets(np.empty((0, 2)))
        prey_scatter.set_offsets(np.empty((0, 2)))
        return hunter_scatter, prey_scatter

    def update(frame):
        hunters, prey = env.steps[frame]
        hunter_scatter.set_offsets(hunters)
        prey_scatter.set_offsets([prey])
        return hunter_scatter, prey_scatter

    ani = animation.FuncAnimation(fig, update, frames=len(env.steps), init_func=init, blit=True, repeat=False)
    plt.show()


if __name__ == "__main__":
    main()

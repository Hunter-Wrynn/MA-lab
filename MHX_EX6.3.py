import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import copy

class MultiAgentEnv:
    def __init__(self, grid_size, num_hunters, shelter_positions, distance_threshold):
        self.grid_size = grid_size
        self.num_hunters = num_hunters
        self.hunter_positions = self._init_positions(num_hunters)
        self.prey_position = self._init_positions(1)[0]
        self.shelter_positions = shelter_positions
        self.steps = []
        self.shelter_timer = 0
        self.in_shelter = False
        self.post_shelter_timer = 0
        self.distance_threshold = distance_threshold

    def _init_positions(self, num_agents):
        return [tuple(np.random.randint(0, self.grid_size, size=2)) for _ in range(num_agents)]

    def move_prey(self):
        move_options = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        if self.in_shelter:
            self.shelter_timer += 1
            if self.shelter_timer >= 2:
                self.in_shelter = False
                self.shelter_timer = 0
                self.post_shelter_timer = 3  # 禁止再次进入避难所的时间
                # 随机朝四个方向走一步离开避难所
                while True:
                    move = random.choice(move_options)
                    new_position = (self.prey_position[0] + move[0], self.prey_position[1] + move[1])
                    if self._is_within_grid(new_position):
                        self.prey_position = new_position
                        break
        elif self.post_shelter_timer > 0:
            self.post_shelter_timer -= 1
            # 远离最近的追捕者
            distances_to_hunters = [np.linalg.norm(np.array(self.prey_position) - np.array(hunter)) for hunter in self.hunter_positions]
            nearest_hunter = self.hunter_positions[np.argmin(distances_to_hunters)]
            best_move = self.prey_position
            max_distance = np.linalg.norm(np.array(self.prey_position) - np.array(nearest_hunter))
            for move in move_options:
                new_position = (self.prey_position[0] + move[0], self.prey_position[1] + move[1])
                new_distance = np.linalg.norm(np.array(new_position) - np.array(nearest_hunter))
                if new_distance > max_distance and self._is_within_grid(new_position):
                    best_move = new_position
                    max_distance = new_distance
            self.prey_position = best_move
        else:
            distances_to_hunters = [np.linalg.norm(np.array(self.prey_position) - np.array(hunter)) for hunter in self.hunter_positions]
            nearest_hunter = self.hunter_positions[np.argmin(distances_to_hunters)]

            # 找到最近的避难所
            distances_to_shelters = [np.linalg.norm(np.array(self.prey_position) - np.array(shelter)) for shelter in self.shelter_positions]
            nearest_shelter = self.shelter_positions[np.argmin(distances_to_shelters)]

            if np.linalg.norm(np.array(self.prey_position) - np.array(nearest_shelter)) < np.linalg.norm(np.array(self.prey_position) - np.array(nearest_hunter)):
                # 朝最近的避难所移动
                best_move = self.prey_position
                min_distance = np.linalg.norm(np.array(self.prey_position) - np.array(nearest_shelter))
                for move in move_options:
                    new_position = (self.prey_position[0] + move[0], self.prey_position[1] + move[1])
                    new_distance = np.linalg.norm(np.array(new_position) - np.array(nearest_shelter))
                    if new_distance < min_distance and self._is_within_grid(new_position):
                        best_move = new_position
                        min_distance = new_distance

                self.prey_position = best_move

                if self.prey_position == nearest_shelter:
                    self.in_shelter = True
            else:
                # 远离最近的追捕者
                best_move = self.prey_position
                max_distance = np.linalg.norm(np.array(self.prey_position) - np.array(nearest_hunter))
                for move in move_options:
                    new_position = (self.prey_position[0] + move[0], self.prey_position[1] + move[1])
                    new_distance = np.linalg.norm(np.array(new_position) - np.array(nearest_hunter))
                    if new_distance > max_distance and self._is_within_grid(new_position):
                        best_move = new_position
                        max_distance = new_distance

                self.prey_position = best_move

    def move_hunters(self):
        move_options = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for i, hunter in enumerate(self.hunter_positions):
            distance_to_prey = np.linalg.norm(np.array(hunter) - np.array(self.prey_position))
            if self.in_shelter or distance_to_prey > self.distance_threshold:
                # 随机移动
                while True:
                    move = random.choice(move_options)
                    new_position = (hunter[0] + move[0], hunter[1] + move[1])
                    if self._is_within_grid(new_position) and new_position not in self.shelter_positions:
                        self.hunter_positions[i] = new_position
                        break
            else:
                # 向逃亡者移动
                best_move = hunter
                min_distance = distance_to_prey
                for move in move_options:
                    new_position = (hunter[0] + move[0], hunter[1] + move[1])
                    new_distance = np.linalg.norm(np.array(new_position) - np.array(self.prey_position))
                    if new_distance < min_distance and self._is_within_grid(new_position) and new_position not in self.shelter_positions:
                        best_move = new_position
                        min_distance = new_distance
                self.hunter_positions[i] = best_move

    def _is_within_grid(self, position):
        return 0 <= position[0] < self.grid_size and 0 <= position[1] < self.grid_size

    def check_capture(self):
        return any(np.array_equal(hunter, self.prey_position) for hunter in self.hunter_positions)

    def step(self):
        self.move_prey()
        self.move_hunters()
        self.steps.append((copy.deepcopy(self.hunter_positions), copy.deepcopy(self.prey_position), copy.deepcopy(self.shelter_positions)))
        return self.check_capture()

    def run(self, max_steps=100):
        for step in range(max_steps):
            captured = self.step()
            if captured:
                print(f"Prey captured in {step + 1} steps!")
                break
        else:
            print("Prey escaped.")

# 创建一个20x20的网格环境，5个追捕者，多个避难所，设定距离阈值为5
shelter_positions = [(5, 5), (10, 10), (15, 15), (3, 17), (17, 3), (7, 7), (13, 13), (1, 1), (18, 18), (9, 9)]
env = MultiAgentEnv(grid_size=20, num_hunters=5, shelter_positions=shelter_positions, distance_threshold=5)
env.run()

# 动画部分
fig, ax = plt.subplots()
ax.set_xlim(0, env.grid_size)
ax.set_ylim(0, env.grid_size)

hunter_scatter = ax.scatter([], [], c='red', label='Hunter')
prey_scatter = ax.scatter([], [], c='blue', label='Prey')
shelter_scatter = ax.scatter([], [], c='green', label='Shelter')
ax.legend()

def init():
    hunter_scatter.set_offsets(np.empty((0, 2)))
    prey_scatter.set_offsets(np.empty((0, 2)))
    shelter_scatter.set_offsets(shelter_positions)
    return hunter_scatter, prey_scatter, shelter_scatter

def update(frame):
    hunters, prey, shelters = env.steps[frame]
    hunter_scatter.set_offsets(hunters)
    prey_scatter.set_offsets([prey])
    return hunter_scatter, prey_scatter, shelter_scatter

ani = animation.FuncAnimation(fig, update, frames=len(env.steps), init_func=init, blit=True, repeat=False)
plt.show()

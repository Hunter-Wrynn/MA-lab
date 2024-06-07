import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# 定义行人类
class Pedestrian:
    def __init__(self, position, destination, speed=1.3, fov=120, dmax=10):
        self.position = np.array(position, dtype=float)
        self.destination = np.array(destination, dtype=float)
        self.speed = speed
        self.fov = np.radians(fov / 2)  # 视野范围的一半（转换为弧度）
        self.dmax = dmax
        self.direction = self.calculate_direction()

    def calculate_direction(self):
        direction = self.destination - self.position
        norm = np.linalg.norm(direction)
        if norm == 0:
            return np.array([0, 0])
        return direction / norm

    def distance_to_collision(self, alpha, obstacles, other_pedestrians):
        min_distance = self.dmax
        for obs in obstacles:
            direction_to_obs = obs - self.position
            distance_to_obs = np.linalg.norm(direction_to_obs)
            angle_to_obs = np.arctan2(direction_to_obs[1], direction_to_obs[0]) - alpha
            if abs(angle_to_obs) <= self.fov:
                min_distance = min(min_distance, distance_to_obs)

        for ped in other_pedestrians:
            direction_to_ped = ped.position - self.position
            distance_to_ped = np.linalg.norm(direction_to_ped)
            angle_to_ped = np.arctan2(direction_to_ped[1], direction_to_ped[0]) - alpha
            if abs(angle_to_ped) <= self.fov:
                min_distance = min(min_distance, distance_to_ped)

        return min_distance

    def calculate_new_direction(self, obstacles, other_pedestrians):
        alpha0 = np.arctan2(self.destination[1] - self.position[1], self.destination[0] - self.position[0])
        best_direction = alpha0
        min_d = float('inf')
        for alpha in np.linspace(alpha0 - self.fov, alpha0 + self.fov, 100):
            d = self.dmax ** 2 + self.distance_to_collision(alpha, obstacles, other_pedestrians) ** 2 - 2 * self.dmax * self.distance_to_collision(alpha, obstacles, other_pedestrians) * np.cos(alpha0 - alpha)
            if d < min_d:
                min_d = d
                best_direction = alpha
        self.direction = np.array([np.cos(best_direction), np.sin(best_direction)])

    def move(self, time_step=0.1, obstacles=[], other_pedestrians=[]):
        self.calculate_new_direction(obstacles, other_pedestrians)
        self.position += self.direction * self.speed * time_step


# 初始化行人和障碍物
pedestrians = [
    Pedestrian(position=[2, 2], destination=[10, 10]),
    Pedestrian(position=[10, 0], destination=[0, 10]),
    Pedestrian(position=[5, 5], destination=[10, 0])
]

obstacles = [
    np.array([5, 2]),
    np.array([7, 7])
]

# 设置模拟参数
time_step = 0.1
num_steps = 100

# 创建绘图
fig, ax = plt.subplots()
ax.set_xlim(-1, 11)
ax.set_ylim(-1, 11)

# 绘制行人和障碍物的初始位置
start_points = [ax.plot(ped.position[0], ped.position[1], 'bo')[0] for ped in pedestrians]  # 起点用蓝色圆点表示
end_points = [ax.plot(ped.destination[0], ped.destination[1], 'go')[0] for ped in pedestrians]  # 终点用绿色圆点表示
obs_points = [ax.plot(obs[0], obs[1], 'rx')[0] for obs in obstacles]  # 障碍物用红色叉表示
paths = [ax.plot(ped.position[0], ped.position[1], 'ko')[0] for ped in pedestrians]  # 行人用黑色圆点表示


def update(num):
    for i, ped in enumerate(pedestrians):
        other_peds = [p for j, p in enumerate(pedestrians) if j != i]
        ped.move(time_step, obstacles, other_peds)
        paths[i].set_data(ped.position[0], ped.position[1])
    return paths


ani = animation.FuncAnimation(fig, update, frames=num_steps, interval=100, blit=True)

plt.show()

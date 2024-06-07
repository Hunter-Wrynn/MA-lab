import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.distance import pdist, squareform

# Initialize plot
L = 900  # Double the size of the map
fig, ax = plt.subplots()
ax.set_xlim(-L, L)
ax.set_ylim(-L, L)
points, = ax.plot([], [], '.', color='black')
plt.axis('off')

# Define obstacle properties
obstacle_centers = [np.array([0, 0]), np.array([300, 300]), np.array([-300, -300])]
obstacle_radius = 50
obstacle_repulsion = 100  # Increase repulsion to ensure birds avoid the obstacle

# Plot obstacles
for center in obstacle_centers:
    circle = plt.Circle(center, obstacle_radius, color='blue', fill=True)
    ax.add_patch(circle)

tsteps = 200  # Time steps
n = 300  # Number of birds
V0 = 20  # Initial velocity

wall_repulsion = 20
margin = 30  # Increase margin to ensure birds avoid the obstacle
max_speed = 20
neighbors_dist = 70  # Community distance (At which distance they apply rule 1 and rule 3)

# Rule 1: COHESION
R = 0.1  # velocity to center contribution

# Rule 2: SEPARATION
bird_repulsion = 7  # low values make more "stains" of birds, nice visualization
privacy = 14  # Avoid each other ,length. When they see each other at this distance,

# Rule 3: ALIGNMENT
match_velocity = 3

'''
x, y 代表 n只鸟在tsteps时候的 xy 轴坐标
'''
x = np.zeros((n, tsteps))
y = np.zeros((n, tsteps))

# Initialize positions ensuring they are not within the obstacles
for i in range(n):
    while True:
        x[i, 0] = np.random.uniform(low=-L, high=L)
        y[i, 0] = np.random.uniform(low=-L, high=L)
        distances = [np.sqrt((x[i, 0] - center[0])**2 + (y[i, 0] - center[1])**2) for center in obstacle_centers]
        if all(dist > (obstacle_radius + margin) for dist in distances):
            break

# Randomize initial velocity
x[:, 1] = x[:, 0] + np.random.uniform(low=-V0, high=V0, size=(int(n),))
y[:, 1] = y[:, 0] + np.random.uniform(low=-V0, high=V0, size=(int(n),))


# Cohesion
def moveToCenter(x0, y0, neighbors_dist, n, R):
    '''
    此处的x0, y0 代表 0 时刻 n 只鸟的坐标，因此 x0和y0的大小为 (n,)
    '''
    m = squareform(pdist(np.transpose([x0, y0])))
    idx = (m < neighbors_dist)
    center_x = np.zeros(n)
    center_y = np.zeros(n)
    vx = np.zeros(n)
    vy = np.zeros(n)
    for i in range(0, n - 1):
        center_x[i] = np.mean(x0[idx[i,]])
        center_y[i] = np.mean(y0[idx[i]])
        vx[i] = -(x0[i] - center_x[i]) * R
        vy[i] = -(y0[i] - center_y[i]) * R

    return vx, vy


def avoidOthers(x0, y0, n, privacy, bird_repulsion):
    dist = squareform(pdist(np.transpose([x0, y0])))
    idxmat = (dist < privacy) & (dist != 0)
    idx = np.transpose(np.array(np.where(idxmat)))

    vx = np.zeros(n)
    vy = np.zeros(n)

    vx[idx[:, 0]] = (x0[idx[:, 0]] - x0[idx[:, 1]]) * bird_repulsion
    vy[idx[:, 0]] = (y0[idx[:, 0]] - y0[idx[:, 1]]) * bird_repulsion
    return vx, vy


def matchVelocities(x_prev, y_prev, x0, y0, n, neighbors_dist, match_velocity):
    m = squareform(pdist(np.transpose([x_prev, y_prev])))
    idx = (m <= neighbors_dist)

    vmeans_x = np.zeros(n)
    vmeans_y = np.zeros(n)
    for i in range(0, n - 1):
        vmeans_x[i] = np.mean(x0[idx[i, :]] - x_prev[idx[i,]])
        vmeans_y[i] = np.mean(y0[idx[i, :]] - y_prev[idx[i,]])

    return vmeans_x * match_velocity, vmeans_y * match_velocity


def avoidObstacles(x0, y0, obstacle_centers, obstacle_radius, n, obstacle_repulsion):
    vx = np.zeros(n)
    vy = np.zeros(n)
    for center in obstacle_centers:
        dist = np.sqrt((x0 - center[0])**2 + (y0 - center[1])**2)
        idx = dist < (obstacle_radius + margin)  # Add margin to ensure avoidance
        vx[idx] += (x0[idx] - center[0]) * obstacle_repulsion / (dist[idx] + 1e-6)
        vy[idx] += (y0[idx] - center[1]) * obstacle_repulsion / (dist[idx] + 1e-6)
    return vx, vy


def move(x0, y0, x_prev, y_prev, n, neighbors_dist, R, privacy, bird_repulsion, match_velocity, L, margin,
         wall_repulsion, max_speed, obstacle_centers, obstacle_radius, obstacle_repulsion):
    vx1, vy1 = moveToCenter(x0, y0, neighbors_dist, n, R)
    vx2, vy2 = avoidOthers(x0, y0, n, privacy, bird_repulsion)
    vx3, vy3 = matchVelocities(x_prev, y_prev, x0, y0, n, neighbors_dist, match_velocity)
    vx4, vy4 = avoidObstacles(x0, y0, obstacle_centers, obstacle_radius, n, obstacle_repulsion)

    vx = x0 - x_prev + vx1 + vx2 + vx3 + vx4
    vy = y0 - y_prev + vy1 + vy2 + vy3 + vy4

    v_norm = np.zeros((2, n))
    v_vector = np.array([vx, vy])
    norm = np.linalg.norm(v_vector, axis=0)
    v_norm[:, norm != 0] = v_vector[:, norm != 0] / norm[norm != 0] * max_speed

    vx = v_norm[0, :]
    vy = v_norm[1, :]

    right_border_dist = L - x0
    left_border_dist = x0 + L
    upper_border_dist = L - y0
    bottom_border_dist = y0 + L

    vx[right_border_dist < margin] = vx[right_border_dist < margin] - wall_repulsion
    vx[left_border_dist < margin] = vx[left_border_dist < margin] + wall_repulsion
    vy[upper_border_dist < margin] = vy[upper_border_dist < margin] - wall_repulsion
    vy[bottom_border_dist < margin] = vy[bottom_border_dist < margin] + wall_repulsion

    x1 = x0 + vx
    y1 = y0 + vy

    # Ensure birds do not enter the obstacles
    for center in obstacle_centers:
        dist_to_obstacle = np.sqrt((x1 - center[0])**2 + (y1 - center[1])**2)
        inside_obstacle_idx = dist_to_obstacle < (obstacle_radius + margin)
        x1[inside_obstacle_idx] = x0[inside_obstacle_idx] - vx[inside_obstacle_idx]
        y1[inside_obstacle_idx] = y0[inside_obstacle_idx] - vy[inside_obstacle_idx]

    x1 = np.round(x1)
    y1 = np.round(y1)

    return x1, y1


for t in range(1, tsteps - 1):
    x[:, t + 1], y[:, t + 1] = move(x[:, t], y[:, t], x[:, t - 1], y[:, t - 1],
                                    n, neighbors_dist, R, privacy, bird_repulsion, match_velocity,
                                    L, margin, wall_repulsion, max_speed,
                                    obstacle_centers, obstacle_radius, obstacle_repulsion)


def init():
    points.set_data([], [])
    return points,


def animate(i):
    xx = x[:, i]
    yy = y[:, i]
    points.set_data(xx, yy)
    return points,


anim = FuncAnimation(fig, animate, init_func=init,
                     frames=tsteps - 2, interval=80, blit=True)

plt.show()

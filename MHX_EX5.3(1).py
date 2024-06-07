import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Initialize parameters
N = 1000
initial_infected = 10
beta = 0.5
time_steps = 100  # Reduced time steps
infection_radius = 0.05  # Increased infection radius

# Initialize population
population = np.zeros(N, dtype=int)
population[:initial_infected] = 1  # Initial infected individuals

# Initialize positions
positions = np.random.rand(N, 2)

# SI model update function
def update(frame, population, positions, beta, infection_radius):
    new_population = population.copy()
    for i in range(N):
        if population[i] == 1:
            for j in range(N):
                if population[j] == 0:
                    distance = np.linalg.norm(positions[i] - positions[j])
                    if distance < infection_radius and np.random.rand() < beta:
                        new_population[j] = 1
    population[:] = new_population
    scat.set_offsets(positions)
    scat.set_array(population)
    return scat,

# Create plot
fig, ax = plt.subplots()
scat = ax.scatter(positions[:, 0], positions[:, 1], c=population, cmap='bwr', s=5)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title('SI Model')

# Create legend
healthy_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=5, label='Healthy')
infected_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=5, label='Infected')
ax.legend(handles=[healthy_patch, infected_patch], loc='upper right')

# Create animation
ani = animation.FuncAnimation(fig, update, frames=time_steps, fargs=(population, positions, beta, infection_radius), interval=100, repeat=False)

plt.show()

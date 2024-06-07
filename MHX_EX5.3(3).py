import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap

# Initialize parameters
N = 1000
initial_infected = 10
beta = 0.3
gamma = 0.1  # Recovery rate
time_steps = 200  # Increased time steps
infection_radius = 0.1

# Initialize population: 0 = Susceptible, 1 = Infected
population = np.zeros(N, dtype=int)
population[:initial_infected] = 1  # Initial infected individuals

# Initialize positions
positions = np.random.rand(N, 2)

# Custom colormap: Blue for susceptible, Red for infected
cmap = ListedColormap(['blue', 'red'])

# SIS model update function
def update(frame, population, positions, beta, gamma, infection_radius):
    new_population = population.copy()
    for i in range(N):
        if population[i] == 1:
            # Check for infection spread
            for j in range(N):
                if population[j] == 0:
                    distance = np.linalg.norm(positions[i] - positions[j])
                    if distance < infection_radius and np.random.rand() < beta:
                        new_population[j] = 1
            # Check for recovery (back to susceptible)
            if np.random.rand() < gamma:
                new_population[i] = 0

    # Print the counts for debugging
    num_infected = np.sum(new_population == 1)
    num_susceptible = np.sum(new_population == 0)
    print(f"Step {frame}: Infected = {num_infected}, Susceptible = {num_susceptible}")

    population[:] = new_population
    scat.set_offsets(positions)
    scat.set_array(population)
    return scat,

# Create plot
fig, ax = plt.subplots()
scat = ax.scatter(positions[:, 0], positions[:, 1], c=population, cmap=cmap, s=5, vmin=0, vmax=1)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title('SIS Model')

# Create legend
healthy_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=5, label='Susceptible')
infected_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=5, label='Infected')
ax.legend(handles=[healthy_patch, infected_patch], loc='upper right')

# Create animation
ani = animation.FuncAnimation(fig, update, frames=time_steps,
                              fargs=(population, positions, beta, gamma, infection_radius), interval=100, repeat=False)

plt.show()

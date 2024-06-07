import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap

# Initialize parameters
N = 1000
initial_infected = 10
beta = 0.3  # Infection rate
sigma = 0.1  # Rate at which exposed individuals become infected
gamma = 0.1  # Recovery rate
time_steps = 200
infection_radius = 0.1

# Initialize population: 0 = Susceptible, 1 = Exposed, 2 = Infected, 3 = Recovered
population = np.zeros(N, dtype=int)
population[:initial_infected] = 2  # Initial infected individuals

# Initialize positions
positions = np.random.rand(N, 2)

# Custom colormap: Blue for susceptible, Yellow for exposed, Red for infected, Green for recovered
cmap = ListedColormap(['blue', 'yellow', 'red', 'green'])

# SEIR model update function
def update(frame, population, positions, beta, sigma, gamma, infection_radius):
    new_population = population.copy()
    for i in range(N):
        if population[i] == 2:  # Infected individuals
            # Check for infection spread
            for j in range(N):
                if population[j] == 0:
                    distance = np.linalg.norm(positions[i] - positions[j])
                    if distance < infection_radius and np.random.rand() < beta:
                        new_population[j] = 1  # Susceptible to exposed
            # Check for recovery
            if np.random.rand() < gamma:
                new_population[i] = 3  # Infected to recovered
        elif population[i] == 1:  # Exposed individuals
            # Check for becoming infected
            if np.random.rand() < sigma:
                new_population[i] = 2  # Exposed to infected

    # Print the counts for debugging
    num_susceptible = np.sum(new_population == 0)
    num_exposed = np.sum(new_population == 1)
    num_infected = np.sum(new_population == 2)
    num_recovered = np.sum(new_population == 3)
    print(f"Step {frame}: Susceptible = {num_susceptible}, Exposed = {num_exposed}, Infected = {num_infected}, Recovered = {num_recovered}")

    population[:] = new_population
    scat.set_offsets(positions)
    scat.set_array(population)
    return scat,

# Create plot
fig, ax = plt.subplots()
scat = ax.scatter(positions[:, 0], positions[:, 1], c=population, cmap=cmap, s=5, vmin=0, vmax=3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title('SEIR Model')

# Create legend
susceptible_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=5, label='Susceptible')
exposed_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=5, label='Exposed')
infected_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=5, label='Infected')
recovered_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=5, label='Recovered')
ax.legend(handles=[susceptible_patch, exposed_patch, infected_patch, recovered_patch], loc='upper right')

# Create animation
ani = animation.FuncAnimation(fig, update, frames=time_steps,
                              fargs=(population, positions, beta, sigma, gamma, infection_radius), interval=100, repeat=False)

plt.show()

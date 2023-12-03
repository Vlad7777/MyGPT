#MyGPT_Travel

"""
The Traveling Salesman Problem (TSP) is a classic optimization problem where the goal is to find the shortest possible tour that visits a set of cities and returns to the starting city. Solving the TSP optimally for a large number of cities is a computationally intensive task, but various algorithms can be used to find approximate solutions. One such algorithm is the Held-Karp dynamic programming approach for small instances, and the 2-Opt algorithm for larger instances. Here's an example using the 2-Opt algorithm:

python
Copy code
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from itertools import permutations

def total_distance(tour, distances):
    return sum(distances[tour[i], tour[i + 1]] for i in range(len(tour) - 1)) + distances[tour[-1], tour[0]]

def two_opt(tour, distances):
    n = len(tour)
    improved = True

    while improved:
        improved = False

        for i in range(1, n - 2):
            for j in range(i + 1, n):
                if j - i == 1:
                    continue  # changes nothing, skip then
                new_tour = tour[:i] + tour[i:j][::-1] + tour[j:]
                if total_distance(new_tour, distances) < total_distance(tour, distances):
                    tour = new_tour
                    improved = True

    return tour

# Generate random cities
np.random.seed(42)
num_cities = 10
cities = np.random.rand(num_cities, 2)

# Calculate pairwise distances between cities
distances = distance.cdist(cities, cities, 'euclidean')

# Find the optimal tour using 2-Opt
initial_tour = np.arange(num_cities)
optimized_tour = two_opt(initial_tour, distances)

# Plot the result
plt.figure(figsize=(8, 8))
plt.scatter(cities[:, 0], cities[:, 1], c='red', marker='o', label='Cities')
plt.plot(cities[optimized_tour, 0], cities[optimized_tour, 1], c='blue', linestyle='-', linewidth=2, label='Optimal Tour')
plt.title('Traveling Salesman Problem - 2-Opt Algorithm')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.show()

"""
In this example:

Random cities are generated using NumPy.
Pairwise distances between cities are calculated using scipy.spatial.distance.cdist.
The 2-Opt algorithm is applied to find an optimized tour.
The result is plotted using Matplotlib.
Make sure to have the required libraries installed:

bash
Copy code
pip install numpy matplotlib scipy
Keep in mind that for large instances of the TSP, more advanced algorithms or heuristics may be required for obtaining feasible solutions in a reasonable amount of time.


"""



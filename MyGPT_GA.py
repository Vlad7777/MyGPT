#MyGPT_GA

"""
Genetic algorithms are optimization algorithms inspired by the process of natural selection. They are used to find approximate solutions to optimization and search problems. Here's a simple example of a genetic algorithm implemented in Python:

python
Copy code
"""


import random

# Define the target phrase
target_phrase = "Hello, World!"

# Define the population size and mutation rate
population_size = 100
mutation_rate = 0.01

# Function to generate a random individual
def generate_individual(length):
    return ''.join(random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ,.!?") for _ in range(length))

# Function to calculate the fitness of an individual
def calculate_fitness(individual, target):
    return sum(1 for i, j in zip(individual, target) if i == j)

# Function to perform crossover between two individuals
def crossover(parent1, parent2):
    split_point = random.randint(0, len(parent1) - 1)
    child1 = parent1[:split_point] + parent2[split_point:]
    child2 = parent2[:split_point] + parent1[split_point:]
    return child1, child2

# Function to perform mutation on an individual
def mutate(individual, mutation_rate):
    return ''.join(
        gene if random.random() > mutation_rate else random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ,.!?")
        for gene in individual
    )

# Initialize the population
population = [generate_individual(len(target_phrase)) for _ in range(population_size)]

# Main loop
generation = 1
while True:
    # Calculate fitness for each individual
    fitness_scores = [calculate_fitness(individual, target_phrase) for individual in population]

    # Check for a perfect match
    if max(fitness_scores) == len(target_phrase):
        print(f"Target phrase '{target_phrase}' found in generation {generation}")
        break

    # Select parents for crossover based on fitness
    parents = random.choices(population, weights=fitness_scores, k=2)

    # Perform crossover
    offspring1, offspring2 = crossover(parents[0], parents[1])

    # Perform mutation on offspring
    offspring1 = mutate(offspring1, mutation_rate)
    offspring2 = mutate(offspring2, mutation_rate)

    # Replace two least fit individuals with the offspring
    min_fitness_index1 = fitness_scores.index(min(fitness_scores))
    fitness_scores[min_fitness_index1] = calculate_fitness(offspring1, target_phrase)
    population[min_fitness_index1] = offspring1

    min_fitness_index2 = fitness_scores.index(min(fitness_scores))
    fitness_scores[min_fitness_index2] = calculate_fitness(offspring2, target_phrase)
    population[min_fitness_index2] = offspring2

    # Print progress
    if generation % 100 == 0:
        best_individual = population[fitness_scores.index(max(fitness_scores))]
        print(f"Generation {generation}: {best_individual} (Fitness: {max(fitness_scores)})")

    generation += 1


#This example aims to evolve a population of strings to match the target phrase "Hello, World!". The genetic algorithm involves selection, crossover, and mutation operations on the population until a perfect match is found. Keep in mind that this is a basic illustration, and fine-tuning and modifications may be needed for more complex problems.





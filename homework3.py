import math
import random
from typing import List, Tuple
import time

class Coordinate:
    def __init__(self, x, y, z):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
    
    def __repr__(self) -> str:
        return f"{int(self.x)} {int(self.y)} {int(self.z)}"

class Path:
    def __init__(self, coordinates):
        self.coordinates = []
        for cord in coordinates:
            self.coordinates.append(cord)
        self.n = len(self.coordinates)
        self.length = self.calc_length()

    def calc_length(self) -> float:
        total_distance = 0
        for i in range(1, self.n):
            total_distance += euclidean_distance(self.coordinates[i], self.coordinates[i - 1])
        total_distance += euclidean_distance(self.coordinates[0], self.coordinates[self.n - 1])
        return total_distance
    
    def partial_length(self, s, e) -> float:
        total_distance = 0
        for i in range(s + 1, e + 1):
            total_distance += euclidean_distance(self.coordinates[i], self.coordinates[i - 1])
        return total_distance
    
    def __repr__(self) -> str:
        path_string = ''
        for cord in self.coordinates:
            path_string += cord.__repr__() + "\n"
        path_string += self.coordinates[0].__repr__()
        return path_string

def euclidean_distance(a, b) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)

def create_init_paths(original_path, size) -> List[Path]:
    paths = []
    coordinates = original_path.coordinates
    n = len(coordinates)

    for i in range(n):
        paths.append(generate_greedy_path(coordinates, i))
        # paths.append(generate_heuristic_path(coordinates, i))

    # for _ in range(size - n):
    #     random_path = random.sample(coordinates, len(coordinates))
    #     paths.append(Path(random_path))

    return paths

def generate_greedy_path(coordinates, start_idx) -> Path:
    start = coordinates[start_idx]
    remaining = set(coordinates)
    path = [start]
    remaining.remove(start)
    
    while remaining:
        nearest = min(remaining, key = lambda cord: euclidean_distance(path[-1], cord))
        path.append(nearest)
        remaining.remove(nearest)
    
    return Path(path)

def generate_heuristic_path(coordinates, start_idx) -> Path:
    start = coordinates[start_idx]
    remaining = set(coordinates)
    path = [start]
    remaining.remove(start)
    
    while remaining:
        nearest = min(remaining, key = lambda cord: euclidean_distance(path[-1], cord) + euclidean_distance(path[0], cord))
        path.append(nearest)
        remaining.remove(nearest)
    
    return Path(path)

def generate_weights(paths, bias) -> List[float]:
    total_length = 0.0
    for path in paths:
        total_length += 1 / path.length
    weighted_probs = []
    for path in paths:
        weighted_probs.append((1 / path.length) * bias / total_length)

    return weighted_probs

def choose_parents(paths, weighted_probs) -> Tuple[Path, Path]:
    parent1 = random.choices(paths, weights = weighted_probs, k = 1)[0]
    parent2 = parent1
    while parent2 == parent1:
        parent2 = random.choices(paths, weights = weighted_probs, k = 1)[0]
    return parent1, parent2

def create_child_half(parent1, parent2) -> Path:
    n = len(parent1.coordinates)
    start = random.randint(0, int(n / 2) - 1)
    end = start + int(n / 2)
    parent1_part_len = parent1.partial_length(start, end)
    parent2_part_len = parent2.partial_length(start, end)
    shorter = parent1
    longer = parent2
    if parent1_part_len > parent2_part_len:
        shorter = parent2
        longer = parent1
    child = [None] * n
    
    for i in range(start, end + 1):
        child[i] = shorter.coordinates[i]
    
    i = 0
    for cord in longer.coordinates:
        if cord not in child:
            while(child[i] is not None):
                i += 1
            child[i] = cord

    return Path(child)

def create_child_random(parent1, parent2) -> Path:
    n = len(parent1.coordinates)
    start, end = sorted(random.sample(range(n), 2))
    
    child = [None] * n
    for i in range(start, end + 1):
        child[i] = parent1.coordinates[i]
    
    i = 0
    for cord in parent2.coordinates:
        if cord not in child:
            while(child[i] is not None):
                i += 1
            child[i] = cord

    return Path(child)

def mutate(path, mutation_rate) -> Path:
    new_coordinates = path.coordinates.copy()
    n = len(new_coordinates)
    for i in range(n):
        if random.random() < mutation_rate:
            j = random.randint(0, n - 1)
            new_coordinates[i], new_coordinates[j] = new_coordinates[j], new_coordinates[i]
    
    return Path(new_coordinates)

def swap(path, mutation_rate) -> Path:
    new_coordinates = path.coordinates.copy()
    n = len(new_coordinates)

    if random.random() < mutation_rate:
        i = random.randint(0, n - 1)
        j = i
        while j == i:
            j = random.randint(0, n - 1)
        new_coordinates[i], new_coordinates[j] = new_coordinates[j], new_coordinates[i]
    
    return Path(new_coordinates)

def simulated_annealing(path, initial_temp, cooling_rate, iterations, mutation_rate):
    current_path = path
    best_path = current_path
    temp = initial_temp

    for _ in range(iterations):
        # new_path = mutate(current_path, mutation_rate)
        new_path = swap(current_path, mutation_rate)
        len_diff = new_path.length - current_path.length
        
        if len_diff < 0 or random.random() < math.exp(-len_diff / temp):
            current_path = new_path

            if current_path.length < best_path.length:
                best_path = current_path
        
        temp *= cooling_rate

    return best_path

def genetic_algorithm(path, population_size, generations, bias, elite_size, mutation_rate, initial_temp, cooling_rate) -> Path:
    paths = create_init_paths(path, population_size)
    n  = len(path.coordinates)

    for _ in range(generations):
    # for _ in range(elite_size):
        paths = sorted(paths, key = lambda p: p.length)
        new_paths = paths[:elite_size]
        weighted_probs = generate_weights(paths, bias) 
        # for _ in range(population_size - elite_size):
        for _ in range(elite_size):
            parent1, parent2 = choose_parents(paths, weighted_probs)
            # child = create_child_half(parent1, parent2)
            child = create_child_random(parent1, parent2)
            # child = mutate(child, mutation_rate)
            child = simulated_annealing(child, initial_temp, cooling_rate, elite_size, mutation_rate)
            new_paths.append(child)
        paths = new_paths
    
    shortest_path = min(paths, key = lambda p: p.length)
    return shortest_path

def main():
    input = open('input.txt', 'r')
    content = input.read()
    lines = content.split("\n")
    input.close()

    n = int(lines[0])
    coordinates = []
    for i in range(0, n):
        line = lines[i + 1]
        cord = line.split(' ')
        coordinates.append(Coordinate(cord[0], cord[1], cord[2]))

    original_path = Path(coordinates)
    population_size = n
    generations = n
    bias = 1
    elite_size = math.ceil(n * 0.01)
    mutation_rate = 1

    initial_temp = 100
    cooling_rate = 0.95

    shortest_path = genetic_algorithm(original_path, population_size, generations, bias, elite_size, mutation_rate, initial_temp, cooling_rate)
    output = open('output.txt', 'w')
    output.write(str(shortest_path.length) + "\n")
    output.write(shortest_path.__repr__())
    output.close()

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"{elapsed_time:.2f} seconds")
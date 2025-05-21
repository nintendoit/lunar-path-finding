import rasterio
import numpy as np
import cv2
import heapq
import matplotlib.pyplot as plt

# 📥 Load NASA Lunar Terrain Data (Optimized for CPU)
def load_nasa_terrain(filename, scale_factor=0.2, max_size=1000):
    with rasterio.open(filename) as dataset:
        terrain_data = dataset.read(1)  # Read elevation band

    # Limit terrain size dynamically
    if max(terrain_data.shape) > max_size:
        scale_factor = max_size / max(terrain_data.shape)

    # Resize terrain for faster processing
    terrain_data = cv2.resize(terrain_data, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

    # Normalize terrain data (scale to 0.5 - 1.5)
    terrain_data = (terrain_data - np.min(terrain_data)) / (np.max(terrain_data) - np.min(terrain_data)) * 1.0 + 0.5

    return terrain_data

# 🌡 Generate Synthetic Lunar Thermal & Power Maps (Optimized)
def generate_thermal_map(shape):
    return np.random.uniform(100, 400, shape) * np.tile(np.linspace(0.7, 1, shape[1]), (shape[0], 1))

def generate_power_map(shape):
    return np.random.uniform(0.3, 1.0, shape) * np.tile(np.linspace(0.2, 1, shape[1]), (shape[0], 1))

# 📌 Optimized A* Algorithm (Efficient Memory Use)
def heuristic(x, y, goal_x, goal_y, terrain, thermal, power):
    return abs(goal_x - x) + abs(goal_y - y) + terrain[x, y] * 0.5 + thermal[x, y] * 0.05 + (1 - power[x, y]) * 0.05

def astar(start, goal, terrain, thermal, power):
    GRID_SIZE = terrain.shape
    open_set = [(0, start)]
    g_cost = {start: 0}
    came_from = {}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        x, y = current
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE[0] and 0 <= ny < GRID_SIZE[1]:
                new_cost = g_cost[current] + terrain[nx, ny] * 0.5 + thermal[nx, ny] * 0.05 + (1 / power[nx, ny]) * 0.05
                if (nx, ny) not in g_cost or new_cost < g_cost[(nx, ny)]:
                    g_cost[(nx, ny)] = new_cost
                    heapq.heappush(open_set, (new_cost + heuristic(nx, ny, goal[0], goal[1], terrain, thermal, power), (nx, ny)))
                    came_from[(nx, ny)] = current

    return None

# 📌 Optimized APF Algorithm (No GPU, Faster Execution)
def apf(start, goal, terrain, thermal, power, alpha=0.1, beta=0.5, max_iter=500):
    path = [start]
    current = np.array(start, dtype=np.int32)

    for _ in range(max_iter):
        # Compute Attractive Force
        attractive_force = alpha * (np.array(goal) - current)

        # Compute Repulsive Force from Obstacles (Avoid High Terrain)
        repulsive_force = np.zeros(2, dtype=np.float32)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = current + np.array([dx, dy])
            if 0 <= neighbor[0] < terrain.shape[0] and 0 <= neighbor[1] < terrain.shape[1]:
                obstacle_penalty = max(0, 1.5 - terrain[neighbor[0], neighbor[1]])
                repulsive_force += beta * obstacle_penalty * -np.sign(neighbor - current)

        # Compute Final Move
        force = attractive_force + repulsive_force
        step = np.sign(force).astype(np.int32)
        next_pos = current + step

        # Prevent Out of Bounds
        if 0 <= next_pos[0] < terrain.shape[0] and 0 <= next_pos[1] < terrain.shape[1]:
            current = next_pos
            path.append(tuple(current))

        # Stop if reached goal
        if tuple(current) == goal:
            break

    return path

# 📌 Optimized Bug Algorithm (Faster Obstacle Handling)
def bug_algorithm(start, goal, terrain, max_steps=3000):
    current = np.array(start, dtype=np.int32)
    path = [tuple(current)]

    while tuple(current) != goal and len(path) < max_steps:
        next_step = current + np.sign(np.array(goal) - current)
        if 0 <= next_step[0] < terrain.shape[0] and 0 <= next_step[1] < terrain.shape[1] and terrain[next_step[0], next_step[1]] < 1.2:
            current = next_step
        else:
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                boundary_step = current + np.array([dx, dy])
                if 0 <= boundary_step[0] < terrain.shape[0] and 0 <= boundary_step[1] < terrain.shape[1] and terrain[boundary_step[0], boundary_step[1]] < 1.2:
                    current = boundary_step
                    break
        path.append(tuple(current))

    return path

# 📌 Visualization (Fast Rendering)
def visualize_path(image, path, title):
    if path is None:
        print(f"⚠ {title} could not find a path.")
        return

    output = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for i in range(1, len(path)):
        cv2.line(output, (path[i-1][1], path[i-1][0]), (path[i][1], path[i][0]), (0, 0, 255), 1)

    plt.figure(figsize=(6, 6))
    plt.imshow(output)
    plt.title(title)
    plt.axis("off")
    plt.show()

# 📌 Path Cost Calculation
def calculate_path_cost(path, terrain, thermal, power):
    if path is None:
        return float('inf')
    return sum(terrain[x, y] * 0.5 + thermal[x, y] * 0.05+ (1 / power[x, y]) * 0.05 for x, y in path)

# 📌 Run Optimized Navigation
def run_lunar_navigation(filename):
    print("📥 Loading NASA Terrain Data...")
    terrain = load_nasa_terrain(filename, scale_factor=0.2)  # Reduce size to 20%
    thermal = generate_thermal_map(terrain.shape)
    power = generate_power_map(terrain.shape)

    start, goal = (5, 5), (terrain.shape[0] - 5, terrain.shape[1] - 5)

    print("🔄 Running A* Algorithm...")
    path_astar = astar(start, goal, terrain, thermal, power)
    print(f"A* Path Cost: {calculate_path_cost(path_astar, terrain, thermal, power)}")
    visualize_path(terrain, path_astar, "A* Path")

    print("🔄 Running Bug Algorithm...")
    path_bug = bug_algorithm(start, goal, terrain)
    print(f"Bug Algorithm Path Cost: {calculate_path_cost(path_bug, terrain, thermal, power)}")
    visualize_path(terrain, path_bug, "Bug Algorithm Path")

    print("🔄 Running APF Algorithm...")
    path_apf = apf(start, goal, terrain, thermal, power)
    print(f"APF Path Cost: {calculate_path_cost(path_apf, terrain, thermal, power)}")
    visualize_path(terrain, path_apf, "APF Path")

# 📌 Run
run_lunar_navigation("lunar.tif")
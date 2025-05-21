import rasterio
import numpy as np
import cv2
import heapq
import matplotlib.pyplot as plt

# 📥 Load NASA Lunar Terrain Data (Optimized for CPU)
def load_nasa_terrain(filename, scale_factor=0.2, max_size=1000):
    with rasterio.open(filename) as dataset:
        terrain_data = dataset.read(1)

    if max(terrain_data.shape) > max_size:
        scale_factor = max_size / max(terrain_data.shape)

    terrain_data = cv2.resize(terrain_data, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    terrain_data = (terrain_data - np.min(terrain_data)) / (np.max(terrain_data) - np.min(terrain_data)) * 1.0 + 0.5

    return terrain_data

# 🌡 Generate Synthetic Lunar Thermal & Power Maps
def generate_thermal_map(shape):
    return np.random.uniform(100, 400, shape) * np.tile(np.linspace(0.7, 1, shape[1]), (shape[0], 1))

def generate_power_map(shape):
    return np.random.uniform(0.3, 1.0, shape) * np.tile(np.linspace(0.2, 1, shape[1]), (shape[0], 1))

# 📌 Optimized A* Algorithm with Enhanced Heuristic
def heuristic(x, y, goal_x, goal_y, terrain, thermal, power, s_min, s_max, P_max, F):
    d_remaining = abs(goal_x - x) + abs(goal_y - y)
    h_terra = d_remaining * (s_min / s_max)
    h_th = d_remaining * F
    h_p = d_remaining * (1 - power[x, y] / P_max)
    return 0.5*h_terra + 0.2*h_th + 0.3*h_p

def astar(start, goal, terrain, thermal, power, s_min, s_max, P_max, F):
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
                h_cost = heuristic(nx, ny, goal[0], goal[1], terrain, thermal, power, s_min, s_max, P_max, F)

                if (nx, ny) not in g_cost or new_cost < g_cost[(nx, ny)]:
                    g_cost[(nx, ny)] = new_cost
                    heapq.heappush(open_set, (new_cost + h_cost, (nx, ny)))
                    came_from[(nx, ny)] = current

    return None

# 📌 Visualization of Path
def visualize_path(image, path, title):
    if path is None:
        print(f"⚠ {title} could not find a path.")
        return

    output = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for i in range(1, len(path)):
        cv2.line(output, (path[i - 1][1], path[i - 1][0]), (path[i][1], path[i][0]), (0, 0, 255), 1)

    plt.figure(figsize=(6, 6))
    plt.imshow(output)
    plt.title(title)
    plt.axis("off")
    plt.show()

# 📌 Run Full Lunar Navigation with Power Consumption Calculation
def run_lunar_navigation(filename):
    print("📥 Loading NASA Terrain Data...")
    terrain = load_nasa_terrain(filename)
    thermal = generate_thermal_map(terrain.shape)
    power = generate_power_map(terrain.shape)

    start, goal = (5, 5), (terrain.shape[0] - 5, terrain.shape[1] - 5)

    print("🔄 Running Optimized A* Algorithm...")
    path = astar(start, goal, terrain, thermal, power, 0.1, 1.0, 1.0, 0.5)

    if path is not None:
        power_consumption = sum(1 / power[x, y] * 0.05 for x, y in path)
        total_path_cost = sum(terrain[x, y] * 0.5 + thermal[x, y] * 0.05 + (1 / power[x, y]) * 0.05 for x, y in path)
        print(f"🔋 Total Power Consumption for the Path: {power_consumption:.4f} KJ")
        print(f"📊 Total Path Cost: {total_path_cost:.4f} M")
    
    visualize_path(terrain, path, "Optimized MHA Path")

# 📌 Example Run
# Replace the file path with the location of your lunar surface image (in TIFF format)
run_lunar_navigation("lunar.tif") 
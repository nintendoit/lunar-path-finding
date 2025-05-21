import rasterio
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Constants for APF
K_att = 1.0
K_rep_terrain = 100.0
K_rep_temp = 50.0
K_rep_power = 50.0
StepSize = 2
GoalMargin = 2.0
MaxIterations = 500

# 📥 Load DEM and Generate Maps
def load_maps(filename, scale_factor=0.2, max_size=500):
    print("📥 Loading NASA Terrain Data...")
    with rasterio.open(filename) as dataset:
        dem = dataset.read(1)

    if max(dem.shape) > max_size:
        scale_factor = max_size / max(dem.shape)

    dem = cv2.resize(dem, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    dem = (dem - np.min(dem)) / (np.max(dem) - np.min(dem)) * 1.0 + 0.5  # Normalize

    thermal = np.random.uniform(100, 400, dem.shape) * np.tile(np.linspace(0.7, 1, dem.shape[1]), (dem.shape[0], 1))
    power = np.random.uniform(0.3, 1.0, dem.shape) * np.tile(np.linspace(0.2, 1, dem.shape[1]), (dem.shape[0], 1))

    return dem, thermal, power

# 💡 Attractive Potential
def attractive_force(pos, goal):
    direction = goal - pos
    distance = np.linalg.norm(direction)
    return K_att * direction / (distance + 1e-6)

# 🛑 Repulsive Potentials
def repulsive_force(pos, dem, thermal, power, terrain_thresh=3.0, temp_thresh=350, power_thresh=0.4):
    x, y = int(pos[0]), int(pos[1])
    if x < 0 or x >= dem.shape[0] or y < 0 or y >= dem.shape[1]:
        return np.array([0.0, 0.0])

    gradient_x, gradient_y = np.gradient(dem)
    steepness = np.sqrt(gradient_x[x, y] ** 2 + gradient_y[x, y] ** 2)

    rep_force = np.array([0.0, 0.0])

    if steepness > terrain_thresh:
        rep_force += -K_rep_terrain * np.array([gradient_x[x, y], gradient_y[x, y]])

    if thermal[x, y] > temp_thresh:
        rep_force += -K_rep_temp * np.array([1, 1])

    if power[x, y] < power_thresh:
        rep_force += K_rep_power * np.array([1, 1])

    return rep_force

# 📍 APF Path Planning
def apf_path_planning(dem, thermal, power, start, goal):
    path = [start]
    current_pos = np.array(start)

    for _ in range(MaxIterations):
        att_force = attractive_force(current_pos, goal)
        rep_force = repulsive_force(current_pos, dem, thermal, power)
        total_force = att_force + rep_force

        new_pos = current_pos + StepSize * total_force / (np.linalg.norm(total_force) + 1e-6)
        new_pos = np.clip(new_pos, [0, 0], [dem.shape[0] - 1, dem.shape[1] - 1])

        path.append(new_pos)
        current_pos = new_pos

        if np.linalg.norm(goal - current_pos) < GoalMargin:
            return np.array(path)

    return None

# 🧭 Run a single APF instance and return path info
def run_apf(dem, thermal, power, start, goal, path_name="Path"):
    print(f"\n🚀 Running {path_name} from {start} to {goal}...")
    path = apf_path_planning(dem, thermal, power, start, goal)

    if path is not None:
        cost = 0
        power_consumption = 0
        for point in path:
            x, y = int(point[0]), int(point[1])
            cost += dem[x, y] * 1.0 + thermal[x, y] * 0.1 + (1 / power[x, y]) * 0.1
            power_consumption += (1 / power[x, y]) * 0.12

        print(f"🔋 {path_name} - Power Consumption: {power_consumption:.4f} KJ")
        print(f"📊 {path_name} - Total Path Cost: {cost:.4f} M")
    else:
        print(f"❌ {path_name} could not find a valid path.")

    return {"path": path, "start": start, "goal": goal, "name": path_name}

# 🎨 Combined Visualization of All Paths
def visualize_all_paths(image, paths_info, title="All APF Paths"):
    output = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    plt.figure(figsize=(12, 12))
    plt.imshow(output)

    colors = [(1,0,0), (0,1,0), (0,0,1)]
    for i, info in enumerate(paths_info):
        path = info["path"]
        start = info["start"]
        goal = info["goal"]
        name = info["name"]

        if path is not None:
            path = np.array(path)
            plt.plot(path[:, 1], path[:, 0], color=colors[i % len(colors)], linewidth=2, label=name)
            plt.scatter(start[1], start[0], c='red', marker='o', edgecolors='black')
            plt.scatter(goal[1], goal[0], c='green', marker='X', edgecolors='black')
        else:
            print(f"⚠ {name} could not find a path.")

    plt.title(title)
    plt.legend(loc='upper right')
    plt.axis("off")
    plt.show()

# 📌 Main Execution for Multiple Paths (Combined View)
def main():
    dem, thermal, power = load_maps("lunar.tif")

    # Define multiple paths
    paths = [
        {"start": np.array([50, 30]), "goal": np.array([dem.shape[0] - 10, dem.shape[1] - 40]), "name": "Path 1"},
        {"start": np.array([30, 70]), "goal": np.array([dem.shape[0] - 30, dem.shape[1] - 30]), "name": "Path 2"},
        {"start": np.array([20, 100]), "goal": np.array([dem.shape[0] - 80, dem.shape[1] - 35]), "name": "Path 3"},
    ]

    all_paths_info = []
    for path_data in paths:
        result = run_apf(dem, thermal, power, path_data["start"], path_data["goal"], path_data["name"])
        all_paths_info.append(result)

    visualize_all_paths(dem, all_paths_info, title="Multiple Optimized RAPF Paths")

if __name__ == "__main__":
    main()

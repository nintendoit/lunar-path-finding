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
    # dem = (dem - np.min(dem)) / (np.max(dem) - np.min(dem)) * 10  # Normalize
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

# 🎨 Visualization like A* style
def visualize_apf_path(image, path, start, goal, title="APF Path"):
    #output = cv2.cvtColor((image / np.max(image) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    output = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    plt.figure(figsize=(10, 10))
    plt.imshow(output)

    if path is not None:
        path = np.array(path)
        plt.plot(path[:, 1], path[:, 0], color='blue', linewidth=2, label=title)
        plt.scatter(start[1], start[0], c='red', marker='o', edgecolors='black', label='Start')
        plt.scatter(goal[1], goal[0], c='green', marker='o', edgecolors='black', label='Goal')
    else:
        print(f"⚠ {title} could not find a path.")

    plt.title("Optimized APF Path")
    plt.legend(loc='upper right')
    plt.axis("off")
    plt.show()

# 📌 Main Execution
def main():
    dem, thermal, power = load_maps("lunar.tif")
    start = np.array([50, 30])
    goal = np.array([dem.shape[0] - 70, dem.shape[1] - 100])

    print("🔄 Running Optimized APF Algorithm...")
    path = apf_path_planning(dem, thermal, power, start, goal)

    # Calculate path cost & power consumption
    if path is not None:
        cost = 0
        power_consumption = 0
        for point in path:
            x, y = int(point[0]), int(point[1])
            cost += dem[x, y] * 0.5 + thermal[x, y] * 0.05 + (1 / power[x, y]) * 0.05
            power_consumption += (1 / power[x, y]) * 0.05
        print(f"🔋 Total Power Consumption for the Path: {power_consumption:.4f} KJ")
        print(f"📊 Total Path Cost: {cost:.4f} M")
    else:
        print("⚠ Path could not be found using APF.")

    visualize_apf_path(dem, path, start, goal)

if __name__ == "__main__":
    main()

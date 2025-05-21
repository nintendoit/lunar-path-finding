import matplotlib.pyplot as plt
import numpy as np

# Data from image
paths = ['Path_1', 'Path_2', 'Path_3']

# A* values
astar_power = [107.9161, 88.9991, 76.0867]
astar_cost = [4885.0946, 4720.7085, 4163.4530]

# RAPF values
rapf_power = [113.3763, 108.6426, 80.4782]
rapf_cost = [5705.6404, 7115.2295, 4944.8713]

x = np.arange(len(paths))  # label locations
width = 0.35  # width of the bars

# --- Power Consumption Plot ---
fig, ax = plt.subplots(figsize=(10, 6))
bar1 = ax.bar(x - width/2, astar_power, width, label='MHA', color='steelblue')
bar2 = ax.bar(x + width/2, rapf_power, width, label='RAPF', color='orange')

ax.set_ylabel('Power Consumption (KJ)')
ax.set_title('Comparison of Power Consumption (KJ) for MHA and RAPF Across Different Paths')
ax.set_xticks(x)
ax.set_xticklabels(paths)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# --- Path Cost Plot ---
fig, ax = plt.subplots(figsize=(10, 6))
bar1 = ax.bar(x - width/2, astar_cost, width, label='MHA', color='seagreen')
bar2 = ax.bar(x + width/2, rapf_cost, width, label='RAPF', color='tomato')

ax.set_ylabel('Path Cost (m)')
ax.set_title('Comparison of Path Cost (m) for MHA and RAPF Across Different Paths')
ax.set_xticks(x)
ax.set_xticklabels(paths)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

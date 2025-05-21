import matplotlib.pyplot as plt
import numpy as np

# Data for comparison
runs = ["Terra Model", "Thermal Model", "Power Model"]
#power_consumption = [254.0599, 273.2712, 257.5718]
path_cost = [11554.9743, 12569.0824,11535.8515]  # Assuming different path costs for illustration

# Number of runs and width for bars
x = np.arange(len(runs))
new_x = [0.3*i for i in x]
bar_width = 0.2

# Plotting the grouped bar chart
plt.figure(figsize=(4, 6))
#plt.bar(new_x, power_consumption, width=bar_width, color='green', label='Power Consumption')
plt.bar(new_x, path_cost, width=bar_width, color='orange', label='Path Cost',align='center')

# Adding labels and title
plt.xlabel("Models")
plt.ylabel("Values (m)")
plt.title("Comparison of Power consumption with different Models")
plt.xticks(new_x, runs)
plt.grid(axis='y', alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()
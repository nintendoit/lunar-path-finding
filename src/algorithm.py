import matplotlib.pyplot as plt
import numpy as np

# Data for comparison
algo = ["MHA", "RAPF"]
power_consumption = [252.5759, 297.5606]
#path_cost = [11557.5612, 14632.3283]  # Assuming different path costs for illustration

# Number of runs and width for bars
x = [1,1.5] #np.arange(len(algo))
bar_width = 0.25

# Plotting the grouped bar chart
#plt.figure(figsize=(2, 6))
plt.bar(x, power_consumption, width=bar_width, color='blue', label='Power Consumption')
#plt.bar(x , path_cost, width=bar_width, color='red', label='Path Cost')

# Adding labels and title
plt.xlabel("Algorithms")
plt.ylabel("Values (KJ)")
plt.title("Comparison of Path cost with different Algorithm")
plt.xticks(x, algo)
plt.grid(axis='y', alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()
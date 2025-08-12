import matplotlib.pyplot as plt
import numpy as np

file_path = 'csp-683423.txt'

x_data = []
y_data = []

with open(file_path, 'r') as file:
    for line in file:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        x_data.append(float(parts[0]))
        y_data.append(float(parts[1]))

x_data = np.array(x_data)
y_data = np.array(y_data)

velocity = np.gradient(y_data, x_data)

acceleration = np.gradient(velocity, x_data)

plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(x_data, y_data, marker='o', linewidth=0.1)
plt.ylabel('Y Data (mm)')
plt.title('Original Data')

plt.subplot(3, 1, 2)
plt.plot(x_data, velocity, marker='o', color='orange', linewidth=0.1)
plt.ylabel('Velocity (mm/deg)')
plt.title('First Derivative (Velocity)')

plt.subplot(3, 1, 3)
plt.plot(x_data, acceleration, marker='o', color='green', linewidth=0.1)
plt.xlabel('X Data (deg)')
plt.ylabel('Acceleration (mm/deg²)')
plt.title('Second Derivative (Acceleration)')

plt.tight_layout()
plt.show()

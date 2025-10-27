%matplotlib notebook
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create grid
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

# Function that depends on g
def f(X, Y, g):
    return np.sin(np.sqrt(X**2 + Y**2) + g)

# Create interactive figure
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot multiple surfaces for g = 1...10
for g in range(1, 11):
    Z = f(X, Y, g)
    ax.plot_surface(X, Y, Z, alpha=0.5, cmap='viridis')

# Labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Surface for Different g Values')

plt.show()

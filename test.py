import numpy as np
import matplotlib.pyplot as plt

# Generate sample velocity field data
x = np.linspace(-5, 5, 20)
y = np.linspace(-5, 5, 20)
X, Y = np.meshgrid(x, y)

U = -Y  # Velocity component in x-direction
V = X   # Velocity component in y-direction

# Plot the velocity field
plt.figure(figsize=(8, 8))
plt.quiver(X, Y, U, V, color='b')

# Plot the x-axis and y-axis
plt.axhline(y=0, color='k', linestyle='--')  # x-axis
plt.axvline(x=0, color='k', linestyle='--')  # y-axis

# Set labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Velocity Field with Axes')

# Set equal scaling
plt.axis('equal')

plt.show()

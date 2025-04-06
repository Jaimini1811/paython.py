import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# === Exercise 1: Plotting lines with same slope ===
x_range = np.linspace(-10, 10, 100)

plt.figure(figsize=(8, 6))
plt.plot(x_range, 2 * x_range + 1, 'r-', label='y = 2x + 1')
plt.plot(x_range, 2 * x_range + 2, 'g--', label='y = 2x + 2')
plt.plot(x_range, 2 * x_range + 3, 'b:', label='y = 2x + 3')

plt.title('Linear Equations with Common Slope')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Exercise 2: Plotting individual data points ===
x_points = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
y_points = np.array([-0.57, -2.57, -4.80, -7.36, -8.78, -10.52, -12.85, -14.69, -16.78])

plt.figure(figsize=(6, 5))
plt.scatter(x_points, y_points, color='black', marker='+', s=100)
plt.title('Scatter Plot Example')
plt.xlabel('x values')
plt.ylabel('y values')
plt.grid(True)
plt.tight_layout()
plt.show()

# === Exercise 3: Working with height and weight data ===
# Ensure the CSV file is present in the current working directory
df = pd.read_csv("weight-height.csv")

# Extract columns and convert to metric
heights_inch = df['Height'].to_numpy()
weights_pound = df['Weight'].to_numpy()

heights_cm = heights_inch * 2.54
weights_kg = weights_pound * 0.453592

# Mean calculations
avg_height_cm = np.mean(heights_cm)
avg_weight_kg = np.mean(weights_kg)

print(f"Average Height: {avg_height_cm:.2f} cm")
print(f"Average Weight: {avg_weight_kg:.2f} kg")

# Plotting histogram for heights
plt.figure(figsize=(7, 5))
plt.hist(heights_cm, bins=20, color='skyblue', edgecolor='black')
plt.title("Height Distribution (cm)")
plt.xlabel("Height in cm")
plt.ylabel("Number of Students")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Exercise 4: Matrix inverse validation ===
matrix = np.array([
    [1, 2, 3],
    [0, 1, 4],
    [5, 6, 0]
])

# Compute inverse of matrix
inverse_matrix = np.linalg.inv(matrix)

# Identity check using dot product
result1 = np.dot(matrix, inverse_matrix)
result2 = np.dot(inverse_matrix, matrix)

print("Matrix * Inverse:")
print(result1)

print("\nInverse * Matrix:")
print(result2)

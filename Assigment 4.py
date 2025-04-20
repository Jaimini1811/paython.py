import numpy as np
import matplotlib.pyplot as plt

# Define number of dice rolls for each test
samples = [500, 1000, 2000, 5000, 10000, 15000, 20000, 50000, 100000]

for rolls in samples:
    # Generate sums of two dice using random integers between 1 and 6
    total = np.random.randint(1, 7, rolls) + np.random.randint(1, 7, rolls)

    # Count how often each total appears
    counts = [(total == i).sum() for i in range(2, 13)]

    # Convert counts to probabilities
    probabilities = np.array(counts) / rolls

    # Plot bar chart
    plt.figure(figsize=(8, 4))
    plt.bar(range(2, 13), probabilities, color='salmon', edgecolor='black')
    plt.title(f"Dice Roll Probabilities (n = {rolls})")
    plt.xlabel("Dice Sum")
    plt.ylabel("Relative Frequency")
    plt.xticks(range(2, 13))
    plt.ylim(0, 0.2)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

#Exercise 2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load data
data = pd.read_csv("weight-height.csv")

# Independent (X) and dependent (y) variables
heights = data['Height'].values.reshape(-1, 1)  # Convert to 2D array
weights = data['Weight'].values  # 1D array is fine for y

# Initialize and fit the model
reg = LinearRegression()
reg.fit(heights, weights)

# Predict weights
predicted_weights = reg.predict(heights)


plt.figure(figsize=(8, 5))
plt.scatter(heights, weights, color='orange', alpha=0.3, label='Actual Data')
plt.plot(heights, predicted_weights, color='blue', linewidth=2, label='Fitted Line')
plt.title("Height vs. Weight - Linear Regression")
plt.xlabel("Height (inches)")
plt.ylabel("Weight (pounds)")
plt.legend()
plt.grid(True)
plt.show()

rmse = np.sqrt(mean_squared_error(weights, predicted_weights))
r_squared = r2_score(weights, predicted_weights)

print("Model Evaluation:")
print(f"  ➤ RMSE: {rmse:.2f}")
print(f"  ➤ R² Score: {r_squared:.4f}")

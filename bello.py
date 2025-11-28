import pandas as pd
import matplotlib.pyplot as plt
import re

# Read the txt file and parse the data
data = []
with open('a.txt', 'r') as f:  # Replace 'your_data.txt' with your actual filename
    for line in f:
        # Extract iteration, best_makespan, and global_best using regex
        match = re.search(r'Iteration (\d+): Best Makespan = (\d+), Global Best = (\d+)', line)
        if match:
            iteration = int(match.group(1))
            best_makespan = int(match.group(2))
            global_best = int(match.group(3))
            data.append({'iteration': iteration, 'best_makespan': best_makespan, 'global_best_makespan': global_best})

# Create dataframe
df = pd.DataFrame(data)

# Create the plot
plt.figure(figsize=(12, 6))

# Plot iteration makespan
plt.plot(df['iteration'], df['best_makespan'], marker='o', label='Iteration Makespan', linewidth=2, markersize=4)

# Plot global best makespan
plt.plot(df['iteration'], df['global_best_makespan'], marker='s', label='Global Best Makespan', linewidth=2, markersize=4)

# Add C_opt baseline
c_opt = 666
plt.axhline(y=c_opt, color='r', linestyle='--', linewidth=2, label=f'C_opt Baseline ({c_opt})')

# Formatting
plt.xlabel('Iteration', fontsize=12, fontweight='bold')
plt.ylabel('Makespan', fontsize=12, fontweight='bold')
plt.title('ACO Convergence: Iteration Makespan vs Global Best', fontsize=14, fontweight='bold')
plt.legend(fontsize=10, loc='best')
plt.grid(True, alpha=0.3)

# Set x-axis ticks to show only every 100th iteration
max_iter = df['iteration'].max()
plt.xticks(range(0, max_iter + 1, 100))

plt.tight_layout()
plt.savefig('aco_convergence.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Graph saved as 'aco_convergence.png'")
print(f"\nC_opt baseline: {c_opt}")
print(f"Initial makespan: {df['best_makespan'].iloc[0]}")
print(f"Final best makespan: {df['global_best_makespan'].iloc[-1]}")
print(f"Total iterations: {len(df)}")

import matplotlib.pyplot as plt

# Hypothetical data
dataset_sizes = [5000, 10000, 15000, 20000]
patterns_support_1 = [51, 120, 111, 159]
patterns_support_2 = [126, 293, 285, 386]
patterns_support_3 = [218, 520, 519, 696]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(dataset_sizes, patterns_support_1, label='MaxGap 1', color='blue')
plt.plot(dataset_sizes, patterns_support_2, label='MaxGap 2', color='green')
plt.plot(dataset_sizes, patterns_support_3, label='MaxGap 3', color='red')

# Adding titles and labels
plt.title('Number of Patterns Found vs. Dataset Size for Different MaxGaps')
plt.xlabel('Dataset Size')
plt.ylabel('Number of Patterns Found')
plt.legend()
plt.grid(True)

# Show plot
plt.show()

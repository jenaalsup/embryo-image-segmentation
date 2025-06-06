import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the results table
file_path = 'Results2.csv'  # Replace with your actual file name
df = pd.read_csv(file_path)

# Drop any unnamed index column if it exists
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Set up the figure
plt.figure(figsize=(12, 5))

# Plot 1: Aspect Ratio vs. Area
plt.subplot(1, 2, 1)
sns.scatterplot(data=df, x='Area', y='AR', hue='AR', palette='viridis', legend=False)
plt.title('Aspect Ratio vs. Area')
plt.xlabel('Area (pixels)')
plt.ylabel('Aspect Ratio')

# Plot 2: Circularity vs. Area
plt.subplot(1, 2, 2)
sns.scatterplot(data=df, x='Area', y='Circ.', hue='Circ.', palette='plasma', legend=False)
plt.title('Circularity vs. Area')
plt.xlabel('Area (pixels)')
plt.ylabel('Circularity')

plt.tight_layout()
plt.show()

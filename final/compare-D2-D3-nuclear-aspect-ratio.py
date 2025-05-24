import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Directory containing all your CSVs
data_dir = "data/"

# Collect all files
d2_files = glob.glob(os.path.join(data_dir, "D2_slice53.csv"))
d3_files = glob.glob(os.path.join(data_dir, "D3_slice64.csv"))

def load_aspect_ratios(file_list, day_label):
    dfs = []
    for f in file_list:
        df = pd.read_csv(f)
        # Standardize column name
        col = "AR"
        df = df[[col]].copy()
        df.columns = ["Aspect Ratio"]
        df["Day"] = day_label
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# Load and label data
d2_data = load_aspect_ratios(d2_files, "Day 2")
d3_data = load_aspect_ratios(d3_files, "Day 3")

# Combine all data
all_data = pd.concat([d2_data, d3_data], ignore_index=True)

# Calculate means
d2_mean = d2_data["Aspect Ratio"].mean()
d3_mean = d3_data["Aspect Ratio"].mean()
print(f"Mean Aspect Ratio for Day 2: {d2_mean:.2f}")
print(f"Mean Aspect Ratio for Day 3: {d3_mean:.2f}")


# Plot
plt.figure(figsize=(6, 4))
all_data.boxplot(column="Aspect Ratio", by="Day", grid=False)
plt.title("Nuclear Aspect Ratio: Day 2 vs Day 3")
plt.suptitle("")  # Remove automatic "Boxplot grouped by Day" title
plt.ylabel("Aspect Ratio")
plt.tight_layout()
plt.show()

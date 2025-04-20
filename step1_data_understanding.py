import os
import pandas as pd
import matplotlib.pyplot as plt

# pastikan folder visualizations ada
os.makedirs("visualizations", exist_ok=True)

# 1. Load data
df = pd.read_csv("train.csv")

# 2. Statistik deskriptif
stats = df.describe().T
stats["median"] = df.median(numeric_only=True)
stats = stats[["count","mean","median","std","min","25%","50%","75%","max"]]
print("Descriptive statistics:\n", stats)

# 3. Simpan tabel statistik sebagai gambar
fig, ax = plt.subplots(figsize=(12,8))
ax.axis('off')
ax.table(
    cellText=stats.round(2).values,
    colLabels=stats.columns,
    rowLabels=stats.index,
    loc='center'
)
plt.savefig("visualizations/step1_descriptive_stats.png")
plt.close()

# 4. Analisis missing values
missing = df.isnull().sum()
print("\nMissing values per column:\n", missing[missing>0])

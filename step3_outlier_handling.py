import pandas as pd
import matplotlib.pyplot as plt

# 1. Load data
df = pd.read_csv("train.csv")

# 2. Boxplot numeric features
num_cols = df.select_dtypes(include="number").columns
plt.figure(figsize=(12,6))
df[num_cols].boxplot(rot=90)
plt.tight_layout()
plt.savefig("visualizations/step3_boxplots.png")
plt.close()

# 3. Remove outlier dengan IQR
Q1 = df[num_cols].quantile(0.25)
Q3 = df[num_cols].quantile(0.75)
IQR = Q3 - Q1
mask = ~((df[num_cols] < (Q1 - 1.5*IQR)) |
         (df[num_cols] > (Q3 + 1.5*IQR))).any(axis=1)
df_no_out = df[mask]
df_no_out.to_csv("df_no_out.csv", index=False)
print(f"Step 3: {len(df)-len(df_no_out)} baris outlier dihapus.")

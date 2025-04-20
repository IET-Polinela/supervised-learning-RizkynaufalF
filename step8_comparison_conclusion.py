import json
import pandas as pd

# 1. Load semua metrik
lin  = json.load(open("metrics_lin.json"))
poly = json.load(open("metrics_poly.json"))
knn  = json.load(open("metrics_knn.json"))

# 2. Gabungkan ke DataFrame
data = []
data.append(["Linear_with_outlier", lin["mse_out"], lin["r2_out"]])
data.append(["Linear_no_outlier",   lin["mse_nout"], lin["r2_nout"]])
for deg, m in poly.items():
    data.append([f"Poly_{deg}", m["mse"], m["r2"]])
for k, m in knn.items():
    data.append([f"KNN_{k}", m["mse"], m["r2"]])

df_comp = pd.DataFrame(data, columns=["Model","MSE","R2"])
df_comp.to_csv("visualizations/step8_comparison.csv", index=False)
print(df_comp)

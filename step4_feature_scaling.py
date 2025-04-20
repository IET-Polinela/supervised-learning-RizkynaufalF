import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# 1. Load data tanpa outlier
df = pd.read_csv("df_no_out.csv")

# 2. Drop kolom yang seluruhnya NaN
df = df.dropna(axis=1, how='all')

# 3. Identifikasi kolom numerik dan kategorikal
num_cols = df.select_dtypes(include="number").columns
cat_cols = df.select_dtypes(include="object").columns

# 4. Imputasi missing values
df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])
df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])

# 5. One-hot encoding, versi aman semua sklearn
version = sklearn.__version__
major, minor = map(int, version.split(".")[:2])

if major >= 1 and minor >= 2:
    ohe = OneHotEncoder(drop="first", sparse_output=False)
else:
    ohe = OneHotEncoder(drop="first", sparse=False)

df_cat_encoded = pd.DataFrame(
    ohe.fit_transform(df[cat_cols]),
    columns=ohe.get_feature_names_out(cat_cols)
)
df_num = df[num_cols].reset_index(drop=True)
df_encoded = pd.concat([df_num, df_cat_encoded], axis=1)

# 6. Scaling & histogram
X = df_encoded.drop("SalePrice", axis=1)

for name, scaler in [("standard", StandardScaler()), ("minmax", MinMaxScaler())]:
    arr = scaler.fit_transform(X)
    plt.figure(figsize=(6, 4))
    plt.hist(arr.flatten(), bins=50)
    plt.title(f"Histogram after {name.capitalize()} Scaling")
    plt.tight_layout()
    plt.savefig(f"visualizations/step4_{name}_hist.png")
    plt.close()

print("Step 4 selesai: scaling berhasil dilakukan & histogram disimpan.")

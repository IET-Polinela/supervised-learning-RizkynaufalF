import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# 1. Load data
df = pd.read_csv("train.csv")

# 2. Imputasi missing values
num_cols = df.select_dtypes(include="number").columns
cat_cols = df.select_dtypes(include="object").columns

df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])
df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])

# 3. One-hot encoding untuk kolom kategorikal
# Gunakan sparse_output=False agar kompatibel
ohe = OneHotEncoder(drop="first", sparse_output=False)
X_cat = pd.DataFrame(
    ohe.fit_transform(df[cat_cols]),
    columns=ohe.get_feature_names_out(cat_cols)
)

# 4. Gabungkan data numerik dan kategorikal
X_num = df[num_cols].reset_index(drop=True)
X_pre = pd.concat([X_num, X_cat], axis=1)
X = X_pre.drop("SalePrice", axis=1)
y = df["SalePrice"]

# 5. Split data train dan test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Simpan ke CSV
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("Step 2 selesai: file X_train, X_test, y_train, y_test berhasil disimpan.")

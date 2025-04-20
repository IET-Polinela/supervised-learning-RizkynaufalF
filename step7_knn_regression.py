import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import json

# 1. Load data tanpa outlier
df_no_out = pd.read_csv("df_no_out.csv")
X = df_no_out.drop("SalePrice", axis=1)
y = df_no_out["SalePrice"]

# 2. Menggunakan Pipeline untuk encoding, imputasi dan KNN
# Identifikasi kolom kategorikal
categorical_columns = X.select_dtypes(include=['object']).columns

# 3. Membuat Pipeline dengan OneHotEncoder, Imputer, dan KNN
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_columns),  # Menangani kategori baru
        ('num', SimpleImputer(strategy='median'), X.select_dtypes(exclude=['object']).columns)  # Mengimputasi kolom numerik
    ])

knn_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('knn', KNeighborsRegressor())
])

# 4. Train-Test Split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. K = 3,5,7
results = {}
for k in [3, 5, 7]:
    knn_pipeline.set_params(knn__n_neighbors=k)  # Mengubah nilai n_neighbors untuk setiap iterasi
    knn_pipeline.fit(X_tr, y_tr)
    y_pred = knn_pipeline.predict(X_te)
    
    mse = mean_squared_error(y_te, y_pred)
    r2  = r2_score(y_te, y_pred)
    results[f"k{k}"] = {"mse": mse, "r2": r2}
    
    # Visualisasi
    plt.figure()
    plt.scatter(y_te, y_pred, alpha=0.3)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"KNN k={k}")
    plt.savefig(f"visualizations/step7_knn{k}.png")
    plt.close()

# 6. Simpan metrik
with open("metrics_knn.json", "w") as f:
    json.dump(results, f)

print("Step 7: KNN Regression selesai.")

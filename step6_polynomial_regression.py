import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import json

# 1. Load data tanpa outlier
df_no_out = pd.read_csv("df_no_out.csv")
X = df_no_out.drop("SalePrice", axis=1)
y = df_no_out["SalePrice"]

# Pisahkan data untuk categorical columns dan numeric columns
categorical_cols = X.select_dtypes(include=['object']).columns
numeric_cols = X.select_dtypes(exclude=['object']).columns

# 2. Preprocessing pipeline untuk menghandle data kategorikal dan numerikal
# Imputasi untuk menangani NaN dan One-hot encoding untuk kolom kategorikal, serta PolynomialFeatures untuk kolom numerikal
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),  # Imputasi untuk nilai NaN
            ('poly', PolynomialFeatures(degree=2))  # Terapkan PolynomialFeatures
        ]), numeric_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Imputasi untuk nilai NaN kategori
            ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))  # One-hot encoding untuk kolom kategorikal dengan penanganan kategori baru
        ]), categorical_cols)
    ])

# 3. Train-test split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Pipeline untuk regresi linear dengan preprocessing
lr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# 5. Training model dan evaluasi dengan polynomial degree 2 & 3
results = {}
for deg in [2, 3]:
    # Terapkan PolynomialFeatures dengan degree yang berbeda
    preprocessor.transformers[0][1].steps[1] = ('poly', PolynomialFeatures(degree=deg))  # Change degree
    lr.fit(X_tr, y_tr)  # Fit model
    y_pred = lr.predict(X_te)  # Prediksi
    
    mse = mean_squared_error(y_te, y_pred)
    r2 = r2_score(y_te, y_pred)
    
    results[f"deg{deg}"] = {"mse": mse, "r2": r2}
    
    # Visualisasi hasil
    plt.figure()
    plt.scatter(y_te, y_pred, alpha=0.3)
    plt.title(f"Poly deg={deg}")
    plt.savefig(f"visualizations/step6_poly{deg}.png")
    plt.close()

# 6. Simpan metrik
with open("metrics_poly.json", "w") as f:
    json.dump(results, f)

print("Step 6: Polynomial Regression selesai.")

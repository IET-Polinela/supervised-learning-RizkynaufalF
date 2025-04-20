import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Pastikan folder visualisasi ada
os.makedirs("visualizations", exist_ok=True)

def preprocess_train_test(X_tr, X_te):
    """
    1) Drop columns with all missing values
    2) One-hot encode categorical variables
    3) Align columns between train & test
    4) Impute missing values using median strategy
    """
    # 1. Drop all-NaN columns
    X_tr = X_tr.dropna(axis=1, how='all')
    X_te = X_te.dropna(axis=1, how='all')

    # 2. One-hot encoding
    X_tr_enc = pd.get_dummies(X_tr, drop_first=True)
    X_te_enc = pd.get_dummies(X_te, drop_first=True)

    # 3. Align columns
    X_tr_enc, X_te_enc = X_tr_enc.align(X_te_enc, join='inner', axis=1)

    # 4. Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_tr_imp = pd.DataFrame(
        imputer.fit_transform(X_tr_enc),
        columns=X_tr_enc.columns,
        index=X_tr_enc.index
    )
    X_te_imp = pd.DataFrame(
        imputer.transform(X_te_enc),
        columns=X_te_enc.columns,
        index=X_te_enc.index
    )

    return X_tr_imp, X_te_imp


def train_eval(X_tr, X_te, y_tr, y_te, tag):
    """
    Train Linear Regression, plot results, and return MSE & R2.
    """
    lr = LinearRegression().fit(X_tr, y_tr.values.ravel())
    y_pred = lr.predict(X_te)

    mse = mean_squared_error(y_te, y_pred)
    r2  = r2_score(y_te, y_pred)

    # Scatter plot: Actual vs Predicted
    plt.figure()
    plt.scatter(y_te, y_pred, alpha=0.3)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"LR {tag}")
    plt.savefig(f"visualizations/step5_scatter_{tag}.png")
    plt.close()

    # Histogram: Residuals
    plt.figure()
    plt.hist(y_te.values.ravel() - y_pred, bins=30)
    plt.title(f"Residuals {tag}")
    plt.savefig(f"visualizations/step5_residual_{tag}.png")
    plt.close()

    return mse, r2

# 1. Load data
X_train = pd.read_csv("X_train.csv")
X_test  = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv")
y_test  = pd.read_csv("y_test.csv")

# 2. Preprocess and evaluate with outliers
X_tr_enc, X_te_enc = preprocess_train_test(X_train, X_test)
mse_out, r2_out = train_eval(X_tr_enc, X_te_enc, y_train, y_test, "with_outlier")

# 3. Load no-outlier dataset
df_no_out = pd.read_csv("df_no_out.csv")
X_no = df_no_out.drop("SalePrice", axis=1)
y_no = df_no_out["SalePrice"]

# Drop columns with all missing values
X_no = X_no.dropna(axis=1, how='all')

# Encode then split
X_no_enc = pd.get_dummies(X_no, drop_first=True)
X_tr_no, X_te_no, y_tr_no, y_te_no = train_test_split(
    X_no_enc, y_no, test_size=0.2, random_state=42
)

# Impute missing values
imp_no = SimpleImputer(strategy='median')
X_tr_no_imp = pd.DataFrame(
    imp_no.fit_transform(X_tr_no),
    columns=X_tr_no.columns,
    index=X_tr_no.index
)
X_te_no_imp = pd.DataFrame(
    imp_no.transform(X_te_no),
    columns=X_te_no.columns,
    index=X_te_no.index
)

# 4. Evaluate without outliers
mse_nout, r2_nout = train_eval(
    X_tr_no_imp, X_te_no_imp,
    y_tr_no.to_frame(), y_te_no.to_frame(),
    "no_outlier"
)

# 5. Save metrics
import json
with open("metrics_lin.json", "w") as f:
    json.dump({
        "mse_out": mse_out,
        "r2_out": r2_out,
        "mse_nout": mse_nout,
        "r2_nout": r2_nout
    }, f)

print("Step 5: Linear Regression selesai.")

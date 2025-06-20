# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import joblib
import pathlib
import numpy as np


DATA_PATH = pathlib.Path("train.csv")        # download from Kaggle first!

# ---------- 1. load and select columns ----------
df = pd.read_csv(DATA_PATH, usecols=[
    "SalePrice",        # target
    "GrLivArea",        # square footage above ground
    "BedroomAbvGr",     # number of bedrooms
    "FullBath"          # number of full bathrooms
]).dropna()

X = df[["GrLivArea", "BedroomAbvGr", "FullBath"]]
y = df["SalePrice"]

# ---------- 2. build pipeline ----------
numeric_features = ["GrLivArea", "BedroomAbvGr", "FullBath"]
numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

preprocess = ColumnTransformer(
    transformers=[("num", numeric_transformer, numeric_features)]
)

model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("regressor", LinearRegression())
])

# ---------- 3. train / evaluate ----------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)

print(f"Validation RMSE: {rmse:,.0f}")
print(f"Validation R² : {r2:.3f}")

# ---------- 4. persist ----------
joblib.dump(model, "model.joblib")
print("✅ Saved trained model to model.joblib")

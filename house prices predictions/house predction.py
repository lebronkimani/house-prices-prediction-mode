import kagglehub

# Download latest version
path = kagglehub.competition_download('house-prices-advanced-regression-techniques')

print("Path to competition files:", path)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
df = pd.read_csv("train.csv")

# Select features
features = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "YearBuilt"]
df = df[features + ["SalePrice"]]

# Handle missing values
df = df.fillna(df.median(numeric_only=True))

X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestRegressor(n_estimators=200)
model.fit(X_train, y_train)

# Predict
preds = model.predict(X_test)

# RMSE
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE:", rmse)
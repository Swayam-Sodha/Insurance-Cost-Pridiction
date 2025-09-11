import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import pickle
import warnings
warnings.filterwarnings("ignore")

# -------------------------------
# 1. Load Data
# -------------------------------
df = pd.read_csv("Medical Insurance cost prediction.csv")
print("Dataset Shape:", df.shape)
print(df.head())

# -------------------------------
# 2. Preprocessing
# -------------------------------
X = df.drop("charges", axis=1)
y = df["charges"]

numeric_features = ["age", "bmi", "children"]
categorical_features = ["sex", "smoker", "region"]

numeric_transformer = Pipeline([("scaler", StandardScaler())])
categorical_transformer = Pipeline([("onehot", OneHotEncoder(drop="first", sparse_output=False))])

preprocessor = ColumnTransformer(
    [("num", numeric_transformer, numeric_features),
     ("cat", categorical_transformer, categorical_features)]
)

# -------------------------------
# 3. Train/Test Split
# -------------------------------
X_prepared = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_prepared, y, test_size=0.2, random_state=42
)

# -------------------------------
# 4. Models
# -------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "SVR": SVR(),
    "KNN": KNeighborsRegressor()
}

results = {}

def evaluate_model(name, model):
    model.fit(X_train, y_train)
    y_tr_pred = model.predict(X_train)
    y_te_pred = model.predict(X_test)

    metrics = {
        "Train MAE": mean_absolute_error(y_train, y_tr_pred),
        "Test MAE": mean_absolute_error(y_test, y_te_pred),
        "Train RMSE": mean_squared_error(y_train, y_tr_pred),
        "Test RMSE": mean_squared_error(y_test, y_te_pred),
        "Train R2": r2_score(y_train, y_tr_pred),
        "Test R2": r2_score(y_test, y_te_pred),
    }
    results[name] = metrics

for name, model in models.items():
    evaluate_model(name, model)

# -------------------------------
# 5. Results Comparison
# -------------------------------
comp = pd.DataFrame(results).T
print("\nModel Comparison:")
print(comp)

# -------------------------------
# 6. Hyperparameter Tuning Example (Random Forest)
# -------------------------------
print("\nHyperparameter tuning (Random Forest)...")
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5, 10],
}
rf = RandomForestRegressor(random_state=42)
grid = GridSearchCV(rf, param_grid, cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1)
grid.fit(X_train, y_train)

print("Best RF Params:", grid.best_params_)
best_rf = grid.best_estimator_

# Evaluate tuned RF
y_pred = best_rf.predict(X_test)
print("\nTuned Random Forest Test RMSE:", mean_squared_error(y_test, y_pred))
print("Tuned Random Forest Test R2:", r2_score(y_test, y_pred))

# -------------------------------
# 7. Simple Prediction Function
# -------------------------------
def predict_charge(age, sex, bmi, children, smoker, region):
    sample = pd.DataFrame([{
        "age": age, "sex": sex, "bmi": bmi,
        "children": children, "smoker": smoker, "region": region
    }])
    sample_prepared = preprocessor.transform(sample)
    return best_rf.predict(sample_prepared)[0]

# Example
example = predict_charge(29, "female", 27.5, 0, "no", "southeast")
print(f"\nExample Prediction (29 y/o female, BMI=27.5, non-smoker, SE): ${example:,.2f}")


# Save the preprocessor and model together
with open("insurance_model.pkl", "wb") as f:
    pickle.dump((preprocessor, best_rf), f)

print("✅ Model saved as insurance_model.pkl")

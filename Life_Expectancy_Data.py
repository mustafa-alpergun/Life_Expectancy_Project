import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(r"C:\Users\muham\Downloads\archive (11)\Life Expectancy Data.csv")

print("Dataset loaded:", df.shape)

y = df["Country "]

X = df.drop("Country ", axis=1)

X = X.select_dtypes(include=["number"])

X = X.fillna(X.mean())
y = y.fillna(y.mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


rf = RandomForestRegressor(random_state=42)

parameters = {
    "n_estimators": [20, 50],
    "criterion": ["squared_error"]
}

grid_search = GridSearchCV(estimator=rf, param_grid=parameters, cv=2, verbose=1)

grid_search.fit(X_train, y_train)

print("Best model:", grid_search.best_estimator_)
print("Best CV score:", grid_search.best_score_)


best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

importances = best_model.feature_importances_
features = X.columns

indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
plt.title("Feature Importances")
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), features[indices], rotation=90)
plt.tight_layout()
plt.show()


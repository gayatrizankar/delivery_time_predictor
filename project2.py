import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle

# Load data
data = pd.read_csv("delivery_time.csv")

X = data.drop("delivery_time_min", axis=1)
y = data["delivery_time_min"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("📊 Model Performance:")
print("R2 Score:", r2)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)

# Cross validation
cv_score = np.mean(cross_val_score(model, X, y, cv=5, scoring="r2"))
print("Cross-validation R2 Score:", cv_score)

# Save model
pickle.dump(model, open("delivery_time_model.pkl", "wb"))

print("✅ Model saved as delivery_time_model.pkl")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib

# 1. Load dataset from CSV file (make sure you saved it in your working directory)
data = pd.read_csv("BostonHousing.csv")

# 2. Features and target
X = data.drop("medv", axis=1)  # all columns except target
y = data["medv"]              # target: median home value

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predict
y_pred = model.predict(X_test)

# 6. Evaluate
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"RMSE: {rmse:.3f}")

joblib.dump(model, '/mnt/datalake/instructor/diabetes_model.pkl')
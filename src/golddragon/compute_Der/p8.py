import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv("data/2020-04-15-05-12-00_2T3Y1RFV8KC014025_CAN_Messages_5_Hz.csv")
#print("Available columns:", df.columns.tolist())

data = df[["steer_angle", "yaw_rate"]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
#print("Valid rows:", len(data), "Removed rows:", len(df) - len(data))

X = data[["steer_angle"]] # two-dimensional input
y = data["yaw_rate"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Training rows:", len(X_train), "Test rows:", len(X_test))
print("\nModel Parameters:")
print(f"\nIntercept: {model.intercept_:.12f}")
print(f"\nSlope: {model.coef_[0]:.12f}")
print(f"\nTest R^2: {r2_score(y_test, y_pred):.12f}")

sse = np.sum((y_test.to_numpy() - y_pred) ** 2)
sst = np.sum((y_test.to_numpy() - y_test.mean()) ** 2)
assert np.isclose(1 - sse / sst, r2_score(y_test, y_pred))

grid = pd.DataFrame({"steer_angle": np.linspace(X["steer_angle"].min(),X["steer_angle"].max(), 200)})

plt.figure(figsize=(8, 5))
plt.scatter(X_test["steer_angle"], y_test, s=10, alpha=0.4,label="Held-out test observations")
plt.plot(grid["steer_angle"], model.predict(grid), "r-",label="Training-fitted regression line")
plt.xlabel("Steer angle")
plt.ylabel("Yaw rate")
plt.title("RAV4: steering angle predicts yaw rate")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()
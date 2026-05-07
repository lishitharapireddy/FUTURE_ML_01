import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


data = {
    'Day': [1, 2, 3, 4, 5, 6, 7],
    'Sales': [200, 220, 250, 270, 300, 320, 350]
}

df = pd.DataFrame(data)


X = df[['Day']]
y = df['Sales']


model = LinearRegression()
model.fit(X, y)


future_days = [[8], [9], [10]]
predictions = model.predict(future_days)

print("Future Sales Predictions:")
print(predictions)


plt.plot(df['Day'], df['Sales'], marker='o')
plt.xlabel("Day")
plt.ylabel("Sales")
plt.title("Sales Forecasting")
plt.show()
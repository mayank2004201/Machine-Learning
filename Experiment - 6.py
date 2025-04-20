import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
data = {
    'Date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
    'Close': [100, 102, 101, 105, 107, 110, 108, 112, 115, 117]  
}
df = pd.DataFrame(data)

df['Date'] = df['Date'].map(pd.Timestamp.toordinal)  
X = df[['Date']]
y = df['Close']

# Built-in linear regression
model = LinearRegression()
model.fit(X, y)
y_pred_builtin = model.predict(X)

print(f'Coefficient: {model.coef_[0]}')
print(f'Intercept: {model.intercept_}','\n')

# Manual computations
X = df['Date'].values.reshape(-1, 1)
y = df['Close'].values
n = len(X)

X_mean, y_mean = np.mean(X), np.mean(y)
a = np.sum((X - X_mean) * (y - y_mean)) / np.sum((X - X_mean) ** 2)                                                                                                                                                                                                                        
b = y_mean - a * X_mean

print(f'Slope (a): {a}')
print(f'Intercept (b): {b}')

y_pred_manual = a * X + b

sse = np.sum((y - y_pred_manual.flatten()) ** 2)
mse = sse / n
r_squared = 1 - (sse / np.sum((y - y_mean) ** 2))

print(f'SSE: {sse}')
print(f'MSE: {mse}')
print(f'R²: {r_squared}')

# Plotting both predictions on the same graph
plt.figure(figsize=(14, 7))
plt.scatter(df['Date'], df['Close'], color='blue', label='Actual Prices', marker='o')
plt.plot(df['Date'], y_pred_builtin, color='red', label='Predicted Prices (Built-in)', linewidth=2)           
plt.plot(df['Date'], y_pred_manual, color='green', label='Predicted Prices (Manual)', linewidth=2, linestyle='--')
plt.title('Stock Price Prediction: Built-in vs Manual Computations')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.xticks(rotation=45)
plt.show()
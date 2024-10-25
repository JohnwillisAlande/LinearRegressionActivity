import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Loading and Inspection of our provided data of Nairobi Office Prices
data = pd.read_csv("Nairobi Office Price Ex (1).csv")
print(data.head())

#Separating office sizes (our feature x) and office prices (our target y)
x = data['SIZE'].values
y = data['PRICE'].values

#Mean squared error function-Performance measure technique
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

#Gradient descent function-Learning algorithm
def gradient_descent(x, y, m, c, learning_rate, epochs):
    n = len(y)
    for epoch in range(epochs):
        y_pred = m * x + c
        dm = -(2 / n) * np.dot(x, (y - y_pred))
        dc = -(2 / n) * np.sum(y - y_pred)
        m -= learning_rate * dm
        c -= learning_rate * dc

        error = mean_squared_error(y, y_pred)
        print(f"Epoch {epoch + 1}, Error: {error}")
    return m, c

#Initializing variables for slope and intercept
m = np.random.rand()
c = np.random.rand()
learning_rate = 0.0001
epochs = 10

#Training the model
m, c = gradient_descent(x, y, m, c, learning_rate, epochs)

#Plotting the line of best fit
plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x, m * x + c, color='red', label='Line of Best Fit')
plt.xlabel('Office Size (sq. ft.)')
plt.ylabel('Office Price')
plt.legend()
plt.show()

#Making a prediction
office_size = 100
predicted_price = m * office_size + c
print(f"Predicted office price for 100 sq. ft.: {predicted_price}")
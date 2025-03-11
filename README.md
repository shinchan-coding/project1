import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Creating a sample dataset
data = {
    'Square_Feet': [1200, 1500, 1800, 2200, 2500, 2800, 3200, 3500, 3800, 4000],
    'Bedrooms': [2, 3, 3, 4, 4, 4, 5, 5, 5, 6],
    'Bathrooms': [1, 2, 2, 2, 3, 3, 3, 4, 4, 5],
    'Location_Score': [7, 8, 9, 6, 5, 7, 8, 9, 6, 7],
    'Price': [200000, 250000, 280000, 350000, 400000, 420000, 480000, 500000, 520000, 550000]
}

df = pd.DataFrame(data)

# Display the dataset
print(df.head())
# Defining features (X) and target variable (y)
X = df[['Square_Feet', 'Bedrooms', 'Bathrooms', 'Location_Score']]
y = df['Price']

# Splitting data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Creating and training the model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)
# Evaluating model performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R² Score: {r2}")
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()

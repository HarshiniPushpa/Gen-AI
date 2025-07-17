
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load data
weather = pd.read_csv('weather.csv')

# Features and target
X = weather[['Humidity', 'WindSpeed', 'Rainfall']]
y = weather['Temperature']

# Create model and train
model = LinearRegression()
model.fit(X, y)

# Example prediction: Humidity=75%, WindSpeed=12 km/h, Rainfall=3 mm
predicted_temp = model.predict([[75, 12, 3]])
print("Predicted Temperature (°C):", predicted_temp[0])

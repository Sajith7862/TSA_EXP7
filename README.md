# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 
### NAME:MOHAMED HAMEEM SAJITH J
### REG_NO:212223240090


### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM :
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

try:
    df_full = pd.read_csv('GlobalLandTemperaturesByCity.csv')
except FileNotFoundError:
    print("Error: 'GlobalLandTemperaturesByCity.csv' not found.")
    print("Please make sure the file is in the same directory as your script.")
    exit()


df_city = df_full[df_full['City'] == 'Ahmadabad'].copy()

df_city['Date'] = pd.to_datetime(df_city['dt'])

df_city = df_city.set_index('Date')

df = df_city[['AverageTemperature']]

df = df.dropna()
data_monthly = df['AverageTemperature'].resample('MS').mean()

data_monthly = data_monthly.ffill()

print("Dataset (First 5 Rows of filtered Madras data):")
print(data_monthly.head())
print("\n")

print("Running Augmented Dickey-Fuller test...")
result = adfuller(data_monthly)
print('ADF Statistic:', result[0])
print('p-value:', result[1])
print("If p-value > 0.05, the data is non-stationary (as expected here).\n")

train_size = int(len(data_monthly) * 0.8)
train_data = data_monthly[:train_size]
test_data = data_monthly[train_size:]

print(f"Training data size: {len(train_data)}")
print(f"Testing data size: {len(test_data)}\n")

model = AutoReg(train_data, lags=13)
model_fit = model.fit()

print("Model fitting complete.\n")

print("Displaying ACF and PACF plots...")

# Plot ACF
plt.figure(figsize=(12, 4))
plot_acf(data_monthly, lags=40, title='Autocorrelation Function (ACF)')
plt.grid(True)
plt.show()

# Plot PACF
plt.figure(figsize=(12, 4))
plot_pacf(data_monthly, lags=40, title='Partial Autocorrelation Function (PACF)')
plt.grid(True)
plt.show()

# 6. Make predictions using the AR model
# We predict from the start of the test set to the end
predictions = model_fit.predict(
    start=len(train_data),
    end=len(train_data) + len(test_data) - 1,
    dynamic=False
)
# Set the correct index for plotting
predictions.index = test_data.index

# 7. Calculate Mean Squared Error (MSE) and Plot
print("Calculating error and plotting results...")

mse = mean_squared_error(test_data, predictions)
rmse = np.sqrt(mse)
print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.4f}\n')

# Plot the test data and predictions
plt.figure(figsize=(12, 6))
plt.plot(train_data.iloc[-200:], label='Train (last 200 pts)') # Plot only last part of train for clarity
plt.plot(test_data, label='Test Data (Actual)', color='orange')
plt.plot(predictions, label='Predictions', color='green', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Temperature (C)')
plt.title('AR Model Predictions vs Test Data')
plt.legend()
plt.grid(True)
plt.show()
```
### OUTPUT:

### GIVEN DATA :
<img width="533" height="343" alt="image" src="https://github.com/user-attachments/assets/5301b4a7-f887-4bac-b37e-33fde62113eb" />


### PACF - ACF :
<img width="568" height="435" alt="image" src="https://github.com/user-attachments/assets/6a2844a2-bd46-4f57-9e83-e97a015e1226" />

<img width="568" height="435" alt="image" src="https://github.com/user-attachments/assets/203bc9a6-31da-4ce9-8004-e9020e51a681" />


### PREDICTION :
<img width="340" height="62" alt="image" src="https://github.com/user-attachments/assets/0ab97054-cc6a-4f9e-8133-a2d0f6e18f46" />


### FINIAL PREDICTION :
<img width="1010" height="547" alt="image" src="https://github.com/user-attachments/assets/f683027d-f314-4807-87bb-1a7d10daea52" />

### RESULT:
Thus we have successfully implemented the auto regression function using python.

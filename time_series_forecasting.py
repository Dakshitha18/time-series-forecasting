# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Step 1: Load and Preview Data
# Sample dataset: Replace with your actual dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
df = pd.read_csv(url)

# Convert 'Month' column to datetime format
df['Month'] = pd.to_datetime(df['Month'], format='%Y-%m')
df.set_index('Month', inplace=True)

# Check the first few rows of the data
print(df.head())

# Step 2: Visualizing the Time Series Data
plt.figure(figsize=(10, 6))
plt.plot(df)
plt.title('Airline Passengers Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.show()

# Step 3: Data Cleaning (if necessary)
# Check for missing values
print(df.isnull().sum())

# Step 4: Exploratory Data Analysis (EDA)
# Rolling mean and standard deviation
rolling_mean = df['Passengers'].rolling(window=12).mean()
rolling_std = df['Passengers'].rolling(window=12).std()

plt.figure(figsize=(10, 6))
plt.plot(df, label='Original Data')
plt.plot(rolling_mean, color='red', label='Rolling Mean')
plt.plot(rolling_std, color='blue', label='Rolling Std')
plt.legend()
plt.title('Rolling Mean and Std of Passengers Data')
plt.show()

# Step 5: Stationarity Test (ADF Test)
result = adfuller(df['Passengers'])
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')

if result[1] < 0.05:
    print("The series is stationary")
else:
    print("The series is not stationary. Differencing is needed.")

# Step 6: Differencing to Make Data Stationary
df_diff = df['Passengers'].diff().dropna()

# Visualizing Differenced Data
plt.figure(figsize=(10, 6))
plt.plot(df_diff)
plt.title('Differenced Data')
plt.show()

# Step 7: ACF and PACF Plots
# Autocorrelation (ACF) and Partial Autocorrelation (PACF) plots to determine p and q values for ARIMA
plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(df_diff, lags=40, ax=plt.gca())
plt.subplot(122)
plot_pacf(df_diff, lags=40, ax=plt.gca())
plt.show()

# Step 8: Build ARIMA Model
# Use the ARIMA model with p=1, d=1, q=1 (example, tune accordingly based on ACF/PACF)
model = ARIMA(df['Passengers'], order=(1,1,1))
model_fit = model.fit()

# Step 9: Model Summary
print(model_fit.summary())

# Step 10: Make Predictions
# Predicting the next 12 months (for example)
forecast_steps = 12
forecast = model_fit.forecast(steps=forecast_steps)

# Visualizing the Forecast
plt.figure(figsize=(10, 6))
plt.plot(df, label='Historical Data')
plt.plot(pd.date_range(df.index[-1], periods=forecast_steps, freq='M'), forecast, color='red', label='Forecast')
plt.legend()
plt.title('Sales Forecast for Next 12 Months')
plt.show()

# Step 11: Model Evaluation (Train-Test Split)
train_size = int(len(df) * 0.8)
train, test = df['Passengers'][:train_size], df['Passengers'][train_size:]

# Build the model using the training set
model_train = ARIMA(train, order=(1,1,1))
model_train_fit = model_train.fit()

# Forecasting on the test set
forecast_test = model_train_fit.forecast(steps=len(test))

# Evaluate the model
mae = mean_absolute_error(test, forecast_test)
mse = mean_squared_error(test, forecast_test)
rmse = np.sqrt(mse)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')

# Visualizing the test set prediction vs actual
plt.figure(figsize=(10, 6))
plt.plot(test, label='Actual Data')
plt.plot(test.index, forecast_test, color='red', label='Predicted Data')
plt.legend()
plt.title('ARIMA Forecast vs Actual (Test Set)')
plt.show()

# Step 12: Conclusion
print("The model has been successfully trained and evaluated. You can now use the model to forecast future sales.")

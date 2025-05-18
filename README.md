# Ex.No-6-HOLT-WINTERS-METHOD

## AIM:
To implement the Holt Winters Method Model using Python.

## ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as datetime, set it as index, and perform some initial data exploration
3. Resample it to a monthly frequency beginning of the month
4. You plot the time series data, and determine whether it has additive/multiplicative trend/seasonality
5. Split test,train data,create a model using Holt-Winters method, train with train data and Evaluate the model predictions against test data
6. Create teh final model and predict future data and plot it

## PROGRAM:

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose

# Load dataset
df = pd.read_csv('/content/web_traffic.csv', parse_dates=['Timestamp'], dayfirst=True)

# Set timestamp as index and sort
df.set_index('Timestamp', inplace=True)
df.sort_index(inplace=True)

# Resample to hourly frequency (sum of 2 half-hour periods)
data_hourly = df['TrafficCount'].resample('h').sum()
data_hourly = data_hourly.asfreq('h')  # Ensure consistent frequency
data_hourly = data_hourly.fillna(method='ffill')  # Fill missing values

# Plot original hourly data
data_hourly.plot(title="Hourly Traffic Volume")
plt.ylabel("Vehicle Count")
plt.show()

# Scale data
scaler = MinMaxScaler()
scaled_data = pd.Series(
    scaler.fit_transform(data_hourly.values.reshape(-1, 1)).flatten(),
    index=data_hourly.index
)
scaled_data.plot(title="Scaled Traffic Data")
plt.ylabel("Scaled Count")
plt.show()

# Seasonal Decomposition (unscaled data)
decomposition = seasonal_decompose(data_hourly, model="additive", period=12)  # 12 = 6 hours (2 per hour)
decomposition.plot()
plt.show()

# Train-test split (80% train, 20% test)
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]

# Fit Holt-Winters model on training data
model = ExponentialSmoothing(train_data, trend='add', seasonal='add', seasonal_periods=12).fit()

# Forecast test period
test_pred = model.forecast(steps=len(test_data))

# Plot predictions vs actual
ax = train_data.plot(label="Train")
test_data.plot(ax=ax, label="Test")
test_pred.plot(ax=ax, label="Predicted")
plt.legend()
plt.title("Traffic Volume Forecast - Visual Evaluation")
plt.show()

# Evaluation
rmse = np.sqrt(mean_squared_error(test_data, test_pred))
print(f"RMSE on test set: {rmse:.4f}")

# Ensure positive values for multiplicative model
if (data_hourly <= 0).any():
    data_hourly_positive = data_hourly + 1e-6
else:
    data_hourly_positive = data_hourly

# Fit final model on full dataset (multiplicative seasonality)
final_model = ExponentialSmoothing(data_hourly_positive, trend='add', seasonal='mul', seasonal_periods=12).fit()

# Forecast next 120 hours
forecast_steps = 120
final_forecast = final_model.forecast(steps=forecast_steps)

# Plot final forecast
ax = data_hourly.plot(label="Observed",figsize=(10,6))
final_forecast.plot(ax=ax, label="Forecast")
plt.title("Final Forecast (Next 6 Hours)")
plt.ylabel("Traffic Count")
plt.xlabel("Time")
plt.legend()
plt.show()
```

## OUTPUT

### Scaled data plot:
![image](https://github.com/user-attachments/assets/8927f9c1-aa73-4b55-b86c-13ac600256dd)

### Decomposed plot:
![image](https://github.com/user-attachments/assets/cb4c1bb8-54cd-498e-b0a0-d5bb60f314b9)

### Test prediction:
![image](https://github.com/user-attachments/assets/9e263f37-a81f-4a5f-b6cf-5edc92b78c79)

### Model performance metrics:
![Screenshot (107)](https://github.com/user-attachments/assets/d512ad6f-3e02-4fe6-bc8b-2751e1e91d49)

### Final prediction:
![image](https://github.com/user-attachments/assets/aa1997e5-38e4-4145-a15b-6bab6e50b643)


## RESULT:
### Thus the program run successfully based on the Holt Winters Method model.

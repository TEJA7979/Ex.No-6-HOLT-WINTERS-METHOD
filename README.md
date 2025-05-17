# Ex.No-6-HOLT-WINTERS-METHOD

## AIM:
To implement the Holt Winters Method Model using Python.

## ALGORITHM:
1, You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as datetime, set it as index, and perform some initial data exploration
3. Resample it to a monthly frequency beginning of the month
4. You plot the time series data, and determine whether it has additive/multiplicative trend/seasonality
5. Split test,train data,create a model using Holt-Winters method, train with train data and Evaluate the model predictions against test data
6. Create teh final model and predict future data and plot it

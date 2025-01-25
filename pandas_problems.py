'''
Problem 1: Time Series Anomaly Detection & Forecasting with Pandas & Numpy
You are given a dataset that represents hourly sales data for a retail store over the past year. The dataset has 3 columns:

timestamp (datetime object)
sales (numeric)
store_id (categorical representing multiple stores)
Objective:

Use pandas and numpy to clean and preprocess the data, ensuring that there are no missing timestamps or sales data.
Implement a method to detect anomalies in sales patterns. Anomalies might be days with significantly higher or lower sales than expected. Use techniques like moving averages or Z-score to detect anomalies.
Forecast the next month's sales using statistical techniques or machine learning methods (e.g., ARIMA, exponential smoothing, or neural networks).
Compare the forecasting model's performance using appropriate evaluation metrics (e.g., MAE, RMSE).
Constraints:

Ensure that your anomaly detection is scalable for a large dataset (several million rows).
The model must handle time series seasonality and holidays.
'''


import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
from scipy.stats import zscore
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report 
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np

# Set parameters for the synthetic time series dataset
n_hours = 365 * 24  # 1 year of hourly data
n_stores = 5  # Multiple stores

# Generate hourly timestamps for a year
timestamps = pd.date_range(start="2024-01-01", periods=n_hours, freq="H")

# Random sales data (add some noise for variability)
np.random.seed(42)
sales = np.random.normal(loc=100, scale=30, size=n_hours)  # Average sales around 100, with some variance

# Adding anomalies in sales for anomaly detection (spikes in sales)
anomalies = np.random.choice(range(n_hours), size=50, replace=False)  # Random anomaly indices
sales[anomalies] *= 3  # Triple sales on anomalies

# Assign store_id to the sales data (randomly assigning stores to data points)
store_ids = np.random.choice([f"Store_{i+1}" for i in range(n_stores)], size=n_hours)

# Create DataFrame
df = pd.DataFrame({
    "timestamp": timestamps,
    "sales": sales,
    "store_id": store_ids
})
def fill_missing_timestamps(store_df):
    full_range = pd.date_range(start=store_df['timestamp'].min(),
                               end=store_df['timestamp'].max(),
                               freq='H')
    store_df = store_df.set_index('timestamp')
    store_df = store_df.reindex(full_range, fill_value=np.nan)
    store_df.index.name = 'timestamp'
    return store_df.reset_index()

# Fill missing timestamps for all stores
df = df.groupby('store_id').apply(fill_missing_timestamps).reset_index(drop=True)

# Fill missing sales data with forward fill (or any other imputation method)
df['sales'] = df['sales'].fillna(method='ffill')

# Check for missing values after filling
print(df.isnull().sum())

# Calculate a 24-hour moving average for sales
df['moving_avg_24h'] = df.groupby('store_id')['sales'].transform(
    lambda x: x.rolling(window=24, min_periods=1).mean())

# Calculate Z-score for sales
df['z_score'] = df.groupby('store_id')['sales'].transform(lambda x: zscore(x, nan_policy='omit'))

# Define anomaly threshold (e.g., |Z| > 3)
anomaly_threshold = 3
df['is_anomaly'] = (df['z_score'].abs() > anomaly_threshold)

# Visualize anomalies for a sample store
store_sample = df[df['store_id'] == df['store_id'].unique()[0]]
plt.figure(figsize=(15, 6))
plt.plot(store_sample['timestamp'], store_sample['sales'], label='Sales')
plt.plot(store_sample['timestamp'], store_sample['moving_avg_24h'], label='Moving Average')
plt.scatter(store_sample['timestamp'][store_sample['is_anomaly']],
            store_sample['sales'][store_sample['is_anomaly']],
            color='red', label='Anomaly', zorder=5)
plt.legend()
plt.title('Anomaly Detection')
plt.show()

from statsmodels.tsa.arima.model import ARIMA 
forcasting_data  = df.set_index('timestamp')['sales'][:20]
traning_data_size = int( len(forcasting_data) * 0.7)
traning_data , testing_data = forcasting_data[:traning_data_size], forcasting_data[traning_data_size:]

models = ARIMA(traning_data.values,order= (1,1,1)).fit() 
print(models.summary())

# Assuming forecasting_data and forecasting_test are pandas Series or numpy arrays
forcasting_test = models.forecast(steps= 30 )
print(traning_data)
plt.figure(figsize=(10, 6))

# Plot the actual data (forecasting_data)
plt.plot(forcasting_data, label='Actual Data', color='blue')

# Plot the forecasted data (forecasting_test)
plt.plot(forcasting_test, linestyle='--', color='red', label='Forecasted Data')

# Add labels, title, and legend
plt.title('Forecasting Data vs Test Data', fontsize=16)
plt.xlabel('Timestamp', fontsize=12)
plt.ylabel('Sales', fontsize=12)
plt.legend(fontsize=12)

# Show the grid for better visualization
plt.grid(True, linestyle='--', alpha=0.7)

# Display the plot
plt.show()

test_result = models.forecast(len(testing_data))

print(f"mean absolute error : {mean_absolute_error(testing_data,test_result)}")
print(f"mean squared error : {np.sqrt(mean_squared_error(testing_data,test_result))}")

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA


# Function to load and preprocess data
def load_and_preprocess_data(file_path='https://raw.githubusercontent.com/kaushik051099/Projects/Time-Series-Analysis/Normal_trafficdata.csv'):
    # Read the data from the provided file path (either URL or local file)
    df = pd.read_csv(file_path)
    
    # Convert the 'Time' column to datetime format
    df['Time'] = pd.to_datetime(df['Time'], unit='s')  # Ensure correct conversion of time
    df.set_index('Time', inplace=True)  # Set 'Time' as the index column for time series analysis
    
    # Drop rows with missing values
    df.dropna(inplace=True)
    
    # Ensure 'Length' column is treated as a float type
    df['Length'] = df['Length'].astype(float)
    
    return df


# Function to visualize additional network traffic details
def visualize_traffic_details(df):
    # Traffic volume per second
    traffic_volume = df['Length'].resample('1S').sum()
    plt.figure(figsize=(10, 6))
    traffic_volume.plot(title="Traffic Volume Over Time (per second)", xlabel="Time", ylabel="Total Length (bytes)")
    plt.show()

    # Protocol distribution
    if 'Protocol' in df.columns:
        protocol_counts = df['Protocol'].value_counts()
        plt.figure(figsize=(10, 6))
        protocol_counts.plot(kind='bar', title="Protocol Distribution", xlabel="Protocol", ylabel="Frequency")
        plt.xticks(rotation=45)
        plt.show()

    # Top 10 source addresses
    if 'Source' in df.columns:
        top_sources = df['Source'].value_counts().head(10)
        plt.figure(figsize=(10, 6))
        top_sources.plot(kind='bar', title="Top 10 Source Addresses", xlabel="Source", ylabel="Frequency")
        plt.xticks(rotation=45)
        plt.show()

    # Top 10 destination addresses
    if 'Destination' in df.columns:
        top_destinations = df['Destination'].value_counts().head(10)
        plt.figure(figsize=(10, 6))
        top_destinations.plot(kind='bar', title="Top 10 Destination Addresses", xlabel="Destination", ylabel="Frequency")
        plt.xticks(rotation=45)
        plt.show()


# Function to resample traffic data to minute-level granularity and visualize it
def resample_and_visualize(df):
    # Resample the data by minute ('T') and aggregate packet lengths
    df_resampled = df.resample('T').agg({'Length': 'sum'})
    
    # Plot traffic volume over time
    plt.figure(figsize=(10, 6))
    plt.plot(df_resampled.index, df_resampled['Length'], label='Traffic Volume (Packet Size)')
    plt.title('Network Traffic Volume Over Time')
    plt.xlabel('Time')
    plt.ylabel('Packet Size (bytes)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()
    
    return df_resampled


# Function for time series forecasting with ARIMA
def forecast_with_arima(df_resampled):
    # Fit ARIMA model (Order (1,1,0) - first order differencing and AR(1) process)
    model = ARIMA(df_resampled['Length'], order=(1, 1, 0))
    model_fit = model.fit()
    
    # Forecast the next 100 time steps
    forecast_steps = 100
    forecast = model_fit.forecast(steps=forecast_steps)
    
    # Plot actual traffic vs forecasted traffic
    plt.figure(figsize=(10, 6))
    plt.plot(df_resampled.index, df_resampled['Length'], label='Actual Traffic')
    plt.plot(pd.date_range(df_resampled.index[-1], periods=forecast_steps, freq='T'), forecast, label='Forecasted Traffic', color='red')
    plt.title('ARIMA Model Forecast for Network Traffic')
    plt.xlabel('Time')
    plt.ylabel('Packet Size (bytes)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()


# Function for anomaly detection based on residuals
def detect_anomalies(df_resampled, model_fit):
    # Calculate residuals (actual - predicted)
    residuals = df_resampled['Length'] - model_fit.fittedvalues
    
    # Define anomaly threshold (1 standard deviation)
    threshold = 1 * np.std(residuals)
    
    # Flag anomalies
    anomalies = residuals[residuals.abs() > threshold]
    
    # Plot residuals and anomalies
    plt.figure(figsize=(10, 6))
    plt.plot(df_resampled.index, residuals, label='Residuals')
    plt.axhline(y=threshold, color='red', linestyle='--', label='Upper Threshold')
    plt.axhline(y=-threshold, color='red', linestyle='--', label='Lower Threshold')
    plt.title('Residuals and Anomalies in Network Traffic')
    plt.xlabel('Time')
    plt.ylabel('Residuals')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()
    
    # Print anomalies
    print("Anomalies Detected at these time points:")
    print(anomalies)


# Main function to execute the process
if __name__ == '__main__':
    # Load and preprocess data
    df = load_and_preprocess_data()  # Using default GitHub URL
    
    # Visualize additional traffic details
    visualize_traffic_details(df)
    
    # Resample and visualize traffic data
    df_resampled = resample_and_visualize(df)
    
    # Fit the ARIMA model and forecast
    model_fit = ARIMA(df_resampled['Length'], order=(1, 1, 0)).fit()
    forecast_with_arima(df_resampled)
    
    # Detect anomalies
    detect_anomalies(df_resampled, model_fit)

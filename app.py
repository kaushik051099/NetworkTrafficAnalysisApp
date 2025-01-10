import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Function to load and preprocess data
def load_and_preprocess_data(file_path='https://github.com/kaushik051099/NetworkTrafficAnalysisApp/blob/main/Normal_trafficdata.csv'):
    df = pd.read_csv(file_path)
    df['Time'] = pd.to_datetime(df['Time'], unit='s')  # Ensure correct conversion of time
    df.set_index('Time', inplace=True)
    df.dropna(inplace=True)
    df['Length'] = df['Length'].astype(float)
    return df

# Streamlit app layout and functionality
st.title("Network Traffic Analysis and Forecasting")

# Sidebar for file input or default dataset
st.sidebar.header("Data Input")
file_option = st.sidebar.radio("Choose Data Source:", ["Default Dataset", "Upload CSV File"])

if file_option == "Upload CSV File":
    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file:
        df = load_and_preprocess_data(uploaded_file)
    else:
        st.warning("Please upload a CSV file.")
        st.stop()
else:
    st.write("Using default dataset.")
    df = load_and_preprocess_data()

# Display the first few rows of data
st.write("### Data Preview")
st.write(df.head())

# Visualize traffic details
st.write("### Traffic Details")
if st.checkbox("Show Traffic Volume Over Time"):
    traffic_volume = df['Length'].resample('1S').sum()
    plt.figure(figsize=(10, 6))
    traffic_volume.plot(title="Traffic Volume Over Time (per second)", xlabel="Time", ylabel="Total Length (bytes)")
    st.pyplot(plt)

if 'Protocol' in df.columns and st.checkbox("Show Protocol Distribution"):
    protocol_counts = df['Protocol'].value_counts()
    plt.figure(figsize=(10, 6))
    protocol_counts.plot(kind='bar', title="Protocol Distribution", xlabel="Protocol", ylabel="Frequency")
    plt.xticks(rotation=45)
    st.pyplot(plt)

if 'Source' in df.columns and st.checkbox("Show Top 10 Source Addresses"):
    top_sources = df['Source'].value_counts().head(10)
    plt.figure(figsize=(10, 6))
    top_sources.plot(kind='bar', title="Top 10 Source Addresses", xlabel="Source", ylabel="Frequency")
    plt.xticks(rotation=45)
    st.pyplot(plt)

if 'Destination' in df.columns and st.checkbox("Show Top 10 Destination Addresses"):
    top_destinations = df['Destination'].value_counts().head(10)
    plt.figure(figsize=(10, 6))
    top_destinations.plot(kind='bar', title="Top 10 Destination Addresses", xlabel="Destination", ylabel="Frequency")
    plt.xticks(rotation=45)
    st.pyplot(plt)

# Resample traffic data and visualize
st.write("### Resampled Traffic Data")
df_resampled = df.resample('T').agg({'Length': 'sum'})
st.write(df_resampled.head())

plt.figure(figsize=(10, 6))
plt.plot(df_resampled.index, df_resampled['Length'], label='Traffic Volume (Packet Size)')
plt.title('Network Traffic Volume Over Time')
plt.xlabel('Time')
plt.ylabel('Packet Size (bytes)')
plt.xticks(rotation=45)
plt.legend()
st.pyplot(plt)

# ARIMA Forecasting
st.write("### Time Series Forecasting with ARIMA")
model_fit = ARIMA(df_resampled['Length'], order=(1, 1, 0)).fit()
forecast_steps = 100
forecast = model_fit.forecast(steps=forecast_steps)

plt.figure(figsize=(10, 6))
plt.plot(df_resampled.index, df_resampled['Length'], label='Actual Traffic')
plt.plot(pd.date_range(df_resampled.index[-1], periods=forecast_steps, freq='T'), forecast, label='Forecasted Traffic', color='red')
plt.title('ARIMA Model Forecast for Network Traffic')
plt.xlabel('Time')
plt.ylabel('Packet Size (bytes)')
plt.xticks(rotation=45)
plt.legend()
st.pyplot(plt)

# Anomaly Detection
st.write("### Anomaly Detection")
residuals = df_resampled['Length'] - model_fit.fittedvalues
threshold = 1 * np.std(residuals)
anomalies = residuals[residuals.abs() > threshold]

plt.figure(figsize=(10, 6))
plt.plot(df_resampled.index, residuals, label='Residuals')
plt.axhline(y=threshold, color='red', linestyle='--', label='Upper Threshold')
plt.axhline(y=-threshold, color='red', linestyle='--', label='Lower Threshold')
plt.title('Residuals and Anomalies in Network Traffic')
plt.xlabel('Time')
plt.ylabel('Residuals')
plt.xticks(rotation=45)
plt.legend()
st.pyplot(plt)

st.write("Anomalies Detected at these time points:")
st.write(anomalies)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from urllib.error import URLError
import requests
from io import StringIO  # Added this import

def load_data(file_path=None):
    """Load and preprocess network traffic data."""
    try:
        if file_path is None:
            # Default dataset URL
            file_path = 'https://raw.githubusercontent.com/kaushik051099/NetworkTrafficAnalysisApp/08d447e039be62e7e459ad0c56469ccf1ba6e25e/Normal_trafficdata_new.csv'
            # For URLs, download content first
            response = requests.get(file_path)
            response.raise_for_status()  # Raise an exception for bad status codes
            content = response.content.decode('utf-8', errors='replace')
            df = pd.read_csv(StringIO(content))
        else:
            # Try different encodings for uploaded files
            encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    if encoding == encodings[-1]:  # If this was the last encoding to try
                        st.error(f"Unable to read the file. Please ensure it's a valid CSV file.")
                        st.stop()
                    continue

        # Validate required columns
        required_columns = ['Time', 'Length']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            st.stop()

        # Convert Time column
        try:
            df['Time'] = pd.to_datetime(df['Time'], unit='s')
            df.set_index('Time', inplace=True)
        except ValueError as e:
            st.error("Error processing time column. Please ensure it contains valid timestamps.")
            st.stop()

        # Clean and convert Length column
        df['Length'] = pd.to_numeric(df['Length'], errors='coerce')
        df.dropna(inplace=True)

        if df.empty:
            st.error("No valid data remained after preprocessing.")
            st.stop()

        return df

    except requests.RequestException as e:
        st.error(f"Error downloading the default dataset: {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the data: {str(e)}")
        st.stop()

def plot_traffic_volume(data):
    """Plot traffic volume over time."""
    fig, ax = plt.subplots(figsize=(10, 6))
    traffic_volume = data['Length'].resample('1S').sum()
    traffic_volume.plot(ax=ax)
    ax.set_title("Traffic Volume Over Time (per second)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Total Length (bytes)")
    return fig

def plot_protocol_distribution(data):
    """Plot protocol distribution if available."""
    if 'Protocol' in data.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        data['Protocol'].value_counts().plot(kind='bar', ax=ax)
        ax.set_title("Protocol Distribution")
        ax.set_xlabel("Protocol")
        ax.set_ylabel("Frequency")
        plt.xticks(rotation=45)
        return fig
    return None

def plot_address_distribution(data, column, title):
    """Plot top 10 addresses for source/destination."""
    if column in data.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        data[column].value_counts().head(10).plot(kind='bar', ax=ax)
        ax.set_title(f"Top 10 {title}")
        ax.set_xlabel(title)
        ax.set_ylabel("Frequency")
        plt.xticks(rotation=45)
        return fig
    return None

def create_arima_forecast(data, p, d, q, forecast_steps=100):
    """Create ARIMA forecast."""
    try:
        model_fit = ARIMA(data['Length'], order=(p, d, q)).fit()
        forecast = model_fit.forecast(steps=forecast_steps)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data.index, data['Length'], label='Actual Traffic')
        ax.plot(pd.date_range(data.index[-1], periods=forecast_steps, freq='T'),
                forecast, label='Forecasted Traffic', color='red')
        ax.set_title('ARIMA Model Forecast for Network Traffic')
        ax.set_xlabel('Time')
        ax.set_ylabel('Packet Size (bytes)')
        plt.xticks(rotation=45)
        plt.legend()
        return fig, model_fit
    except Exception as e:
        st.error(f"Error in ARIMA forecasting: {str(e)}")
        st.stop()

def detect_anomalies(data, model_fit, threshold_factor):
    """Detect and plot anomalies."""
    try:
        residuals = data['Length'] - model_fit.fittedvalues
        threshold = threshold_factor * np.std(residuals)
        anomalies = residuals[residuals.abs() > threshold]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data.index, residuals, label='Residuals')
        ax.scatter(anomalies.index, anomalies, color='red', label='Anomalies')
        ax.axhline(y=threshold, color='red', linestyle='--', label='Upper Threshold')
        ax.axhline(y=-threshold, color='red', linestyle='--', label='Lower Threshold')
        ax.set_title('Residuals and Anomalies in Network Traffic')
        ax.set_xlabel('Time')
        ax.set_ylabel('Residuals')
        plt.xticks(rotation=45)
        plt.legend()
        return fig, anomalies
    except Exception as e:
        st.error(f"Error in anomaly detection: {str(e)}")
        st.stop()

def main():
    st.title("Network Traffic Analysis and Forecasting")
    
    # Add a cache cleanup button in the sidebar
    st.sidebar.header("Settings")
    if st.sidebar.button("Clear Cache"):
        st.cache_data.clear()
        st.experimental_rerun()
    
    # Sidebar for data input
    st.sidebar.header("Data Input")
    file_option = st.sidebar.radio("Choose Data Source:", ["Default Dataset", "Upload CSV File"])
    
    # Load data
    try:
        if file_option == "Upload CSV File":
            uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
            if uploaded_file is None:
                st.warning("Please upload a CSV file.")
                return
            df = load_data(uploaded_file)
        else:
            with st.spinner('Loading default dataset...'):
                df = load_data()
        
        # Data preview
        st.write("### Data Preview")
        st.write(df.head())
        
        # Traffic analysis section
        st.write("### Traffic Analysis")
        
        if st.checkbox("Show Traffic Volume Over Time"):
            st.pyplot(plot_traffic_volume(df))
        
        col1, col2 = st.columns(2)
        with col1:
            if st.checkbox("Show Protocol Distribution"):
                fig = plot_protocol_distribution(df)
                if fig:
                    st.pyplot(fig)
                else:
                    st.info("No protocol data available in the dataset")
        
        with col2:
            if st.checkbox("Show Address Distribution"):
                source_fig = plot_address_distribution(df, 'Source', 'Source Addresses')
                if source_fig:
                    st.pyplot(source_fig)
                else:
                    st.info("No source address data available")
                    
                dest_fig = plot_address_distribution(df, 'Destination', 'Destination Addresses')
                if dest_fig:
                    st.pyplot(dest_fig)
                else:
                    st.info("No destination address data available")
        
        # Time series analysis section
        st.write("### Time Series Analysis")
        df_resampled = df.resample('T').agg({'Length': 'sum'})
        
        # ARIMA configuration
        st.write("#### ARIMA Model Configuration")
        col1, col2, col3 = st.columns(3)
        with col1:
            p = st.slider("Order (p)", 0, 5, 1)
        with col2:
            d = st.slider("Order (d)", 0, 3, 1)
        with col3:
            q = st.slider("Order (q)", 0, 5, 0)
        
        with st.spinner('Fitting ARIMA model...'):
            forecast_fig, model_fit = create_arima_forecast(df_resampled, p, d, q)
            st.pyplot(forecast_fig)
        
        # Anomaly detection
        st.write("### Anomaly Detection")
        threshold_factor = st.slider("Anomaly Detection Sensitivity", 0.5, 3.0, 1.0)
        anomaly_fig, anomalies = detect_anomalies(df_resampled, model_fit, threshold_factor)
        st.pyplot(anomaly_fig)
        
        if not anomalies.empty:
            st.write("#### Detected Anomalies:")
            st.write(anomalies)
    
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()

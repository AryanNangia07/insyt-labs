# streamlit_app.py

import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os
from datetime import timedelta
import holidays

st.title("Sales Forecast by SKU")

# Set up folder containing SKU Excel files
DATA_FOLDER = "data"  # <-- Make sure your Excel files are inside a "data" folder in your working directory

# List available SKU files
sku_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".xlsx")]

# SKU selector
selected_sku_file = st.selectbox(
    "Select SKU to Forecast",
    sku_files,
    format_func=lambda x: x.replace('.xlsx', '')
)
sku_path = os.path.join(DATA_FOLDER, selected_sku_file)

# Load data
df = pd.read_excel(sku_path)
df.columns = df.columns.str.replace(' ', '_').str.lower()
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Dictionary of SKUs and their MRP
sku_mrp = {
    'FBT': 4499,
    'Venice': 1495,
    'Veleno': 1995,
    'Razor': 149
}

# Create holiday dataframe
ind_holidays = holidays.India(years=[2023, 2024, 2025])
holiday_df = pd.DataFrame([
    {'ds': pd.to_datetime(date), 'holiday': 'indian_holiday'}
    for date in ind_holidays.keys()
])

# Prepare data for Prophet
df_copy = df.copy()
df_copy = df_copy.rename(columns={'date': 'ds', 'sales': 'y'})


# Forecast UI
months = st.selectbox("Select forecast horizon (months):", options=[1, 2, 3], index=0)
monthly_marketing = st.slider("Monthly Marketing Spend", min_value=0, max_value=10000000, value=100000, step=100000)
selling_price = st.slider("Selling Price (SP)", min_value=0, max_value=4499, value=3499, step=50)

# Calculate regressors
forecast_days = months * 30
daily_marketing = monthly_marketing / 30

# Get the MRP for the selected SKU
selected_sku = selected_sku_file.replace('.xlsx', '')
selected_mrp = sku_mrp[selected_sku]

# Calculate discount based on selected SKU
discount = selected_mrp - selling_price

# Filter and clean training data
cutoff_date = pd.Timestamp('2025-06-30')
df_copy['discount'].fillna(0, inplace=True)
df_copy['marketing'].fillna(0, inplace=True)
df_train = df_copy[df_copy['ds'] <= cutoff_date].copy()

# Initialize Prophet model with holidays
model = Prophet(
    growth='linear',
    daily_seasonality=True,
    weekly_seasonality=True,
    changepoint_prior_scale=0.3,
    holidays=holiday_df
)

model.add_seasonality(
    name='monthly',
    period=30.5,
    fourier_order=5
)

model.add_regressor('discount')
model.add_regressor('marketing')

# Fit model
model.fit(df_train)

# Build future dataframe
future = model.make_future_dataframe(periods=forecast_days)

# Merge known historical regressors
future = future.merge(df_copy[['ds', 'discount', 'marketing']], on='ds', how='left')

# Fill in user-defined future values for forecast period
future.loc[future['ds'] > cutoff_date, 'discount'] = discount
future.loc[future['ds'] > cutoff_date, 'marketing'] = daily_marketing

# Fill any remaining NaNs (safety)
future['discount'] = future['discount'].fillna(0)
future['marketing'] = future['marketing'].fillna(0)

# Forecast
forecast = model.predict(future)

# Plot
st.write("### Forecast Plot")
# fig1 = model.plot(forecast)
# st.pyplot(fig1)

# Create Plotly figure
import plotly.graph_objects as go

# Define cutoff date
cutoff_date = pd.Timestamp("2025-06-30")

# Split forecast data
forecast_hist = forecast[forecast['ds'] <= cutoff_date]
forecast_future = forecast[forecast['ds'] > cutoff_date]

# Create Plotly figure
fig = go.Figure()

# Historical line (black, no markers)
fig.add_trace(go.Scatter(
    x=forecast_hist['ds'],
    y=forecast_hist['yhat'],
    mode='lines',
    name='Fitted Sales',
    line=dict(color='black')
))

# Forecast line (blue, no markers)
fig.add_trace(go.Scatter(
    x=forecast_future['ds'],
    y=forecast_future['yhat'],
    mode='lines',
    name='Forecasted Sales',
    line=dict(color='blue')
))

# Confidence interval (forecast only)
fig.add_trace(go.Scatter(
    x=forecast_future['ds'],
    y=forecast_future['yhat_upper'],
    mode='lines',
    line=dict(width=0),
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=forecast_future['ds'],
    y=forecast_future['yhat_lower'],
    fill='tonexty',
    fillcolor='rgba(0, 0, 255, 0.2)',
    line=dict(width=0),
    name='Confidence Interval'
))

# Layout
fig.update_layout(
    title="Sales Forecast (Historical vs Forecast)",
    xaxis_title="Date",
    yaxis_title="Sales",
    hovermode="x unified"
)

# Show in Streamlit
st.plotly_chart(fig, use_container_width=True)

# TRIAL

st.write("### Forecast Plot (Custom)")

fig, ax = plt.subplots(figsize=(10, 6))

# Plot actuals (if any in training set)
df_plot = df_train[df_train['ds'] <= cutoff_date]
ax.plot(df_plot['ds'], df_plot['y'], label="Historical Sales", color='black')

# Plot forecast
ax.plot(forecast['ds'], forecast['yhat'], label="Forecast", color='red')
ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                color='pink', alpha=0.3, label='Confidence Interval')

# Formatting
ax.axvline(cutoff_date, color='gray', linestyle='--', label='Forecast Start')
ax.set_title("Sales Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# TRIAL

st.markdown("---")

# TABLE SECTION

# Filter forecast for prediction months only (i.e., after cutoff_date)
forecast_pred = forecast[forecast['ds'] > cutoff_date].copy()

# Extract month
forecast_pred['month'] = forecast_pred['ds'].dt.to_period('M')

# Group and summarize
monthly_summary = forecast_pred.groupby('month').agg({
    'yhat': 'sum',
    'yhat_lower': 'sum',
    'yhat_upper': 'sum'
}).reset_index()

# Rename columns
monthly_summary.columns = ['Month', 'Predicted Sales', 'Lower End', 'Higher End']

# Round for display
monthly_summary[['Predicted Sales', 'Lower End', 'Higher End']] = monthly_summary[[
    'Predicted Sales', 'Lower End', 'Higher End']].round(0)

# Convert month back to string for nicer display
monthly_summary['Month'] = monthly_summary['Month'].astype(str)

# Display in Streamlit
st.write("### Monthly Forecast Summary")
st.dataframe(monthly_summary)

st.markdown("---")

# Filter forecast for prediction period only (after cutoff_date)
forecast_pred = forecast[forecast['ds'] > cutoff_date].copy()

# Select and rename relevant columns
daily_summary = forecast_pred[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
daily_summary.columns = ['Date', 'Predicted Sales', 'Lower End', 'Higher End']

# Round values for readability
daily_summary[['Predicted Sales', 'Lower End', 'Higher End']] = daily_summary[[
    'Predicted Sales', 'Lower End', 'Higher End']].round(0)

# Display in Streamlit
st.write("### Daily Forecast Details")
daily_summary['Date'] = pd.to_datetime(daily_summary['Date']).dt.date
st.dataframe(daily_summary)

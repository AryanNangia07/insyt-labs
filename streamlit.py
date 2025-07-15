import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import timedelta
import holidays
from sqlalchemy import create_engine

# Streamlit App Title
st.title("Sales Forecast by Category")

# Database credentials (you can switch back to st.secrets when deploying)
DB_USER = st.secrets["DB_USER"]
DB_PASSWORD = st.secrets["DB_PASSWORD"]
DB_HOST = st.secrets["DB_HOST"]
DB_PORT = st.secrets["DB_PORT"]
DB_NAME = st.secrets["DB_NAME"]

# Create SQLAlchemy engine using psycopg2
engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# Display Name → Actual SKU mapping
category_dict = {
    "Trimmers": {
        "FBT": "APPLIANCES_FULL_BODY_TRIMMER",
        "FBT SC": "APPLIANCES_FULL_BODY_TRIMMER_SPL_EDITION",
        "Power Styler": "APPLIANCES_POWER_STYLER_EDGE_TRIMMER"
    },
    "Perfumes": {
        "Venice": "PERFUME_VENICE_BLUE_30ML",
        "Veleno": "PERFUME_VELENO_100"
    }
}

# Category selection
selected_category = st.selectbox("Select Product Category", list(category_dict.keys()))
sku_mapping = category_dict[selected_category]

# Forecast horizon
months = st.selectbox("Select forecast horizon (months):", options=[1, 2, 3, 4, 5, 6], index=2)
forecast_days = months * 30

# Warning for longer horizons
if months > 3:
    st.warning("Forecasts beyond 3 months may be less accurate due to increased uncertainty.")

# Set cutoff date for training data
from datetime import datetime

# Get maximum date in your data (or fallback to today)
max_available_date = pd.to_datetime("today").normalize()

cutoff_options = {
    "Today": 0,
    "1 day ago": 1,
    "7 days ago": 7,
    "14 days ago": 14,
    "30 days ago": 30,
    "60 days ago": 60,
    "90 days ago": 90
}

cutoff_label = st.select_slider(
    "Select cutoff date for forecast:",
    options=list(cutoff_options.keys()),
    value="7 days ago"
)

cutoff_date = max_available_date - timedelta(days=cutoff_options[cutoff_label])
test_start_date = cutoff_date - timedelta(days=90)

st.caption(f"🔍 Forecasts are generated from data up to **{cutoff_date.date()}**. "
           f"Backtest checks accuracy from **{test_start_date.date()} to {cutoff_date.date()}**.")


# Define Indian holidays
ind_holidays = holidays.India(years=[2023, 2024, 2025])
holiday_df = pd.DataFrame([
    {"ds": pd.to_datetime(date), "holiday": "indian_holiday"}
    for date in ind_holidays.keys()
])

# Forecast results container
sku_forecasts = {}

# Forecast loop per SKU
for display_sku, db_sku in sku_mapping.items():
    try:
        query = """
        SELECT orderdate, sku, quantity 
        FROM shopify.operationalpnl
        WHERE sku = %s;
        """
        df = pd.read_sql(query, con=engine, params=(db_sku,))

        if df.empty:
            st.warning(f"No data found for {display_sku}. Skipping.")
            continue

        # Clean and format
        df = df.dropna(subset=["orderdate", "quantity"])
        df["orderdate"] = pd.to_datetime(df["orderdate"], errors="coerce")
         
         # ✅ Convert cutoff and test dates to datetime
        cutoff_date = pd.to_datetime(cutoff_date)
        test_start_date = pd.to_datetime(test_start_date)

        # ✅ Group by day and sum quantity sold
        daily_sales = (
            df.groupby("orderdate")["quantity"]
            .sum()
            .reset_index()
            .rename(columns={"orderdate": "ds", "quantity": "y"})
        )

        # Train cutoff
                # Train cutoff
        df_train = daily_sales[daily_sales["ds"] <= cutoff_date].copy()

        # Backtest setup: last 90 days before cutoff
        df_test = daily_sales[
            (daily_sales["ds"] > test_start_date) & (daily_sales["ds"] <= cutoff_date)
        ].copy()

        mape = None  # default value

        if not df_test.empty and len(df_test) >= 30:
            backtest_model = Prophet(
                growth="linear",
                daily_seasonality=True,
                weekly_seasonality=True,
                changepoint_prior_scale=0.3,
                holidays=holiday_df
            )
            backtest_model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
            backtest_model.fit(daily_sales[daily_sales["ds"] < test_start_date])

            days_to_forecast = (cutoff_date - test_start_date).days
            backtest_future = backtest_model.make_future_dataframe(periods=days_to_forecast)
            backtest_forecast = backtest_model.predict(backtest_future)
            backtest_forecast = backtest_forecast[["ds", "yhat"]].clip(lower=0)

            merged_bt = pd.merge(df_test, backtest_forecast, on="ds", how="left")
            merged_bt["error"] = abs(merged_bt["y"] - merged_bt["yhat"])
            merged_bt["pct_error"] = merged_bt["error"] / merged_bt["y"] * 100
            mape = merged_bt["pct_error"].mean()

        if df_train.empty:
            st.warning(f"Not enough historical data for {display_sku}. Skipping.")
            continue


        # Build Prophet model
        model = Prophet(
            growth="linear",
            daily_seasonality=True,
            weekly_seasonality=True,
            changepoint_prior_scale=0.3,
            holidays=holiday_df
        )
        model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
        model.fit(df_train)

        # Forecast future
        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)
        forecast[["yhat", "yhat_lower", "yhat_upper"]] = forecast[["yhat", "yhat_lower", "yhat_upper"]].clip(lower=0)

        # Extract only future months
        forecast_pred = forecast[forecast["ds"] > cutoff_date].copy()
        forecast_pred["month"] = forecast_pred["ds"].dt.to_period("M")
        monthly_sum = forecast_pred.groupby("month")["yhat"].sum().reset_index()
        monthly_sum.columns = ["Month", display_sku]
        sku_forecasts[display_sku] = monthly_sum

    except Exception as e:
        st.error(f"Error processing SKU {display_sku}: {e}")
        continue

# Merge forecast results
summary_df = pd.DataFrame()
for sku, df_sku in sku_forecasts.items():
    if summary_df.empty:
        summary_df = df_sku
    else:
        summary_df = pd.merge(summary_df, df_sku, on="Month", how="outer")

# Display results
# Display results
if not summary_df.empty:
    summary_df = summary_df.fillna(0)
    summary_df["Month"] = summary_df["Month"].astype(str)
    st.write("### Forecast Summary by SKU (Monthly)")
    st.dataframe(summary_df.set_index("Month").round(0))

    if mape is not None:
        st.caption(f"📉 **Model Accuracy (MAPE)** over last 90 days before cutoff: **{mape:.2f}%**")
else:
    st.info("No forecasts to display.")

import streamlit as st
import pandas as pd
from prophet import Prophet
from datetime import timedelta
import holidays
from sqlalchemy import create_engine
import traceback

# —––––– Streamlit App Title –––––—
st.title("Sales Forecast by Category")

# —––––– Database credentials (fill these) –––––—
DB_USER = st.secrets["DB_USER"]
DB_PASSWORD = st.secrets["DB_PASSWORD"]
DB_HOST = st.secrets["DB_HOST"]
DB_PORT = st.secrets["DB_PORT"]
DB_NAME = st.secrets["DB_NAME"]

# Create SQLAlchemy engine
engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# Display Name → Actual SKU mapping
category_dict = {
    "Trimmers": {
        "FBT": "APPLIANCES_FULL_BODY_TRIMMER",
        "FBT SC": "APPLIANCES_FULL_BODY_TRIMMER_SPL_EDITION",
        "Power Styler": "APPLIANCES_POWER_STYLER_EDGE_TRIMMER",
    },
    "Perfumes": {
        "Venice": "PERFUME_VENICE_BLUE_30ML",
        "Veleno": "PERFUME_VELENO_100",
        "Cairo": "PERFUME_EDP_CAIRO_BLACK_100ML",
    },
}

# Category selection
selected_category = st.selectbox(
    "Select Product Category",
    list(category_dict.keys())
)
sku_mapping = category_dict[selected_category]

# Forecast horizon
months = st.selectbox(
    "Select forecast horizon (months):",
    options=[1, 2, 3, 4, 5, 6],
    index=2
)
forecast_days = months * 30
if months > 3:
    st.warning("Forecasts beyond 3 months may be less accurate.")

# Cutoff date slider
max_available_date = pd.to_datetime("today").normalize()
cutoff_options = {
    "Today": 0, "1 day ago": 1, "7 days ago": 7,
    "14 days ago": 14, "30 days ago": 30,
    "60 days ago": 60, "90 days ago": 90,
}
cutoff_label = st.select_slider(
    "Select cutoff date for forecast:",
    options=list(cutoff_options.keys()),
    value="7 days ago",
)
cutoff_date = max_available_date - timedelta(days=cutoff_options[cutoff_label])
test_start_date = cutoff_date - timedelta(days=90)
cutoff_date = pd.to_datetime(cutoff_date)
test_start_date = pd.to_datetime(test_start_date)

st.caption(
    f"🔍 Data up to **{cutoff_date.date()}**; "
    f"backtest period: **{test_start_date.date()} to {cutoff_date.date()}**."
)

# Indian holidays
ind_holidays = holidays.India(years=[2023, 2024, 2025])
holiday_df = pd.DataFrame([
    {"ds": pd.to_datetime(d), "holiday": "indian_holiday"}
    for d in ind_holidays.keys()
])

# Container for forecasts and errors
sku_forecasts = {}
error_pct_3m = {}

for display_sku, db_sku in sku_mapping.items():
    try:
        # Fetch data
        query = """
            SELECT orderdate, quantity
            FROM shopify.operationalpnl
            WHERE sku = %s;
        """
        df = pd.read_sql(query, con=engine, params=(db_sku,))
        if df.empty:
            st.warning(f"No data for {display_sku}. Skipping.")
            continue

        # Clean & parse dates
        df = df.dropna(subset=["orderdate", "quantity"])
        df["orderdate"] = pd.to_datetime(df["orderdate"], errors="coerce")

        # Aggregate daily sales
        daily_sales = (
            df.groupby("orderdate")["quantity"]
              .sum()
              .reset_index()
              .rename(columns={"orderdate": "ds", "quantity": "y"})
        )

        # Split train/test
        df_train = daily_sales[daily_sales["ds"] <= cutoff_date].copy()
        df_test  = daily_sales[
            (daily_sales["ds"] > test_start_date) &
            (daily_sales["ds"] <= cutoff_date)
        ].copy()

        if df_train.empty:
            st.warning(f"Not enough historical data for {display_sku}. Skipping.")
            continue
        if df_test.empty:
            st.warning(f"No test‐period data for {display_sku}. Skipping backtest.")
            # we can still forecast future but skip error calc

        # Train backtest model on history before test window
        bt_model = Prophet(
            growth="linear",
            daily_seasonality=True,
            weekly_seasonality=True,
            changepoint_prior_scale=0.3,
            holidays=holiday_df
        )
        bt_model.add_seasonality("monthly", 30.5, fourier_order=5)
        bt_model.fit(daily_sales[daily_sales["ds"] < test_start_date])

        # Forecast for the 90‐day backtest window
        days_bt = (cutoff_date - test_start_date).days
        bt_future = bt_model.make_future_dataframe(periods=days_bt)
        bt_forecast = bt_model.predict(bt_future)[["ds", "yhat"]]
        bt_forecast["yhat"] = bt_forecast["yhat"].clip(lower=0)

        # Filter forecast to test window
        bt_pred_window = bt_forecast[
            (bt_forecast["ds"] > test_start_date) &
            (bt_forecast["ds"] <= cutoff_date)
        ].copy()

        # Sum actual vs predicted over entire 90 days
        sum_actual = df_test["y"].sum()
        sum_pred   = bt_pred_window["yhat"].sum()

        # Compute 3‑month error percentage
        if sum_actual > 0:
            err_pct = (sum_pred - sum_actual) / sum_actual * 100
        else:
            err_pct = None  # or handle zero‐actual case separately

        error_pct_3m[display_sku] = err_pct

        # --- Final forecast model on df_train ---
        model = Prophet(
            growth="linear",
            daily_seasonality=True,
            weekly_seasonality=True,
            changepoint_prior_scale=0.3,
            holidays=holiday_df
        )
        model.add_seasonality("monthly", 30.5, fourier_order=5)
        model.fit(df_train)

        # Forecast future period
        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)
        for col in ["yhat", "yhat_lower", "yhat_upper"]:
            forecast[col] = forecast[col].clip(lower=0)

        preds = forecast[forecast["ds"] > cutoff_date].copy()
        preds["month"] = preds["ds"].dt.to_period("M")
        monthly = preds.groupby("month")["yhat"].sum().reset_index()
        monthly.columns = ["Month", display_sku]
        sku_forecasts[display_sku] = monthly

    except Exception as e:
        st.error(f"Error processing SKU {display_sku}: {e}")
        st.text(traceback.format_exc())
        continue

# Merge all SKU forecasts
if sku_forecasts:
    summary_df = None
    for df_sku in sku_forecasts.values():
        summary_df = df_sku if summary_df is None else summary_df.merge(df_sku, on="Month", how="outer")
    summary_df = summary_df.fillna(0)
    summary_df["Month"] = summary_df["Month"].astype(str)
    st.write("### Forecast Summary by SKU (Monthly)")
    st.dataframe(summary_df.set_index("Month").round(0))

    # Display 3‑month error percentages
    st.write("### 3‑Month Backtest Error % by SKU")
    for sku, err in error_pct_3m.items():
        if err is None:
            st.caption(f"{sku}: actual sum = 0 (no error computed)")
        else:
            st.caption(f"{sku}: {(err):.2f}%")
else:
    st.info("No forecasts to display.")

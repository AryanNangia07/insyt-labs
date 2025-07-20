import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import timedelta
import holidays
from sqlalchemy import create_engine
import traceback
import plotly.graph_objects as go


# —––––– Streamlit App Title –––––—
st.title("Demand Forecast v3.0")

# —––––– Database credentials –––––—
DB_USER = st.secrets["DB_USER"]
DB_PASSWORD = st.secrets["DB_PASSWORD"]
DB_HOST = st.secrets["DB_HOST"]
DB_PORT = st.secrets["DB_PORT"]
DB_NAME = st.secrets["DB_NAME"]

# Create engine
engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# —––––– SKU Mapping –––––—
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
    "Bombae": {
        "Hairess": "BAE_3-IN-1_HAIR_MULTISTYLER",
        "4 in 1 Face Trimmer - FuzzOff": "BAE_4-IN-1_FAE_TRIMMER_FUZZOFF_1299",
        "6 in 1 Sensitive Trimmer - FuzzOff": "BAE_6-IN-1_SENS_TRIMMER_FUZZOFF_1599",
        "Mini Facial Hair Trimmer": "BAE_FACIAL_HAIR_MINI_TRIMMER",
        "Rollplay Pro Body Razor": "BAE_ROLLPLAY_PRO_BODY_RZR",
    }
}

# Category and SKU selection
default_cat = list(category_dict.keys())[0]
selected_category = st.selectbox("Select Product Category", list(category_dict.keys()), index=0)
sku_list = list(category_dict[selected_category].keys())
selected_skus = st.multiselect("Select SKUs to Forecast", sku_list, default=sku_list)
sku_mapping = {sku: category_dict[selected_category][sku] for sku in selected_skus}

# Forecast horizon
months = st.selectbox("Forecast horizon (months)", [1,2,3,4,5,6], index=2)
forecast_days = months * 30
if months > 3:
    st.warning("Forecasts beyond 3 months may be less accurate.")

# Cutoff date slider
max_date = pd.to_datetime("today").normalize()
opts = {"Today":0,"1 day ago":1,"7 days ago":7,"14 days ago":14,"30 days ago":30,"60 days ago":60,"90 days ago":90}
label = st.select_slider("Select cutoff date:", options=list(opts.keys()), value="7 days ago")
cutoff_date = max_date - timedelta(days=opts[label])
test_start = cutoff_date - timedelta(days=90)
st.caption(f"Data up to {cutoff_date.date()}, backtest {test_start.date()} to {cutoff_date.date()}")

# Future avg selling price inputs per SKU
projected_prices = {}
with st.expander("Set Projected Avg Selling Price per SKU", expanded=False):
    for sku, db_sku in sku_mapping.items():
        start = cutoff_date - timedelta(days=7)
        q = """
            SELECT SUM(grosssales) AS total_gross, SUM(quantity) AS total_qty
            FROM shopify.operationalpnl
            WHERE sku=%s AND orderdate BETWEEN %s AND %s;
        """
        r = pd.read_sql(q, con=engine, params=(db_sku, start, cutoff_date))
        val = 0.0
        if not r.empty and r.iloc[0]["total_qty"]>0:
            val = float(r.iloc[0]["total_gross"]/r.iloc[0]["total_qty"])
        projected_prices[sku] = st.number_input(
            f"Avg Price for {sku}", min_value=0.0, max_value=10000.0,
            value=round(val,2), step=10.0, key=f"price_{sku}" )

# Monthly marketing spend split by last month's spend ratios
monthly_mark = st.slider("Projected Monthly Marketing Spend (₹)", 0, 100_000_00, 0, 10_000)
# Determine last month period
first_this = cutoff_date.replace(day=1)
last_month_end = first_this - timedelta(days=1)
last_month_start = last_month_end.replace(day=1)
# Query last month spend per SKU
daily_mark = {}
spend_totals = {}
for sku, db_sku in sku_mapping.items():
    sq = """
        SELECT SUM(totalmarketingspend) AS spend
        FROM shopify.operationalpnl
        WHERE sku = %s
          AND orderdate BETWEEN %s AND %s;
    """
    dfm = pd.read_sql(sq, con=engine, params=(db_sku, last_month_start, last_month_end))
    spend_totals[sku] = dfm.iloc[0]["spend"] or 0
# Compute ratios
total_spend = sum(spend_totals.values()) or 1
ratios = {sku: spend / total_spend for sku, spend in spend_totals.items()}
# Distribute monthly_mark per SKU and per day
for sku, ratio in ratios.items():
    daily_mark[sku] = (monthly_mark * ratio) / 30

# Holidays
hols = holidays.India(years=[2023,2024,2025])
hol_df = pd.DataFrame([{"ds":pd.to_datetime(d),"holiday":"indian_holiday"} for d in hols.keys()])

# Containers
sku_forecasts = {}
error_pct = {}

# Forecast loop
for sku,db_sku in sku_mapping.items():
    df = pd.read_sql("SELECT orderdate,quantity,grosssales,totalmarketingspend FROM shopify.operationalpnl WHERE sku=%s;",
                      engine, params=(db_sku,))
    if df.empty:
        st.warning(f"No data for {sku}. Skipping.")
        continue
    df.dropna(subset=["orderdate","quantity"],inplace=True)
    df["orderdate"] = pd.to_datetime(df["orderdate"],errors="coerce")
    ds = df.groupby("orderdate").agg({"quantity":"sum","grosssales":"sum","totalmarketingspend":"sum"}).reset_index()
    ds.rename(columns={"orderdate":"ds","quantity":"y","totalmarketingspend":"marketing_spend"},inplace=True)
    ds["discount"] = ds["grosssales"]/ds["y"]
    ds.replace([np.inf,-np.inf],np.nan,inplace=True)
    ds.dropna(subset=["discount"],inplace=True)
    train = ds[ds["ds"]<=cutoff_date]
    test = ds[(ds["ds"]>test_start)&(ds["ds"]<=cutoff_date)]
    if train.empty:
        st.warning(f"Not enough history for {sku}. Skipping.")
        continue
    # Backtest
    bt = Prophet(daily_seasonality=True,weekly_seasonality=True,holidays=hol_df)
    bt.add_seasonality("monthly",30.5,5);bt.add_regressor("marketing_spend");bt.add_regressor("discount")
    bt.fit(train[train["ds"]<test_start][["ds","y","marketing_spend","discount"]])
    days_bt=(cutoff_date-test_start).days
    fbt = bt.make_future_dataframe(periods=days_bt).merge(ds[["ds","marketing_spend","discount"]],on="ds",how="left")
    fbt["marketing_spend"].fillna(daily_mark[sku],inplace=True)
    fbt["discount"].fillna(projected_prices[sku],inplace=True)
    pbt=bt.predict(fbt)[["ds","yhat"]];pbt["yhat"]=pbt["yhat"].clip(lower=0)
    bw=pbt[(pbt["ds"]>test_start)&(pbt["ds"]<=cutoff_date)]
    error_pct[sku]=(bw["yhat"].sum()-test["y"].sum())/test["y"].sum()*100 if test["y"].sum()>0 else None
    # Final forecast
    m=Prophet(daily_seasonality=True,weekly_seasonality=True,holidays=hol_df)
    m.add_seasonality("monthly",30.5,5);m.add_regressor("marketing_spend");m.add_regressor("discount")
    m.fit(train[["ds","y","marketing_spend","discount"]])
    fut=m.make_future_dataframe(periods=forecast_days).merge(ds[["ds","marketing_spend","discount"]],on="ds",how="left")
    fut["marketing_spend"].fillna(daily_mark[sku],inplace=True)
    fut["discount"].fillna(projected_prices[sku],inplace=True)
    fc=m.predict(fut)
    fc[["yhat","yhat_lower","yhat_upper"]]=fc[["yhat","yhat_lower","yhat_upper"]].clip(lower=0)
    # merge for plotting
    merged=pd.merge(ds[["ds","y"]].rename(columns={"y":"Actual"}),
                     fc[["ds","yhat"]].rename(columns={"yhat":"Forecast"}),
                     on="ds",how="outer").sort_values("ds")
    sku_forecasts[sku]=merged

# Display & interactive graph
if sku_forecasts:
    # Summary table per SKU
    summary_df=None
    for sku,df_sku in sku_forecasts.items():
        mon=(df_sku[df_sku["ds"]>cutoff_date].groupby(df_sku["ds"].dt.to_period("M"))["Forecast"].sum().reset_index())
        mon.columns=["Month",sku]
        summary_df=mon if summary_df is None else summary_df.merge(mon,on="Month",how="outer")
    summary_df.fillna(0,inplace=True);summary_df["Month"]=summary_df["Month"].astype(str)
    st.write("### Forecast Summary by SKU (Monthly)")
    st.dataframe(summary_df.set_index("Month").round(0))
    st.write("### 3‑Month Backtest Error % by SKU")
    for sku,err in error_pct.items(): st.caption(f"{sku}: {err:.2f}%" if err is not None else f"{sku}: no data")

    # Collapsible graph section with toggle
    with st.expander("📊 View Forecast Graphs", expanded=False):
        graph_skus = st.multiselect("Choose SKU(s) to display:", options=list(sku_forecasts.keys()), default=list(sku_forecasts.keys()))
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            if st.button("Create Graph"):
                fig = go.Figure()
                for sku in graph_skus:
                    dfp = sku_forecasts[sku]
                    fig.add_trace(go.Scatter(x=dfp['ds'], y=dfp['Actual'], mode='lines', name=f'{sku} - Actual', line=dict(color='black')))
                    fig.add_trace(go.Scatter(x=dfp['ds'], y=dfp['Forecast'], mode='lines', name=f'{sku} - Forecast', line=dict(color='blue', dash='dot')))
                fig.update_layout(title='Actual vs Forecasted Daily Sales by SKU', xaxis_title='Date', yaxis_title='Daily Sales', hovermode='x unified', template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No forecasts to display.")
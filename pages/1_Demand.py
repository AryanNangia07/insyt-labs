import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import timedelta
import holidays
from sqlalchemy import create_engine
from sqlalchemy.sql import text
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

# —––––– Category selection —––––—
category = st.selectbox(
    "Select Product Category:",
    options=["Bombae", "Fragrances", "Trimmers", "Shaving"],
    index=0
)

# Build category filter
if category == "Bombae":
    cat_filter = "sku LIKE 'BAE%'"
elif category == "Fragrances":
    cat_filter = "(sku LIKE 'PERFUME%' OR sku LIKE 'DEODRANT%')"
elif category == "Trimmers":
    cat_filter = "sku LIKE 'APPLIANCES%'"
elif category == "Shaving":
    cat_filter = "sku LIKE 'SHAVE%'"

# —––––– Fetch dynamic SKU list (active in last 14 days) —––––—
two_weeks_ago = (pd.Timestamp('today').normalize() - pd.Timedelta(days=14)).date()
sku_query = f"""
    SELECT DISTINCT sku
    FROM shopify.operationalpnl
    WHERE {cat_filter}
      AND orderdate >= '{two_weeks_ago}'
      AND quantity > 0
    ORDER BY sku
"""
sku_df = pd.read_sql_query(text(sku_query), con=engine)
sku_list = sku_df['sku'].tolist()

selected_skus = st.multiselect(
    "Select SKUs to Forecast:",
    options=sku_list,
    default=[]
)
if not selected_skus:
    st.error("Please select at least one SKU.")
    st.stop()
# Build SKU filter clause
sku_filter = "sku IN (" + ",".join(f"'{s}'" for s in selected_skus) + ")"

# —––––– Forecast horizon & cutoff —––––—
months = st.selectbox("Forecast horizon (months)", [1,2,3,4,5,6], index=2)
forecast_days = months * 30
if months > 3:
    st.warning("Forecasts beyond 3 months may be less accurate.")

max_date = pd.to_datetime('today').normalize()
cutoff_opts = {"Today":0, "7 days ago":7, "14 days ago":14, "30 days ago":30}
cutoff_label = st.select_slider("Select cutoff date:", options=list(cutoff_opts.keys()), value="7 days ago")
cutoff_date = max_date - timedelta(days=cutoff_opts[cutoff_label])
test_start = cutoff_date - timedelta(days=90)
st.caption(f"Data up to {cutoff_date.date()}, backtest {test_start.date()} to {cutoff_date.date()}")

# —––––– Projected avg price per SKU —––––—
projected_prices = {}
with st.expander("Set projected avg selling price per SKU", expanded=False):
    for sku in selected_skus:
        q = """
            SELECT SUM(grosssales) AS total_gross, SUM(quantity) AS total_qty
            FROM shopify.operationalpnl
            WHERE sku = %s
              AND orderdate BETWEEN %s AND %s;
        """
        r = pd.read_sql(q, con=engine, params=(sku, cutoff_date - timedelta(days=7), cutoff_date))
        default = 0.0
        if not r.empty and pd.notnull(r.iloc[0]['total_qty']) and r.iloc[0]['total_qty'] > 0:
            default = r.iloc[0]['total_gross'] / r.iloc[0]['total_qty']
        projected_prices[sku] = st.number_input(
            f"Avg Price for {sku}", min_value=0.0, max_value=10000.0,
            value=round(default,2), step=1.0, key=f"price_{sku}" )

# —––––– Monthly marketing spend split —––––—
monthly_spend = st.slider("Projected Monthly Marketing Spend (Category)", 0, 100_000_00, 0, 10_000)
first_this = cutoff_date.replace(day=1)
last_end = first_this - timedelta(days=1)
last_start = last_end.replace(day=1)
sp_totals = {}
for sku in selected_skus:
    q = """
        SELECT SUM(totalmarketingspend) AS spend
        FROM shopify.operationalpnl
        WHERE sku = %s
          AND orderdate BETWEEN %s AND %s;
    """
    r = pd.read_sql(q, con=engine, params=(sku, last_start, last_end))
    sp_totals[sku] = r.iloc[0]['spend'] or 0
# Compute ratios & daily spend
total_sp = sum(sp_totals.values()) or 1
daily_mark = {sku: (monthly_spend * (sp_totals[sku]/total_sp))/30 for sku in selected_skus}

# —––––– Holidays —––––—
hol_df = pd.DataFrame([{"ds": pd.to_datetime(d), "holiday":"indian"}
                       for d in holidays.India(years=[2023,2024,2025]).keys()])

# —––––– Forecast loop —––––—
sku_forecasts, error_pct = {}, {}
for sku in selected_skus:
    df = pd.read_sql_query(text(
        "SELECT orderdate, quantity AS y, grosssales, totalmarketingspend"
        " FROM shopify.operationalpnl WHERE sku = :sku"),
        con=engine, params={"sku": sku}
    )
    if df.empty:
        st.warning(f"No data for {sku}. Skipping.")
        continue
    df['orderdate'] = pd.to_datetime(df['orderdate'])
    ds = df.groupby('orderdate').agg({'y':'sum','grosssales':'sum','totalmarketingspend':'sum'})
    ds = ds.reset_index().rename(columns={'orderdate':'ds','totalmarketingspend':'marketing_spend'})
    ds['discount'] = ds['grosssales']/ds['y']
    ds.replace([np.inf,-np.inf],np.nan,inplace=True)
    ds.dropna(subset=['discount'], inplace=True)

    train = ds[ds['ds'] <= cutoff_date]
    test = ds[(ds['ds'] > test_start) & (ds['ds'] <= cutoff_date)]
    if train.empty:
        st.warning(f"Not enough history for {sku}. Skipping.")
        continue

    # Backtest
    bt = Prophet(daily_seasonality=True, weekly_seasonality=True, holidays=hol_df)
    bt.add_seasonality('monthly',30.5,5)
    bt.add_regressor('marketing_spend')
    bt.add_regressor('discount')
    bt.fit(train[train['ds'] < test_start][['ds','y','marketing_spend','discount']])
    days_bt = (cutoff_date-test_start).days
    fut_bt = bt.make_future_dataframe(periods=days_bt)
    fut_bt = fut_bt.merge(ds[['ds','marketing_spend','discount']], on='ds', how='left')
    fut_bt['marketing_spend'].fillna(daily_mark[sku], inplace=True)
    fut_bt['discount'].fillna(projected_prices[sku], inplace=True)
    pbt = bt.predict(fut_bt)[['ds','yhat']]
    pbt['yhat'] = pbt['yhat'].clip(lower=0)
    slice_bt = pbt[(pbt['ds']>test_start)&(pbt['ds']<=cutoff_date)]
    error_pct[sku] = ((slice_bt['yhat'].sum() - test['y'].sum())/test['y'].sum()*100
                      if test['y'].sum()>0 else None)

    # Final forecast
    m = Prophet(daily_seasonality=True, weekly_seasonality=True, holidays=hol_df)
    m.add_seasonality('monthly',30.5,5)
    m.add_regressor('marketing_spend')
    m.add_regressor('discount')
    m.fit(train[['ds','y','marketing_spend','discount']])
    fut = m.make_future_dataframe(periods=forecast_days)
    fut = fut.merge(ds[['ds','marketing_spend','discount']], on='ds', how='left')
    fut['marketing_spend'].fillna(daily_mark[sku], inplace=True)
    fut['discount'].fillna(projected_prices[sku], inplace=True)
    fc = m.predict(fut)
    fc[['yhat','yhat_lower','yhat_upper']] = fc[['yhat','yhat_lower','yhat_upper']].clip(lower=0)
    merged = pd.merge(ds[['ds','y']].rename(columns={'y':'Actual'}),
                      fc[['ds','yhat']].rename(columns={'yhat':'Forecast'}),
                      on='ds', how='outer').sort_values('ds')
    sku_forecasts[sku] = merged

# —––––– Display & graph —––––—
if sku_forecasts:
    summary_df = None
    for sku, df_sku in sku_forecasts.items():
        mon = (df_sku[df_sku['ds']>cutoff_date]
               .groupby(df_sku['ds'].dt.to_period('M'))['Forecast']
               .sum().reset_index())
        mon.columns = ['Month', sku]
        summary_df = mon if summary_df is None else summary_df.merge(mon, on='Month', how='outer')
    summary_df.fillna(0, inplace=True)
    summary_df['Month'] = summary_df['Month'].astype(str)

    st.write('### Forecast Summary by SKU (Monthly)')
    st.dataframe(summary_df.set_index('Month').round(0))
    st.write('### 3‑Month Backtest Error % by SKU')
    for sku, err in error_pct.items():
        st.caption(f"{sku}: {err:.2f}%" if err is not None else f"{sku}: no data")

    with st.expander('📊 View Forecast Graphs', expanded=False):
        graph_skus = st.multiselect('Choose SKU(s):', list(sku_forecasts.keys()), default=list(sku_forecasts.keys()))
        c1, c2, c3 = st.columns([1,2,1])
        with c2:
            if st.button('Create Graph'):
                fig = go.Figure()
                for sku in graph_skus:
                    dfp = sku_forecasts[sku]
                    fig.add_trace(go.Scatter(x=dfp['ds'], y=dfp['Actual'], mode='lines', name=f'{sku} - Actual'))
                    fig.add_trace(go.Scatter(x=dfp['ds'], y=dfp['Forecast'], mode='lines', name=f'{sku} - Forecast', line=dict(dash='dot')))
                fig.update_layout(
                    title='Actual vs Forecasted Daily Sales by SKU',
                    xaxis_title='Date', yaxis_title='Daily Sales',
                    hovermode='x unified', template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
else:
    st.info('No forecasts to display.')

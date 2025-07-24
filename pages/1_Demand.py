import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import timedelta
import holidays
from sqlalchemy import create_engine, text
import traceback
import plotly.graph_objects as go
from pandas.tseries.offsets import MonthBegin
import io

# —––––– Streamlit App Title –––––—
st.title("Demand Forecast v3.0")

# —––––– Database credentials –––––—
DB_USER = st.secrets["DB_USER"]
DB_PASSWORD = st.secrets["DB_PASSWORD"]
DB_HOST = st.secrets["DB_HOST"]
DB_PORT = st.secrets["DB_PORT"]
DB_NAME = st.secrets["DB_NAME"]

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# —––––– Category selection —––––—
category = st.selectbox(
    "Select Product Category:",
    options=["Skin", "Bombae", "Fragrances", "Trimmers", "Shaving"]
)
cat_filter_map = {
    "Bombae": "sku LIKE 'BAE%'",
    "Fragrances": "(sku LIKE 'PERFUME%' OR sku LIKE 'DEODRANT%')",
    "Trimmers": "sku LIKE 'APPLIANCES%'",
    "Shaving": "sku LIKE 'SHAVE%'",
    "Skin": "sku LIKE 'SKIN%'"
}
cat_filter = cat_filter_map.get(category, "1=1")

# —––––– Dynamic SKU filtering —––––—
today = pd.Timestamp('today').normalize()
two_weeks_ago = today - timedelta(days=14)
sku_query = f"""
    SELECT DISTINCT sku
    FROM shopify.operationalpnl
    WHERE {cat_filter}
      AND orderdate >= '{two_weeks_ago.date()}'
      AND quantity > 0
"""
sku_df = pd.read_sql_query(text(sku_query), con=engine)
skus = sku_df['sku'].tolist()
def_month = today.replace(day=1)
three_months_ago = def_month - pd.DateOffset(months=3)
filter_query = f"""
    SELECT sku, DATE_TRUNC('month', orderdate)::date AS m, SUM(quantity) AS qty
    FROM shopify.operationalpnl
    WHERE {cat_filter}
      AND orderdate >= '{three_months_ago.date()}'
    GROUP BY sku, m
"""
mon = pd.read_sql_query(text(filter_query), con=engine)
piv = mon.pivot(index='sku', columns='m', values='qty').fillna(0)
last3 = sorted(piv.columns)[-3:]
available_skus = [s for s in piv.index if (piv.loc[s, last3] >= 30).all()]
selected_skus = st.multiselect(
    "Select SKUs to Forecast:", options=[s for s in skus if s in available_skus]
)
if not selected_skus:
    st.error("Please select at least one SKU.")
    st.stop()

# —––––– Forecast horizon —––––—
months = st.selectbox("Forecast horizon (months)", [1,2,3,4,5,6], index=2)
forecast_months = months
if months > 3:
    st.warning("Forecasts beyond 3 months may be less accurate.")

# —––––– Regressor inputs —––––—
projected_prices = {}
with st.expander("Set projected avg selling price per SKU", expanded=False):
    for sku in selected_skus:
        q = text(
            "SELECT COALESCE(SUM(grosssales),0) AS total_gross, COALESCE(SUM(quantity),0) AS total_qty"
            " FROM shopify.operationalpnl"
            " WHERE sku=:sku AND orderdate BETWEEN :start AND :end"
        )
        r = pd.read_sql_query(
            q, con=engine,
            params={"sku":sku, "start":today-timedelta(days=7), "end":today}
        )
        default = (r.at[0,'total_gross']/r.at[0,'total_qty']) if r.at[0,'total_qty']>0 else 0.0
        projected_prices[sku] = st.number_input(
            f"Avg Price for {sku}", min_value=0.0, max_value=10000.0,
            value=round(default,2), step=1.0, key=f"price_{sku}"
        )

# —––––– Marketing spend slider —––––—
first = today.replace(day=1)
last_end = first - timedelta(days=1)
last_start = last_end.replace(day=1)
csq = text(
    f"SELECT COALESCE(SUM(totalmarketingspend),0) AS spend"
    f" FROM shopify.operationalpnl WHERE {cat_filter}"
    f" AND orderdate BETWEEN :start AND :end"
)
cat_spend = pd.read_sql_query(csq, con=engine, params={"start":last_start, "end":last_end}).at[0,'spend']
monthly_spend = st.slider(
    "Projected monthly marketing spend (Category)", 0, 500_000_00, int(cat_spend), step=10000
)
sp_totals = {}
for sku in selected_skus:
    sq = text(
        "SELECT COALESCE(SUM(totalmarketingspend),0) AS spend"
        " FROM shopify.operationalpnl WHERE sku=:sku AND orderdate BETWEEN :start AND :end"
    )
    r = pd.read_sql_query(sq, con=engine, params={"sku":sku, "start":last_start, "end":last_end})
    sp_totals[sku] = r.at[0,'spend']
sum_sp = cat_spend or 1
daily_mark = {sku: (monthly_spend * (sp_totals[sku]/sum_sp)) / 30 for sku in selected_skus}

# —––––– Holidays —––––—
hol_df = pd.DataFrame([
    {"ds":pd.to_datetime(d),"holiday":"indian_holiday"}
    for d in holidays.India(years=[2023,2024,2025]).keys()
])

# —––––– Forecast & Backtest —––––—
sku_forecasts = {}
sku_daily_forecasts = {}
error_pct = {}
for sku in selected_skus:
    df = pd.read_sql_query(
        text("SELECT orderdate, quantity AS y, grosssales, totalmarketingspend FROM shopify.operationalpnl WHERE sku=:sku"),
        con=engine, params={"sku":sku}
    )
    if df.empty:
        continue
    df['orderdate'] = pd.to_datetime(df['orderdate'])
    ds = df.groupby('orderdate').agg({'y':'sum','grosssales':'sum','totalmarketingspend':'sum'}).reset_index()
    ds.rename(columns={'orderdate':'ds','totalmarketingspend':'marketing_spend'}, inplace=True)
    ds['discount'] = ds['grosssales']/ds['y']
    ds.replace([np.inf,-np.inf],np.nan, inplace=True);
    ds.dropna(subset=['discount'], inplace=True)

    # Backtest
    test_start = today - pd.DateOffset(months=3)
    train = ds[ds['ds'] < today]
    bt = Prophet(daily_seasonality=True, weekly_seasonality=True, holidays=hol_df, interval_width=0.8)
    bt.add_seasonality('monthly',30.5,5); bt.add_regressor('marketing_spend'); bt.add_regressor('discount')
    bt.fit(train[train['ds']<test_start][['ds','y','marketing_spend','discount']])
    fut_bt = bt.make_future_dataframe(periods=(today-test_start).days)
    fut_bt = fut_bt.merge(ds[['ds','marketing_spend','discount']], on='ds', how='left')
    fut_bt['marketing_spend'].fillna(daily_mark[sku], inplace=True)
    fut_bt['discount'].fillna(projected_prices[sku], inplace=True)
    pbt = bt.predict(fut_bt)[['ds','yhat']]
    pbt['yhat'] = pbt['yhat'].clip(lower=0)
    slice_bt = pbt[(pbt['ds']>test_start)&(pbt['ds']<=today)]
    actual_bt = ds[(ds['ds']>test_start)&(ds['ds']<=today)]['y'].sum()
    error_pct[sku] = ((slice_bt['yhat'].sum()-actual_bt)/actual_bt*100) if actual_bt>0 else None

    # Forecast daily
    m = Prophet(daily_seasonality=True, weekly_seasonality=True, holidays=hol_df, interval_width=0.95)
    m.add_seasonality('monthly',30.5,5); m.add_regressor('marketing_spend'); m.add_regressor('discount')
    m.fit(train[['ds','y','marketing_spend','discount']])
    start_next = today.replace(day=1) + MonthBegin(1)
    m_dates = pd.date_range(start=start_next, periods=forecast_months, freq='MS')
    future_dates = []
    for d in m_dates:
        end = d + MonthBegin(1) - timedelta(days=1)
        future_dates.extend(pd.date_range(d, end))
    fut = pd.DataFrame({'ds':future_dates}).merge(ds[['ds','marketing_spend','discount']], on='ds', how='left')
    fut['marketing_spend'].fillna(daily_mark[sku], inplace=True)
    fut['discount'].fillna(projected_prices[sku], inplace=True)
    fc = m.predict(fut)
    fc['yhat'] = fc['yhat'].clip(lower=0)
    sku_daily_forecasts[sku] = fc[['ds']].assign(Forecast=fc['yhat'])

    # monthly summary
    fc['Month'] = fc['ds'].dt.to_period('M')
    sku_forecasts[sku] = fc.groupby('Month')['yhat'].sum().reset_index().rename(columns={'yhat':'Forecast'})

# —––––– Display —––––—
if sku_forecasts:
    # Monthly table
    df_cat = pd.concat({k:v.set_index('Month')['Forecast'] for k,v in sku_forecasts.items()}, axis=1)
    st.markdown('### Monthly Forecast')
    st.dataframe(df_cat.fillna(0).astype(int))

    # Backtest Accuracy expander
    with st.expander('📈 Backtest Accuracy (3mo)'):
        for sku, err in error_pct.items():
            st.caption(f"{sku}: {err:.2f}%" if err is not None else f"{sku}: No data")

    # Daily graphs expander
    with st.expander('📊 Daily Actual vs Forecast'):
        daily_choice = st.multiselect(
            'Select SKUs for Daily plot', selected_skus,
            default=selected_skus, key='daily_skus')
        if st.button('Create Daily Graph', key='daily_button'):
            fig = go.Figure()
            for sku in daily_choice:
                # Fetch daily actuals for this SKU
                hist_q = text(
                    "SELECT orderdate, SUM(quantity) AS y "
                    "FROM shopify.operationalpnl WHERE sku=:sku "
                    "GROUP BY orderdate ORDER BY orderdate"
                )
                df_hist = pd.read_sql_query(
                    hist_q, con=engine, params={"sku": sku}
                )
                df_hist['orderdate'] = pd.to_datetime(df_hist['orderdate'])
                fig.add_trace(go.Scatter(
                    x=df_hist['orderdate'], y=df_hist['y'], mode='lines',
                    name=f'{sku} Actual'
                ))
                # Forecast daily
                df_fc = sku_daily_forecasts.get(sku)
                if df_fc is not None:
                    fig.add_trace(go.Scatter(
                        x=df_fc['ds'], y=df_fc['Forecast'], mode='lines',
                        name=f'{sku} Forecast', line=dict(dash='dot')
                    ))
            fig.update_layout(
                title='Daily Actual vs Forecast', xaxis_title='Date', yaxis_title='Units',
                template='plotly_white', legend=dict(orientation='h', y=-0.2, x=0.5, xanchor='center')
            )
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info('No forecast data available.')

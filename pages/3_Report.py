import streamlit as st
import pandas as pd
from prophet import Prophet
from sqlalchemy import create_engine, text
from pandas.tseries.offsets import MonthBegin, DateOffset
import io
import re

# --- DATABASE CONNECTION ---
DB_USER = st.secrets["DB_USER"]
DB_PASSWORD = st.secrets["DB_PASSWORD"]
DB_HOST = st.secrets["DB_HOST"]
DB_PORT = st.secrets["DB_PORT"]
DB_NAME = st.secrets["DB_NAME"]

engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# --- PAGE SETUP ---
st.set_page_config(page_title="Master Forecast Report", layout="wide")
st.title("📊 Master Forecast Report")

# --- INPUTS SECTION ---
st.markdown("---")
st.subheader("Forecast Inputs")

# Compute default spend (avg last 3 months)
tmp_sql = text(
    """
    SELECT orderdate AS ds, totalmarketingspend
    FROM shopify.operationalpnl
    WHERE quantity > 0
      AND orderdate >= (current_date - interval '3 months')
    """
)
tmp_df = pd.read_sql_query(tmp_sql, con=engine)
tmp_df['ds'] = pd.to_datetime(tmp_df['ds'])
monthly_spend = tmp_df.set_index('ds')['totalmarketingspend'].resample('MS').sum().fillna(0)
default_spend = float(monthly_spend.mean().round(2))

# Main-page sliders
total_spend = st.slider(
    "Total Marketing Spend Across All SKUs (per month)",
    min_value=0,
    max_value=int(default_spend * 3),
    value=int(default_spend),
    step=1000
)
horizon_months = st.slider(
    "Forecast Horizon (Months)",
    min_value=1,
    max_value=6,
    value=3
)

# --- PRE-COMPUTE LAST MONTH RATIOS ---
hist_sql = text(
    """
    SELECT orderdate AS ds, sku, totalmarketingspend
    FROM shopify.operationalpnl
    WHERE quantity > 0
    """
)
hist_df = pd.read_sql_query(hist_sql, con=engine)
hist_df['ds'] = pd.to_datetime(hist_df['ds'])
today = pd.Timestamp('today').normalize()
prev_end = today.replace(day=1) - pd.Timedelta(days=1)
prev_start = prev_end.replace(day=1)
last_month = hist_df[(hist_df['ds'] >= prev_start) & (hist_df['ds'] <= prev_end)]
spend_by_sku = last_month.groupby('sku')['totalmarketingspend'].sum()
total_last = spend_by_sku.sum() or 1
ratios = (spend_by_sku / total_last).to_dict()

# --- MASTER FORECAST ---
if st.button("📥 Generate Master Report"):
    with st.spinner("Running forecasts…"):
        today = pd.Timestamp('today').normalize()
        start_next = today.replace(day=1) + MonthBegin(1)
        end_date = start_next + DateOffset(months=horizon_months) - pd.Timedelta(days=1)
        future_dates = pd.date_range(start_next, end_date, freq='D')

        # Filter SKUs by recent sales
        first_month = today.replace(day=1)
        three_months_ago = first_month - DateOffset(months=3)
        filter_q = text(f"""
            SELECT sku, DATE_TRUNC('month', orderdate) AS month, SUM(quantity) AS monthly_qty
            FROM shopify.operationalpnl
            WHERE quantity > 0
              AND orderdate >= '{three_months_ago.date()}'
            GROUP BY sku, month
        """
        )
        filt_df = pd.read_sql_query(filter_q, con=engine)
        pivot = filt_df.pivot(index='sku', columns='month', values='monthly_qty').fillna(0)
        last3 = sorted(pivot.columns)[-3:]
        good_skus = pivot[(pivot[last3] >= 30).all(axis=1)].index.tolist()

        sku_sql = text("SELECT DISTINCT sku FROM shopify.operationalpnl WHERE quantity > 0")
        sku_df = pd.read_sql_query(sku_sql, con=engine)
        prefixes = ['SKIN','BAE','PERFUME','DEODRANT','APPLIANCES','SHAVE']
        skus = [s for s in sku_df['sku'] if any(s.startswith(p) for p in prefixes) and s in good_skus]

        hist3_sql = text(f"""
            SELECT sku, DATE_TRUNC('month', orderdate) AS month, SUM(quantity) AS sold
            FROM shopify.operationalpnl
            WHERE quantity > 0
              AND orderdate >= '{three_months_ago.date()}'
              AND sku IN ({', '.join(repr(s) for s in skus)})
            GROUP BY sku, month
        """
        )
        hist3 = pd.read_sql_query(hist3_sql, con=engine)
        hist3_pivot = hist3.pivot(index='sku', columns='month', values='sold').fillna(0)
        hist3_pivot = hist3_pivot.reset_index().rename(columns={'sku': 'SKU'})
        hist3_pivot.columns = ['SKU'] + [m.strftime('%Y-%m') for m in hist3_pivot.columns[1:]]

        progress = st.progress(0)
        rows = []
        for i, sku in enumerate(skus, 1):
            df = pd.read_sql_query(
                text("SELECT orderdate, quantity AS y, totalmarketingspend FROM shopify.operationalpnl WHERE sku = :sku"),
                con=engine, params={'sku': sku}
            )
            if df.empty: continue
            df['orderdate'] = pd.to_datetime(df['orderdate'])
            ds = df.groupby('orderdate').agg({'y': 'sum', 'totalmarketingspend': 'sum'}).reset_index()
            ds.rename(columns={'orderdate': 'ds', 'totalmarketingspend': 'marketing_spend'}, inplace=True)

            ratio = ratios.get(sku, 1/len(skus))
            budget = total_spend * ratio
            daily = budget / len(future_dates)
            fut = pd.DataFrame({'ds': future_dates, 'marketing_spend': daily})

            if len(ds) < 2: continue
            m = Prophet(daily_seasonality=True, weekly_seasonality=True, interval_width=0.95)
            m.add_seasonality('monthly', 30.5, 5)
            m.add_regressor('marketing_spend')
            m.fit(ds[['ds','y','marketing_spend']])

            fc = m.predict(fut)
            fc['yhat'] = fc['yhat'].clip(lower=0)
            fc['Month'] = fc['ds'].dt.to_period('M')
            monthly_sum = fc.groupby('Month')['yhat'].sum().round().astype(int).reset_index()
            monthly_sum['SKU'] = sku
            monthly_sum['Category'] = monthly_sum['SKU'].apply(
                lambda x: 'Skin' if x.startswith('SKIN') else
                          'Bombae' if x.startswith('BAE') else
                          'Fragrances' if x.startswith(('PERFUME','DEODRANT')) else
                          'Trimmers' if x.startswith('APPLIANCES') else
                          'Shaving' if x.startswith('SHAVE') else 'Other'
            )
            rows.append(monthly_sum)
            progress.progress(i / len(skus))

        if not rows:
            st.error("No forecasts generated—verify data availability.")
        else:
            result = pd.concat(rows, ignore_index=True)
            out = result.pivot_table(index=['SKU', 'Category'], columns='Month', values='yhat').reset_index()

            # Convert forecast month columns to strings
            out.columns = ['SKU', 'Category'] + [p.strftime('%Y-%m') for p in out.columns[2:]]

            # Correct merge with historical
            final = out.merge(hist3_pivot, on='SKU', how='left')

            # Reorder columns safely
            all_cols = [str(c) for c in final.columns]
            hist_cols = sorted([c for c in all_cols if re.match(r'\d{4}-\d{2}', c) and c in hist3_pivot.columns])
            fc_cols = [c for c in all_cols if c not in ['SKU', 'Category'] + hist_cols]
            final = final[['SKU', 'Category'] + hist_cols + fc_cols]

            # Export to Excel
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                final.to_excel(writer, index=False, sheet_name='Forecast')
            buf.seek(0)

            st.success("✅ Report ready for download")
            st.download_button(
                "📥 Download Master Forecast Excel",
                data=buf.getvalue(),
                file_name="master_forecast.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

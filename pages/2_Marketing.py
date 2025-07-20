import streamlit as st
import pandas as pd
from prophet import Prophet
from sqlalchemy import create_engine
from datetime import timedelta
import plotly.graph_objects as go

# —––––– Page config –––––—
st.set_page_config(page_title="Marketing Spend Predictor", layout="wide")
st.title("🎯 Marketing Spend Predictor")

# —––––– DB connection —––––—
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
    "Select Category:",
    options=["BSC", "Bombae"],
    index=0
)
# Build SKU filter clause based on category
default_filter = "1=1"
if category == "BSC":
    sku_filter = "sku NOT LIKE 'BAE%'"
elif category == "Bombae":
    sku_filter = "sku LIKE 'BAE%'"
else:
    sku_filter = default_filter

# —––––– Input: choose target type —––––—
target_type = st.radio(
    "Choose Target Metric:",
    options=["Quantity", "Revenue"],
    index=0,
    horizontal=True
)
if target_type == "Quantity":
    sales_target = st.number_input(
        "Enter your quantity target for next month (units)",
        min_value=0.0, step=1.0
    )
    y_col = "quantity"
    y_label = "Predicted Quantity"
else:
    sales_target = st.number_input(
        "Enter your revenue target for next month (₹)",
        min_value=0.0, step=1000.0
    )
    y_col = "grosssales"
    y_label = "Predicted Revenue"

# —––––– Estimate on button click —––––—
if st.button("Estimate Required Marketing Spend"):
    # 1. Fetch historical monthly aggregates filtered by category
    query = f"""
        SELECT
            DATE_TRUNC('month', orderdate)::date AS ds,
            SUM({y_col}) AS y,
            SUM(totalmarketingspend) AS spend
        FROM shopify.operationalpnl
        WHERE {sku_filter}
        GROUP BY 1
        ORDER BY 1
    """
    from sqlalchemy import text
    df = pd.read_sql_query(text(query), con=engine)
    if df.empty or df['spend'].sum() == 0:
        st.error("Not enough historical data to build model for this category.")
        st.stop()

    # 2. Train Prophet model with spend regressor and 95% CI
    m = Prophet(growth="linear", interval_width=0.95)
    m.add_regressor('spend')
    m.fit(df[['ds', 'y', 'spend']])

    # 3. Build spend grid for next month
    next_month = df['ds'].max() + pd.DateOffset(months=1)
    max_spend = int(df['spend'].max() * 2)
    spends = list(range(0, max_spend + 10000, 10000))
    future = pd.DataFrame({'ds': [next_month] * len(spends), 'spend': spends})

    # 4. Forecast for each spend level
    fc = m.predict(future)
    fc['spend'] = spends
    fc['error'] = (fc['yhat'] - sales_target).abs()

    # 5. Best spend estimate
    best = fc.loc[fc['error'].idxmin()]
    recommended = int(best['spend'])

    # 6. Confidence interval on spend
    valid_lower = fc[fc['yhat_upper'] >= sales_target]
    lower = int(valid_lower['spend'].min()) if not valid_lower.empty else None
    valid_upper = fc[fc['yhat_lower'] >= sales_target]
    upper = int(valid_upper['spend'].max()) if not valid_upper.empty else None

    # 7. Display metrics
    st.metric(
        label="📈 Recommended Monthly Marketing Spend",
        value=f"₹{recommended:,}"
    )
    if lower is not None and upper is not None:
        st.caption(f"95% CI on spend: ₹{lower:,} – ₹{upper:,}")
    else:
        st.caption("Confidence bounds unavailable for this target.")

    # 8. Plot Sales vs Spend curve
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fc['spend'], y=fc['yhat'], mode='lines', name=y_label, line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=fc['spend'], y=fc['yhat_lower'], mode='lines', name='Lower 95% CI', line=dict(color='lightblue', dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=fc['spend'], y=fc['yhat_upper'], mode='lines', name='Upper 95% CI', line=dict(color='lightblue', dash='dash')
    ))
    fig.update_layout(
        title=f"Forecasted {y_label} vs Marketing Spend for {next_month.date()}",
        xaxis_title="Marketing Spend (₹)", yaxis_title=y_label,
        hovermode='x unified', template='plotly_white', height=500
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Enter a target and press the button to estimate spend.")
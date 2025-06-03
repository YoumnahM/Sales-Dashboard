import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly
from streamlit_option_menu import option_menu
from prophet.plot import plot_plotly, plot_components
import matplotlib.pyplot as plt
import io

# Page config
st.set_page_config(page_title="Sales Dashboard", layout="wide")

# Load data
df = pd.read_csv('cleaned_sales_data.csv', parse_dates=['date'])
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y') 

# Feature engineering
df['month'] = df['date'].dt.month
df['month_name'] = df['date'].dt.strftime('%B')
df['year'] = df['date'].dt.year
df['day_of_week'] = df['date'].dt.dayofweek
df['day_name'] = df['date'].dt.strftime('%A')
df['day_type'] = df['day_name'].apply(lambda x: 'Weekend' if x in ['Saturday', 'Sunday'] else 'Weekday')

# Sidebar filters
st.sidebar.header("🔎 Filter Data")
years = sorted(df['year'].unique())
months = sorted(df['month_name'].unique(), key=lambda m: pd.to_datetime(m, format='%B').month)

selected_year = st.sidebar.selectbox("Select Year", years, index=len(years)-1)
selected_months = st.sidebar.multiselect("Select Month(s)", months, default=months)

# CSS for selected multiselect options (in sidebar)
st.markdown(
    """
    <style>
    /* Sidebar background */
    .st-cw {  /* sidebar container */
        background: linear-gradient(135deg, #9ebbdc 0%, #007fff 100%);
        color: #003f5c;
    }
    /* Multiselect container background */
    div[role="listbox"] {
        background-color: #e6f0ff !important;
        border-radius: 8px;
    }
    /* You can add more styles here for text color if needed */
    </style>
    """,
    unsafe_allow_html=True,
)

# Filtered data
df_filtered = df[(df['year'] == selected_year) & (df['month_name'].isin(selected_months))]

# Aggregated data
monthly_avg = df_filtered.groupby('month_name')['sales'].mean().reindex(months).reset_index()

# Forecast data
# Prepare data
prophet_df = df[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})

# Train model
model = Prophet()
model.fit(prophet_df)

# Predict future
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Prepare forecast display
forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30).rename(columns={
    'ds': 'Date',
    'yhat': 'Predicted Sales',
    'yhat_lower': 'Lower Confidence Bound',
    'yhat_upper': 'Upper Confidence Bound'
})

# KPI
total_sales = int(df_filtered['sales'].sum())
avg_price = round(df_filtered['price'].mean(), 2)
avg_stock = int(df_filtered['stock'].mean())

# Navigation
selected = option_menu(
    menu_title=None,
    options=["Dashboard", "Forecast", "Raw Data"],
    icons=["bar-chart", "graph-up", "table"],
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "#4b4b4b", "font-size": "18px"},
        "nav-link": {
            "font-size": "16px",
            "text-align": "center",
            "margin": "0px",
            "--hover-color": "#eee",
        },
        "nav-link-selected": {"background-color": "#9ebbdc", "color": "white"},
    }
)

if selected == "Dashboard":
    st.title("📊 Sales Performance Dashboard")
    
    # KPI Calculations
    total_sales = df_filtered['sales'].sum()
    avg_price = round(df_filtered['price'].mean(), 2)
    avg_stock = int(df_filtered['stock'].mean())
    total_revenue = round(df_filtered['revenue'].sum(), 2)
    revenue_per_sale = round(total_revenue / total_sales, 2) if total_sales > 0 else 0
    # stock_turnover = round(total_sales / df_filtered['stock'].mean(), 2)
    
    # Best-selling product
    top_product_data = df_filtered.groupby('product_name')['sales'].sum()
    if not top_product_data.empty:
        top_product = top_product_data.idxmax()
        top_sales = int(top_product_data.max())
    else:
        top_product = "N/A"
        top_sales = 0

    # Card CSS styles
    st.markdown("""
<style>
    .card-container {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        justify-content: space-between;
        margin-bottom: 1.5rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        flex: 1;
        min-width: 250px;
        text-align: center;
        color: #034d4d;  /* dark teal text for good contrast */
        font-weight: 600;
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
    }
    .card h3 {
        margin-bottom: 0.5rem;
        font-size: 1.2rem;
    }
    .card p {
        font-size: 1.5rem;
        margin: 0;
        font-weight: 700;
    }

    /* Lighter pastel blue gradients */
    .card1 { background: linear-gradient(135deg, #f8fbff 0%, #A7C7E7 100%); } /* ultra pale to soft blue */
    .card2 { background: linear-gradient(135deg, #f7faff 0%, #9ebbdc 100%); } /* near white to soft blue */
    .card3 { background: linear-gradient(135deg, #f5f9ff 0%, #96afcf 100%); } /* very pale blue to muted blue */
    .card4 { background: linear-gradient(135deg, #f3f7ff 0%, #8ea3c2 100%); } /* whisper blue to light blue-gray */
    .card5 { background: linear-gradient(135deg, #f2f6ff 0%, #8696b5 100%); } /* very light blue to desaturated blue */
    .card6 { background: linear-gradient(135deg, #f0f5ff 0%, #7d8aaa 100%); } /* pastel blue to soft dusty blue */
    .card7 { background: linear-gradient(135deg, #eef4ff 0%, #75809e 100%); } /* softest blue to muted dusty blue */


</style>
""", unsafe_allow_html=True)

    # Display KPIs inside cards
    st.markdown(f"""
    <div class="card-container">
        <div class="card card1">
            <h3>🛒 Total Sales</h3>
            <p>{total_sales:,}</p>
        </div>
        <div class="card card2">
            <h3>💲 Avg. Price</h3>
            <p>${avg_price}</p>
        </div>
        <div class="card card3">
            <h3>📦 Avg. Stock</h3>
            <p>{avg_stock}</p>
        </div>
        <div class="card card4">
            <h3>🧾 Total Revenue</h3>
            <p>${total_revenue:,.2f}</p>
        </div>
        <div class="card card5">
            <h3>📈 Revenue per Sale</h3>
            <p>${revenue_per_sale:.2f}</p>
        </div>
        <div class="card card6">
            <h3>🔥 Best-Selling Product</h3>
            <p>{top_product} ({top_sales} units)</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    # Sales over time
    fig1 = px.line(df_filtered, x='date', y='sales', title="📈 Sales Over Time",
                   color_discrete_sequence=['#00b4d8'])
    fig1.update_layout(xaxis_title="Date", yaxis_title="Sales", template="plotly_dark")
    col1.plotly_chart(fig1, use_container_width=True)

    # Monthly average sales
    fig2 = px.bar(monthly_avg, x='month_name', y='sales', title="📅 Avg Sales by Month",
                  color='sales', color_continuous_scale='viridis')
    fig2.update_layout(xaxis_title="Month", yaxis_title="Average Sales", template="plotly_white")
    col2.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 📦 Outlier Detection")
    fig3 = px.box(df_filtered, y='sales', points="all", title="📉 Sales Distribution",
                  color_discrete_sequence=['#f77f00'])
    fig3.update_layout(template="plotly_white")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### 🎞️ Sales Animation by Day")
    animated = px.bar(df_filtered, x='day_name', y='sales', animation_frame='month_name',
                      category_orders={'day_name': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']},
                      color='sales', color_continuous_scale='Plasma', title="Sales by Day (Animated)")
    animated.update_layout(xaxis_title="Day", yaxis_title="Sales", template="plotly_white")
    st.plotly_chart(animated, use_container_width=True)

    # Category-wise sales pie chart
    st.markdown("### 🛍️ Sales by Product Category")
    fig4 = px.pie(df_filtered, names='product_category', values='sales', title="Sales Distribution by Category")
    st.plotly_chart(fig4, use_container_width=True)

    # Top 10 products bar chart
    st.markdown("### 🏆 Top 10 Products by Sales")
    top10 = df_filtered.groupby('product_name')['sales'].sum().nlargest(10).reset_index()
    fig5 = px.bar(top10, x='product_name', y='sales', title="Top 10 Best-Selling Products", color='sales',
                  color_continuous_scale='Blues')
    fig5.update_layout(xaxis_title="Product", yaxis_title="Sales", template="plotly_white")
    st.plotly_chart(fig5, use_container_width=True)

     # Heatmap of Sales by Day of Week and Month
    st.markdown("### 🌡️ Sales Heatmap (Day of Week vs Month)")
    
    heatmap_data = df_filtered.groupby(['month_name', 'day_name'])['sales'].mean().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='day_name', columns='month_name', values='sales')

    # Ensure correct order
    ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    ordered_months = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
    heatmap_pivot = heatmap_pivot.reindex(index=ordered_days, columns=ordered_months)

    fig6 = px.imshow(
        heatmap_pivot,
        labels=dict(x="Month", y="Day of Week", color="Avg Sales"),
        x=heatmap_pivot.columns,
        y=heatmap_pivot.index,
        color_continuous_scale="GnBu",
        aspect="auto",
        title="📆 Heatmap of Average Sales"
    )
    st.plotly_chart(fig6, use_container_width=True)


# Forecast Section
elif selected == "Forecast":
    st.title("🔮 Forecasting Future Sales")

    # Forecast Plot
    st.plotly_chart(plot_plotly(model, forecast), use_container_width=True)

    # Forecast Data Table & Download
    with st.expander("📄 Forecast Data Table"):
        st.dataframe(forecast_display)

        # Download CSV
        csv = forecast_display.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Forecast CSV",
            data=csv,
            file_name="forecast_30_days.csv",
            mime="text/csv"
        )


elif selected == "Raw Data":
    st.title("🧾 Raw Data")
    st.dataframe(df_filtered, use_container_width=True)
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Filtered Data", data=csv, file_name="filtered_sales_data.csv", mime='text/csv')

# Footer
footer = """
<style>
footer { visibility: hidden; }
.footer-style {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #001f3f;
    color: white;
    text-align: center;
    padding: 10px;
    font-size: 14px;
    opacity: 0.8;
    z-index: 1000;
}
</style>
<div class="footer-style">
    © Copyright 2025 DTG Labs - All rights reserved.
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
import datetime
import io
import pandas as pd
import streamlit as st

from Predict_freight import predict_freight_cost
from predict_demand import (
    prepare_demand_features,
    predict_product_demand,
    predict_batch_demand,
    generate_sample_batch_csv,
    WAREHOUSES,
    CATEGORIES
)
from predict_invoice_flag import predict_invoice_flag

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Supply Chain & Vendor Intelligence Portal",
    page_icon="📦",
    layout="wide"
)

# -------------------------------
# Header
# -------------------------------
st.title("📦 Supply Chain & Vendor Intelligence Portal")
st.subheader("AI-Driven Freight Cost Prediction, Invoice Risk Flagging & Demand Forecasting")

st.markdown("""
This unified enterprise analytics portal leverages machine learning to:
- 🚛 **Forecast Freight Costs** accurately to optimize logistics spend.
- 🚨 **Detect Risky Vendor Invoices** to prevent overbilling and delays.
- 📊 **Predict Product Demand** across warehouses to prevent stockouts and excess inventory.
""")

st.divider()

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("🔍 Navigation")

selected_model = st.sidebar.radio(
    "Choose Prediction Module",
    (
        "Freight Cost Prediction",
        "Invoice Manual Approval Flag",
        "Product Demand Forecasting"
    )
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### Business Impact
- 📈 **Logistics Optimization**: Precise freight cost budgeting
- 🧾 **Financial Security**: Automated discrepancy detection
- 📦 **Inventory Health**: Demand-driven inventory replenishment
- ⚡ **Operational Speed**: Instant AI-backed decision making
""")

# ===========================================================
# 1. Freight Prediction
# ===========================================================
if selected_model == "Freight Cost Prediction":

    st.header("🚛 Freight Cost Prediction")
    st.write("Estimate vendor shipping and freight costs based on shipment volume and invoice value.")

    with st.form("freight_form"):
        col1, col2 = st.columns(2)

        with col1:
            quantity = st.number_input(
                "Shipment Quantity (Units)",
                min_value=1,
                value=1200,
                step=10
            )

        with col2:
            dollars = st.number_input(
                "Invoice Amount ($)",
                min_value=1.0,
                value=18500.0,
                step=100.0
            )

        submit = st.form_submit_button("🚀 Predict Freight Cost", use_container_width=True)

    if submit:
        input_data = {
            "Quantity": [quantity],
            "Dollars": [dollars]
        }

        result = predict_freight_cost(input_data)
        freight = result["Predict_Freight"][0]

        st.success("✅ Prediction Generated Successfully!")

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric(
                label="Estimated Freight Cost",
                value=f"${freight:,.2f}"
            )
        with col_m2:
            freight_ratio = (freight / dollars) * 100 if dollars > 0 else 0
            st.metric(
                label="Freight-to-Invoice Ratio",
                value=f"{freight_ratio:.2f}%"
            )
        with col_m3:
            cost_per_unit = freight / quantity if quantity > 0 else 0
            st.metric(
                label="Freight Cost Per Unit",
                value=f"${cost_per_unit:.3f}"
            )

        st.subheader("Summary")
        st.dataframe(result, use_container_width=True)

# ===========================================================
# 2. Invoice Flag Prediction
# ===========================================================
elif selected_model == "Invoice Manual Approval Flag":

    st.header("🚨 Invoice Risk & Manual Approval Evaluation")
    st.write("Audit incoming vendor invoices against purchase orders to identify billing discrepancies and shipment delays.")

    with st.form("invoice_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            invoice_quantity = st.number_input(
                "Invoice Billed Quantity",
                min_value=1,
                value=50
            )
            freight = st.number_input(
                "Freight Charge ($)",
                min_value=0.0,
                value=1.73,
                step=0.1
            )

        with col2:
            invoice_dollars = st.number_input(
                "Invoice Billed Amount ($)",
                min_value=1.0,
                value=352.95,
                step=10.0
            )
            total_item_quantity = st.number_input(
                "PO Total Item Quantity",
                min_value=1,
                value=162
            )

        with col3:
            total_item_dollars = st.number_input(
                "PO Total Item Amount ($)",
                min_value=1.0,
                value=2476.0,
                step=50.0
            )

        submit = st.form_submit_button("🔍 Evaluate Invoice Risk", use_container_width=True)

    if submit:
        input_data = {
            "invoice_quantity": [invoice_quantity],
            "invoice_dollars": [invoice_dollars],
            "Freight": [freight],
            "total_item_quantity": [total_item_quantity],
            "total_item_dollars": [total_item_dollars]
        }

        result = predict_invoice_flag(input_data)
        flag = int(result["Predict_Flag"][0])
        dollar_discrepancy = abs(invoice_dollars - total_item_dollars)

        if flag == 1:
            st.error("🚨 **Invoice Requires Manual Approval** — Potential price mismatch or delivery anomaly detected.")
        else:
            st.success("✅ **Invoice Approved Automatically** — All audit checks passed.")

        col_i1, col_i2 = st.columns(2)
        with col_i1:
            st.metric(
                label="Invoice Dollar Difference vs PO",
                value=f"${dollar_discrepancy:,.2f}",
                delta=f"{'-' if dollar_discrepancy > 5 else '+'}${dollar_discrepancy:.2f}",
                delta_color="inverse"
            )
        with col_i2:
            st.metric(
                label="Audit Status",
                value="Flagged (Manual Review)" if flag == 1 else "Auto-Approved"
            )

        st.subheader("Prediction Details")
        st.dataframe(result, use_container_width=True)

# ===========================================================
# 3. Product Demand Forecasting
# ===========================================================
else:

    st.header("📊 Product Demand Forecasting")
    st.write("Predict future order demand for specific SKUs across regional warehouses using historical trends and cyclical patterns.")

    tab1, tab2 = st.tabs(["🎯 Single SKU Forecast", "📁 Batch CSV Forecast & Download"])

    with tab1:
        with st.form("demand_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                product_code = st.text_input(
                    "Product SKU Code",
                    value="Product_0993"
                )
                warehouse = st.selectbox(
                    "Regional Warehouse",
                    options=WAREHOUSES,
                    index=2  # Whse_J
                )

            with col2:
                product_category = st.selectbox(
                    "Product Category",
                    options=CATEGORIES,
                    index=27  # Category_028
                )
                forecast_date = st.date_input(
                    "Forecast Target Date",
                    value=datetime.date.today()
                )

            with col3:
                lag_1 = st.number_input(
                    "Yesterday's Order Demand (Lag 1)",
                    min_value=0.0,
                    value=500.0,
                    step=50.0
                )
                lag_7 = st.number_input(
                    "7-Day Prior Demand (Lag 7)",
                    min_value=0.0,
                    value=480.0,
                    step=50.0
                )

            col4, col5, col6 = st.columns(3)
            with col4:
                lag_30 = st.number_input(
                    "30-Day Prior Demand (Lag 30)",
                    min_value=0.0,
                    value=450.0,
                    step=50.0
                )
            with col5:
                rolling_mean_7 = st.number_input(
                    "7-Day Rolling Mean Demand",
                    min_value=0.0,
                    value=490.0,
                    step=25.0
                )
            with col6:
                rolling_std_7 = st.number_input(
                    "7-Day Rolling Standard Deviation",
                    min_value=0.0,
                    value=45.0,
                    step=5.0
                )

            submit = st.form_submit_button("📈 Forecast Product Demand", use_container_width=True)

        if submit:
            features_df = prepare_demand_features(
                product_code=product_code,
                warehouse=warehouse,
                product_category=product_category,
                forecast_date=forecast_date,
                lag_1=lag_1,
                lag_7=lag_7,
                lag_30=lag_30,
                rolling_mean_7=rolling_mean_7,
                rolling_std_7=rolling_std_7
            )

            predicted_units = predict_product_demand(features_df)

            st.success("✅ Forecast Generated Successfully!")

            col_d1, col_d2, col_d3 = st.columns(3)
            with col_d1:
                st.metric(
                    label="Predicted Order Demand",
                    value=f"{predicted_units:,.0f} units"
                )
            with col_d2:
                demand_delta = predicted_units - rolling_mean_7
                st.metric(
                    label="Expected Trend vs 7-Day Average",
                    value=f"{'+' if demand_delta >= 0 else ''}{demand_delta:,.0f} units",
                    delta=f"{demand_delta:,.0f} units"
                )
            with col_d3:
                # Safety stock buffer recommendation (Z=1.65 for 95% service level)
                recommended_stock = int(predicted_units + (1.65 * rolling_std_7))
                st.metric(
                    label="Recommended Inventory Stock",
                    value=f"{recommended_stock:,.0f} units"
                )

            st.subheader("Model Input Features")
            st.dataframe(features_df, use_container_width=True)

    with tab2:
        st.subheader("📁 Bulk SKU Demand Forecasting")
        st.write("Upload a CSV file containing multiple SKUs and historical sales to forecast demand across your entire inventory in seconds.")

        # Download Sample Template
        sample_df = generate_sample_batch_csv()
        csv_buffer = io.StringIO()
        sample_df.to_csv(csv_buffer, index=False)

        col_b1, col_b2 = st.columns([1, 2])
        with col_b1:
            st.download_button(
                label="📥 Download Sample CSV Template",
                data=csv_buffer.getvalue(),
                file_name="sample_demand_forecast_template.csv",
                mime="text/csv",
                use_container_width=True
            )

        uploaded_file = st.file_uploader("Upload CSV for Bulk Forecasting", type=["csv"])

        if uploaded_file is not None:
            input_batch_df = pd.read_csv(uploaded_file)
            st.write("Uploaded Data Preview:")
            st.dataframe(input_batch_df.head(10), use_container_width=True)

            if st.button("🚀 Generate Batch Forecasts", use_container_width=True):
                with st.spinner("Calculating demand forecasts across all SKUs..."):
                    forecasted_batch_df = predict_batch_demand(input_batch_df)

                st.success(f"✅ Successfully generated forecasts for {len(forecasted_batch_df)} SKUs!")

                # Display Visual Chart
                st.subheader("📊 Forecasted Demand by SKU")
                chart_data = forecasted_batch_df[["Product_Code", "Predicted_Order_Demand"]].set_index("Product_Code")
                st.bar_chart(chart_data)

                # Display Table
                st.subheader("📋 Forecasted Results & Recommendations")
                st.dataframe(forecasted_batch_df, use_container_width=True)

                # Download Results Button
                out_buffer = io.StringIO()
                forecasted_batch_df.to_csv(out_buffer, index=False)
                st.download_button(
                    label="💾 Download Forecasted Predictions CSV",
                    data=out_buffer.getvalue(),
                    file_name="forecasted_product_demand_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
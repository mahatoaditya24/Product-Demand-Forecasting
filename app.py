"""
Supply Chain & Vendor Invoice Intelligence System - Interactive Streamlit Portal.
Root entrypoint configured for Streamlit Community Cloud and local execution.
"""

import sys
from pathlib import Path

# Add project root and Inference directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent
INFERENCE_DIR = ROOT_DIR / "Inference"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))

import datetime
import io
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from Inference.Predict_freight import predict_freight_cost
from Inference.predict_demand import (
    prepare_demand_features,
    predict_product_demand,
    predict_batch_demand,
    generate_sample_batch_csv,
    WAREHOUSES,
    CATEGORIES
)
from Inference.predict_invoice_flag import predict_invoice_flag

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Supply Chain & Vendor Intelligence Portal",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-title {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Header
# -------------------------------
st.title("📦 Supply Chain & Vendor Intelligence Portal")
st.markdown("AI-Powered Freight Cost Forecasting, Invoice Risk Auditing & Multi-Warehouse SKU Demand Prediction.")

# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/delivery-box.png", width=64)
    st.title("Supply Chain AI")
    st.caption("Enterprise Operations Suite")
    st.markdown("---")

    selected_model = st.radio(
        "Choose Prediction Module",
        (
            "🚛 Freight Cost Prediction",
            "🚨 Invoice Risk & Approval Flag",
            "📊 Product Demand Forecasting"
        )
    )

    st.markdown("---")
    st.markdown("""
    ### 🎯 Business Value
    - **Logistics**: Accurate freight cost budgeting
    - **Finance**: Automated audit & risk classification
    - **Inventory**: ML-driven safety stock sizing
    - **REST API**: Available on `/api/v1/predict`
    """)


# ===========================================================
# 1. Freight Prediction
# ===========================================================
if selected_model == "🚛 Freight Cost Prediction":

    st.header("🚛 Freight Cost Prediction")
    st.write("Forecast vendor shipping and logistics expenses based on shipment volume and invoice valuation.")

    with st.form("freight_form"):
        col1, col2 = st.columns(2)

        with col1:
            quantity = st.number_input("Shipment Quantity (Units)", min_value=1, value=1200, step=50)

        with col2:
            dollars = st.number_input("Invoice Valuation ($)", min_value=1.0, value=18500.0, step=250.0)

        submit_freight = st.form_submit_button("🚀 Calculate Estimated Freight", use_container_width=True)

    if submit_freight:
        input_data = {"Quantity": [quantity], "Dollars": [dollars]}
        result = predict_freight_cost(input_data)
        freight = float(result["Predict_Freight"][0])

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Predicted Freight Cost", f"${freight:,.2f}")
        with col_m2:
            st.metric("Freight-to-Invoice Ratio", f"{(freight / dollars) * 100:.2f}%")
        with col_m3:
            st.metric("Cost per Shipped Unit", f"${freight / quantity:.2f}")

        st.success(f"✅ Predicted Logistics Expense: **${freight:,.2f}** for shipment of **{quantity:,} units** (${dollars:,.2f} invoice).")


# ===========================================================
# 2. Invoice Flagging
# ===========================================================
elif selected_model == "🚨 Invoice Risk & Approval Flag":

    st.header("🚨 Vendor Invoice Risk & Audit Classification")
    st.write("Cross-validate vendor invoice billing against PO receipt totals to detect overbilling, quantity mismatches, and delays.")

    with st.form("invoice_form"):
        st.subheader("1. Vendor Invoice Details")
        c1, c2, c3 = st.columns(3)
        with c1:
            inv_qty = st.number_input("Invoice Quantity", min_value=1, value=50, step=5)
        with c2:
            inv_dollars = st.number_input("Invoice Dollars ($)", min_value=1.0, value=352.95, step=25.0)
        with c3:
            freight_fee = st.number_input("Freight Amount ($)", min_value=0.0, value=1.73, step=0.5)

        st.subheader("2. Purchase Order (PO) & Warehouse Receipt Totals")
        c4, c5 = st.columns(2)
        with c4:
            po_qty = st.number_input("Total PO Quantity", min_value=1, value=162, step=5)
        with c5:
            po_dollars = st.number_input("Total PO Line-Item Dollars ($)", min_value=1.0, value=2476.0, step=50.0)

        submit_invoice = st.form_submit_button("🔍 Run Invoice Risk Audit", use_container_width=True)

    if submit_invoice:
        input_data = {
            "invoice_quantity": [inv_qty],
            "invoice_dollars": [inv_dollars],
            "Freight": [freight_fee],
            "total_item_quantity": [po_qty],
            "total_item_dollars": [po_dollars]
        }

        res = predict_invoice_flag(input_data)
        flag = int(res["Predict_Flag"][0])
        dollar_diff = abs(inv_dollars - po_dollars)
        qty_diff = abs(inv_qty - po_qty)

        col_i1, col_i2, col_i3 = st.columns(3)
        with col_i1:
            st.metric("Audit Status", "🚨 FLAGGED" if flag == 1 else "🟢 APPROVED")
        with col_i2:
            st.metric("Dollar Discrepancy", f"${dollar_diff:,.2f}")
        with col_i3:
            st.metric("Quantity Variance", f"{qty_diff} Units")

        if flag == 1:
            st.error(f"⚠️ **Action Required: Hold Payment.** Discrepancy of **${dollar_diff:,.2f}** detected between vendor invoice and received line items. Route to Finance Auditor.")
        else:
            st.success("✅ **Straight-Through Processing Approved.** Invoice line-item variance is within acceptable tolerances.")


# ===========================================================
# 3. Demand Forecasting
# ===========================================================
else:
    st.header("📊 Multi-Warehouse SKU Demand Forecasting")
    st.write("Forecast SKU demand across regional distribution centers and calculate safety stock buffer recommendations.")

    tab_single, tab_batch = st.tabs(["🎯 Single SKU Forecast", "📁 Batch CSV Upload"])

    with tab_single:
        with st.form("demand_form"):
            c_d1, c_d2, c_d3 = st.columns(3)
            with c_d1:
                p_code = st.text_input("Product Code", value="Product_0993")
            with c_d2:
                whse = st.selectbox("Distribution Center", options=WAREHOUSES, index=2)
            with c_d3:
                cat = st.selectbox("Product Category", options=CATEGORIES, index=27)

            f_date = st.date_input("Target Forecast Date", value=datetime.date.today())

            st.write("##### Historical Demand Signals & Rolling Statistics")
            c_l1, c_l2, c_l3, c_l4, c_l5 = st.columns(5)
            with c_l1:
                lag1 = st.number_input("Lag 1 (Yesterday)", min_value=0.0, value=520.0, step=10.0)
            with c_l2:
                lag7 = st.number_input("Lag 7 (Last Week)", min_value=0.0, value=500.0, step=10.0)
            with c_l3:
                lag30 = st.number_input("Lag 30 (Last Month)", min_value=0.0, value=480.0, step=10.0)
            with c_l4:
                rm7 = st.number_input("7-Day Rolling Mean", min_value=0.0, value=510.0, step=10.0)
            with c_l5:
                rs7 = st.number_input("7-Day Rolling StdDev", min_value=0.0, value=45.0, step=5.0)

            submit_demand = st.form_submit_button("🔮 Predict Demand & Safety Stock", use_container_width=True)

        if submit_demand:
            feat_df = prepare_demand_features(p_code, whse, cat, f_date, lag1, lag7, lag30, rm7, rs7)
            pred_demand = predict_product_demand(feat_df)
            safety_stock = int(round(pred_demand + (1.65 * rs7)))

            col_d1, col_d2, col_d3 = st.columns(3)
            with col_d1:
                st.metric("Predicted Demand", f"{pred_demand:,.1f} Units")
            with col_d2:
                st.metric("Recommended Safety Stock", f"{safety_stock:,} Units")
            with col_d3:
                st.metric("Target Warehouse", whse)

            st.info(f"📦 Recommended replenishment level for **{p_code}** at **{whse}** on **{f_date}**: **{safety_stock:,} Units** (95% service SLA).")

    with tab_batch:
        st.subheader("📁 Batch CSV Prediction Engine")
        st.caption("Upload multiple SKU historical series to forecast demand across your fulfillment network in bulk.")

        sample_csv = generate_sample_batch_csv()
        st.download_button(
            "📥 Download Sample CSV Template",
            data=sample_csv.to_csv(index=False).encode("utf-8"),
            file_name="sample_demand_batch_template.csv",
            mime="text/csv"
        )

        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
        if uploaded_file:
            input_df = pd.read_csv(uploaded_file)
            st.write("##### Input Data Preview:")
            st.dataframe(input_df.head(10), use_container_width=True)

            if st.button("🚀 Process Batch Forecasts", use_container_width=True):
                batch_res = predict_batch_demand(input_df)
                st.write("##### Forecast Results:")
                st.dataframe(batch_res, use_container_width=True)

                st.download_button(
                    "💾 Download Prediction Results CSV",
                    data=batch_res.to_csv(index=False).encode("utf-8"),
                    file_name="demand_predictions_output.csv",
                    mime="text/csv"
                )

# Footer
st.markdown("---")
st.caption("Supply Chain Intelligence Platform | Machine Learning Serving via FastAPI & Streamlit")

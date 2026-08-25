# 📦 Supply Chain & Vendor Invoice Intelligence System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mahatoaditya24-product-demand-forecasting-inferenceapp-mvyy16.streamlit.app/)
[![CI Tests](https://github.com/mahatoaditya24/Product-Demand-Forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/mahatoaditya24/Product-Demand-Forecasting/actions)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/mahatoaditya24/Product-Demand-Forecasting)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> 🚀 **Live Interactive Demo**: [https://mahatoaditya24-product-demand-forecasting-inferenceapp-mvyy16.streamlit.app/](https://mahatoaditya24-product-demand-forecasting-inferenceapp-mvyy16.streamlit.app/)

An end-to-end enterprise machine learning system and interactive analytics portal for e-commerce and retail supply chain operations (inspired by large-scale logistics ecosystems like Flipkart and Walmart).

The system addresses three core operational bottlenecks:
1. **Freight Cost Prediction**: Forecasting logistics and shipping costs based on invoice valuation and order units.
2. **Vendor Invoice Risk Assessment**: Automated anomaly detection and risk scoring to flag irregular invoices requiring manual finance audit.
3. **Product Demand Forecasting**: Large-scale time-series tabular regression predicting future SKU-level order demand across regional fulfillment centers.

---

## 🏗️ Architecture & System Design

```
Flipkart_2/
├── data/
│   ├── Product_Demand_main.csv        # Product historical demand records (1M+ rows)
│   └── inventory.db                   # SQLite database (purchases, invoices, inventory)
├── Freight Cost Prediction/
│   ├── data_preprocessing.py          # Data ingestion and train/test feature preparation
│   ├── model_evaluation.py            # Regression evaluation (Linear Regression, Decision Tree, Random Forest)
│   ├── train.py                       # Automated model training and selection
│   └── models/
│       └── predict_freight_model.pkl  # Serialized freight prediction model
├── Invoice flagging/
│   ├── data_preprocessing.py          # SQL feature extraction, rule labeling, StandardScaler
│   ├── modeling_evaluation.py         # GridSearchCV Random Forest Classifier (F1-score)
│   ├── train.py                       # Automated classification training pipeline
│   └── models/
│       ├── predict_flag_invoice.pkl   # Serialized invoice risk model
│       └── scaler.pkl                 # Fitted standard scaler
├── Notebook/
│   ├── Demand_forecast.ipynb          # Exploratory analysis & benchmark for demand forecasting
│   ├── Invoice Flagging .ipynb        # Exploratory analysis for invoice flagging
│   ├── Predicting Freight Cost.ipynb  # Exploratory analysis for freight prediction
│   └── best_demand_forecasting_model.pkl # High-performance demand forecast model
├── Inference/
│   ├── app.py                         # Streamlit interactive enterprise dashboard
│   ├── Predict_freight.py             # Inference wrapper for freight cost
│   ├── predict_invoice_flag.py        # Inference wrapper for invoice risk
│   └── predict_demand.py              # Inference wrapper for product demand
├── requirements.txt                   # Project dependencies
├── .gitignore                         # Git exclusion rules
└── README.md                          # Project documentation
```

---

## 🚀 Machine Learning Modules

### 1. 🚛 Freight Cost Prediction
- **Objective**: Accurately forecast transportation and freight expenses.
- **Input Features**: `Quantity` (units shipped), `Dollars` (invoice value).
- **Models Benchmarked**: Linear Regression, Decision Tree Regressor, Random Forest Regressor.
- **Optimization Metric**: Lowest Mean Absolute Error ($\text{MAE}$).

### 2. 🚨 Invoice Risk & Manual Approval Flagging
- **Objective**: Prevent financial leakage by auditing vendor invoices against line-item purchase orders.
- **Rules & Synthetic Label**:
  - PO vs. Invoice discrepancy: $|\text{invoice\_dollars} - \text{total\_item\_dollars}| > \$5.00$.
  - Shipping delay: $\text{avg\_receiving\_delay} > 10\text{ days}$.
- **Features**: `invoice_quantity`, `invoice_dollars`, `Freight`, `total_item_quantity`, `total_item_dollars`.
- **Model**: `RandomForestClassifier` tuned via 5-Fold Cross-Validated `GridSearchCV` on $F_1\text{-score}$.

### 3. 📊 Product Demand Forecasting
- **Objective**: Predict future demand across 2,160 products and 4 regional warehouses (`Whse_A`, `Whse_C`, `Whse_J`, `Whse_S`).
- **Feature Engineering**:
  - **Calendar & Temporal**: Year, Month, Day, Weekday, Quarter.
  - **Cyclical Encoding**: $\sin(2\pi \cdot \text{month}/12)$, $\cos(2\pi \cdot \text{month}/12)$, $\sin(2\pi \cdot \text{weekday}/7)$, $\cos(2\pi \cdot \text{weekday}/7)$.
  - **Historical Lags**: $\text{lag}_1$, $\text{lag}_7$, $\text{lag}_{30}$.
  - **Rolling Window Metrics**: 7-day rolling mean & standard deviation.
- **Performance**:
  - **Random Forest**: $R^2 \approx 0.72$, $\text{MAE} \approx 3,988$
  - **XGBoost**: $R^2 \approx 0.69$, $\text{MAE} \approx 4,436$

---

## 💻 Interactive Streamlit Portal

The web portal provides a multi-page interface to interact with all three models in real time:

```bash
# Launch the dashboard
streamlit run Inference/app.py
```

### Dashboard Capabilities:
- **Freight Cost Calculator**: Instant dollar estimation, freight-to-invoice ratio, and per-unit transport cost.
- **Invoice Auditor**: Instant compliance validation, dollar discrepancy metric, and manual review alert badge.
- **Demand Forecaster**: SKU and warehouse selector, date forecasting, and automated recommended safety stock buffers.

---

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mahatoaditya24/Product-Demand-Forecasting.git
   cd Product-Demand-Forecasting
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Retrain models (Optional)**:
   ```bash
   python "Freight Cost Prediction/train.py"
   python "Invoice flagging/train.py"
   ```

---

## 📈 Business Benefits

- **Cost Reduction**: Reduces unbudgeted logistics overruns and overpayment errors.
- **Operational Speed**: Automates straight-through invoice approvals, reducing finance cycle times from days to seconds.
- **Inventory Efficiency**: Balances warehouse stock levels against predicted customer demand to minimize holding costs and eliminate stockouts.

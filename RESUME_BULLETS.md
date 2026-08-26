# 📄 CV / Resume & Interview Package: Supply Chain & Vendor Intelligence Platform

Use this guide to integrate the **Supply Chain & Vendor Invoice Intelligence System** directly into your CV/Resume, LinkedIn profile, and technical interview discussions.

---

## 🎯 1. Ready-to-Copy Resume Bullet Points

### Option A: Lead / Senior ML Engineer & Data Scientist (Recommended)
> **Enterprise Supply Chain & Vendor Intelligence System** | *Python, Scikit-Learn, XGBoost, Docker, Streamlit, Git*  
> *GitHub:* [https://github.com/mahatoaditya24/Product-Demand-Forecasting](https://github.com/mahatoaditya24/Product-Demand-Forecasting)  
> *Live Demo:* [https://mahatoaditya24-product-demand-forecasting-inferenceapp-mvyy16.streamlit.app/](https://mahatoaditya24-product-demand-forecasting-inferenceapp-mvyy16.streamlit.app/)
> - Built an end-to-end multi-module machine learning platform serving real-time logistics spend estimation, automated vendor invoice auditing, and multi-warehouse SKU demand forecasting.
> - Engineered an automated **Vendor Invoice Risk Classifier** (`RandomForest` with 5-Fold `GridSearchCV`), cross-validating line-item PO receipts to flag pricing variances and prevent financial overpayments.
> - Developed a time-series SKU demand forecasting model across 4 regional distribution centers, engineering cyclical trigonometric encodings ($\sin/\cos$) and rolling lag metrics to achieve an **$R^2 \approx 0.72$**.
> - Containerized the application using **Docker** and deployed an interactive **Streamlit web portal** with automated CI/CD unit testing via GitHub Actions.

---

### Option B: Concise 4-Pointer Version (Fits on 1–2 Lines per Bullet)
> **Enterprise Supply Chain & Vendor Intelligence System** | *Python, Scikit-Learn, XGBoost, Streamlit, Docker*  
> *Live Demo:* [Streamlit App](https://mahatoaditya24-product-demand-forecasting-inferenceapp-mvyy16.streamlit.app/) | *GitHub:* [mahatoaditya24/Product-Demand-Forecasting](https://github.com/mahatoaditya24/Product-Demand-Forecasting)
> - Developed an end-to-end ML platform solving logistics spend regression, vendor invoice risk auditing, and multi-warehouse SKU demand forecasting.
> - Built an automated **Invoice Risk Classifier** cross-auditing PO dollar variances to prevent vendor overbilling and expedite straight-through approvals.
> - Formulated SKU demand models using cyclical trigonometric encodings and historical lag features, achieving an **$R^2 \approx 0.72$** and sizing 95% safety stocks.
> - Deployed an interactive **Streamlit web portal** containerized via **Docker** with automated CI/CD unit tests on **GitHub Actions**.

---

## 🔑 2. ATS Technical Keywords & Skills Checklist

- **Machine Learning & Modeling:** `Scikit-Learn`, `Random Forest`, `XGBoost Regressor`, `Linear Regression`, `GridSearchCV`, `Cross-Validation`, `F1-Score`, `MAE / RMSE / R²`
- **Time-Series & Feature Engineering:** `Cyclical Encodings (sin/cos)`, `Historical Lags (lag_1, lag_7, lag_30)`, `Rolling Means & StdDev`, `Feature Scaling (StandardScaler)`
- **Web UI & Visualization:** `Streamlit`, `Plotly`, `Matplotlib`, `Seaborn`, `Interactive CSV Uploads`
- **DevOps, Testing & CI/CD:** `Docker`, `Docker Compose`, `GitHub Actions (CI/CD)`, `PyTest / Unittest`, `Git`

---

## 🎙️ 3. Technical Interview Preparation & Talking Points

### Q1: "How did you design the Invoice Risk Flagging model?"
> **Answer:**
> *"In enterprise supply chains, invoices from 3rd-party vendors frequently contain price discrepancies or shipping delays compared to original Purchase Orders (POs). I joined line-item purchase orders with incoming invoice data in SQLite, extracting features such as dollar discrepancy $|\text{invoice\_dollars} - \text{PO\_dollars}|$, unit variance, and receiving delays. I then trained a `RandomForestClassifier` with 5-fold `GridSearchCV` optimized on $F_1\text{-score}$, allowing the finance team to automate straight-through approvals while isolating high-risk invoices for manual review."*

### Q2: "What feature engineering techniques did you use for SKU Demand Forecasting?"
> **Answer:**
> *"Demand across regional fulfillment centers exhibits both calendar seasonality and auto-regressive momentum. I engineered: (1) cyclical trigonometric features ($\sin$ and $\cos$ of month and weekday) to preserve continuous temporal cycles, (2) multi-horizon historical lags ($\text{lag}_1, \text{lag}_7, \text{lag}_{30}$) to capture immediate and monthly trend momentum, and (3) a 7-day rolling window mean and standard deviation. The rolling standard deviation is also used downstream to dynamically size a 95% service-level safety stock buffer ($Z = 1.65$)."*
